"""
05_inference_api.py
CSAO RAIL - Flask inference API.

Endpoints:
  POST /recommend       -> cart JSON in, ranked add-ons + latency breakdown out
  GET  /health           -> liveness probe
  GET  /metrics          -> basic counters (requests, avg latency, cache hits)
  POST /reload-model     -> hot-reload the trained model without downtime

Uses an in-process dict as a stand-in for Redis (swap `Cache` for a real
redis.Redis client in production - the interface is identical).
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
from flask import Flask, jsonify, request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

app = Flask(__name__)


class Cache:
    """Drop-in stand-in for a Redis client (1-hour TTL semantics simplified)."""
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


cache = Cache()
METRICS = defaultdict(float)
METRICS["requests"] = 0
METRICS["cache_hits"] = 0
METRICS["total_latency_ms"] = 0.0

MODEL = {"booster": None, "feature_cols": []}
RULES = []


def load_model():
    try:
        import lightgbm as lgb
        MODEL["booster"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_model.txt"))
    except Exception as e:
        print(f"Could not load LightGBM model ({e}); inference will use fallback ranking only.")
        MODEL["booster"] = None

    fc_path = os.path.join(MODEL_DIR, "feature_columns.json")
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            MODEL["feature_cols"] = json.load(f)

    global RULES
    rules_path = os.path.join(DATA_DIR, "association_rules.json")
    if os.path.exists(rules_path):
        with open(rules_path) as f:
            RULES = json.load(f)


def get_user_features(user_id):
    cached = cache.get(f"user:{user_id}")
    if cached:
        METRICS["cache_hits"] += 1
        return cached
    # Fallback synthetic defaults (in prod: DB/feature-store lookup)
    features = {
        "user_segment": 2, "user_csao_acceptance_rate": 0.35,
        "user_order_frequency": 4.0, "user_avg_order_value": 350.0,
        "user_account_age_days": 400, "user_dietary_pref": 2,
        "user_preferred_price_range": 2, "user_city": 0,
    }
    cache.set(f"user:{user_id}", features)
    return features


def cold_start_fallback(cart_items, top_k=3):
    """3-step fallback chain: association rules -> city popularity -> restaurant bestsellers."""
    categories = [i.get("category", "main") for i in cart_items] or ["main"]
    if RULES:
        cart_set = set(categories)
        matches = [r for r in RULES if set(r["antecedent"]).issubset(cart_set)]
        matches.sort(key=lambda r: r["confidence"], reverse=True)
        recs = []
        seen = set(categories)
        for r in matches:
            for cat in r["consequent"]:
                if cat not in seen:
                    seen.add(cat)
                    recs.append({"category": cat, "score": r["confidence"], "reason": "association_rule"})
        if recs:
            return recs[:top_k]

    # Step 2/3 generic fallback
    default_order = ["beverage", "side", "dessert", "condiment"]
    recs = [{"category": c, "score": 0.3, "reason": "popularity_fallback"} for c in default_order if c not in categories]
    return recs[:top_k]


def score_candidates(user_id, cart_items, candidates, meal_time="lunch", device_type="mobile"):
    """Score each candidate item with the LightGBM model if available, else use rank_score."""
    booster = MODEL["booster"]
    if booster is None or not MODEL["feature_cols"]:
        for c in candidates:
            c["score"] = c.get("rank_score", 0.5)
        return sorted(candidates, key=lambda c: c["score"], reverse=True)

    user_feats = get_user_features(user_id)
    rows = []
    for c in candidates:
        row = dict(user_feats)
        row.update({
            "restaurant_rating": 4.0, "restaurant_is_chain": 0, "restaurant_delivery_rating": 4.3,
            "restaurant_avg_delivery_time": 35, "restaurant_total_orders": 1000,
            "restaurant_price_band": 2, "restaurant_primary_cuisine": 0,
            "recommended_item_price": c.get("price", 100), "recommended_item_is_veg": 1,
            "recommended_item_is_popular": 1, "recommended_item_rating": 4.2,
            "recommended_item_order_count": 200,
            "cart_item_price": cart_items[0].get("price", 200) if cart_items else 200,
            "recommendation_position": c.get("position", 0), "rank_score": c.get("rank_score", 0.5),
            "cart_total_value": sum(i.get("price", 0) for i in cart_items),
            "meal_time_code": {"breakfast": 1, "lunch": 2, "snack": 3, "dinner": 4, "late_night": 5}.get(meal_time, 2),
            "hour_of_day": 13, "day_of_week": 2, "is_weekend": 0,
            "device_type_code": {"mobile": 1, "web": 2, "tablet": 3}.get(device_type, 1),
            "user_item_category_affinity": 1, "price_segment_match": 1,
            "meal_category_fit": 1, "position_score": 0.9 ** (c.get("position", 0) // 3),
            "main_item_id_code": 0,
        })
        rows.append([row.get(col, 0) for col in MODEL["feature_cols"]])

    scores = booster.predict(np.array(rows))
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


@app.route("/recommend", methods=["POST"])
def recommend():
    t0 = time.perf_counter()
    payload = request.get_json(force=True) or {}
    user_id = payload.get("user_id", "anonymous")
    cart_items = payload.get("cart_items", [])
    candidates = payload.get("candidates", [])
    meal_time = payload.get("meal_time", "lunch")
    device_type = payload.get("device_type", "mobile")
    top_k = int(payload.get("top_k", 5))

    t_feat0 = time.perf_counter()
    _ = get_user_features(user_id)
    t_feat1 = time.perf_counter()

    if candidates:
        ranked = score_candidates(user_id, cart_items, candidates, meal_time, device_type)[:top_k]
        source = "model"
    else:
        ranked = cold_start_fallback(cart_items, top_k)
        source = "cold_start_fallback"

    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000, 2)

    METRICS["requests"] += 1
    METRICS["total_latency_ms"] += latency_ms

    return jsonify({
        "user_id": user_id,
        "recommendations": ranked,
        "source": source,
        "latency_breakdown_ms": {
            "feature_retrieval": round((t_feat1 - t_feat0) * 1000, 2),
            "total": latency_ms,
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL["booster"] is not None,
        "rules_loaded": len(RULES) > 0,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    n = max(1, int(METRICS["requests"]))
    return jsonify({
        "requests": int(METRICS["requests"]),
        "cache_hits": int(METRICS["cache_hits"]),
        "cache_hit_rate": round(METRICS["cache_hits"] / n, 4),
        "avg_latency_ms": round(METRICS["total_latency_ms"] / n, 2),
    })


@app.route("/reload-model", methods=["POST"])
def reload_model():
    load_model()
    return jsonify({"status": "reloaded", "model_loaded": MODEL["booster"] is not None})


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
