"""
01_data_generation.py
CSAO RAIL - Synthetic dataset generator with real-world food delivery dynamics.

Generates:
  - 5,000 users            -> data/users.csv
  - 800 restaurants        -> data/restaurants.csv
  - ~7,900 menu items      -> data/menu_items.csv
  - 10,000 sessions        -> data/sessions.csv
  - ~90K+ CSAO events      -> data/csao_events.csv

Market-basket priors (biryani->raita 86%, tandoor->bread 88%, etc.) are
hardcoded and used to bias which add-ons get shown/accepted, so the
resulting CSAO event log carries realistic co-purchase signal for
downstream Apriori mining and model training.
"""

import numpy as np
import pandas as pd
import uuid
import os
import json

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
N_USERS = 5000
N_RESTAURANTS = 800
N_SESSIONS = 10000

CITIES = ["Mumbai", "Bangalore", "Hyderabad", "Pune", "Delhi", "Kolkata", "Chennai"]
CUISINES = [
    "North Indian", "Biryani", "Chinese", "Continental", "South Indian",
    "Fast Food", "Italian", "Mughlai", "Desserts", "Beverages",
]
USER_SEGMENTS = ["budget", "occasional", "premium"]
USER_SEGMENT_PROBS = [0.496, 0.304, 0.200]
DIETARY_PREFS = ["veg", "non_veg", "both"]
DIETARY_PROBS = [0.308, 0.398, 0.294]
PRICE_BANDS = ["budget", "moderate", "premium", "luxury"]
PRICE_BAND_PROBS = [0.30, 0.40, 0.20, 0.10]
MEAL_TIMES = ["breakfast", "lunch", "snack", "dinner", "late_night"]
MEAL_TIME_PROBS = [0.227, 0.252, 0.112, 0.267, 0.142]
DEVICE_TYPES = ["mobile", "web", "tablet"]
DEVICE_PROBS = [0.753, 0.203, 0.044]
ITEM_CATEGORIES = ["main", "side", "dessert", "beverage", "condiment"]

# Hardcoded market-basket association priors (category-level).
# key: (trigger_category, consequent_category) -> acceptance probability
MBA_PRIORS = {
    ("biryani_main", "condiment"): 0.86,   # biryani -> raita
    ("biryani_main", "beverage"): 0.92,    # biryani -> beverage
    ("biryani_main", "dessert"): 0.34,     # biryani -> dessert
    ("tandoor_main", "side"): 0.88,        # tandoor -> bread (side)
    ("tandoor_main", "beverage"): 0.75,    # tandoor -> beverage
    ("tandoor_main", "condiment"): 0.62,   # tandoor -> side/condiment
    ("main", "side"): 0.72,
    ("main", "beverage"): 0.68,
    ("main", "condiment"): 0.45,
}

PRICE_RANGES = {
    "main": (150, 400),
    "beverage": (30, 150),
    "dessert": (50, 250),
    "side": (40, 150),
    "condiment": (20, 80),
}


def gen_users(n=N_USERS):
    segments = np.random.choice(USER_SEGMENTS, size=n, p=USER_SEGMENT_PROBS)
    dietary = np.random.choice(DIETARY_PREFS, size=n, p=DIETARY_PROBS)
    cities = np.random.choice(CITIES, size=n)
    account_age = np.random.randint(30, 1826, size=n)

    # CSAO accept rate correlated with segment (premium users accept more)
    base_accept = {"budget": (0.25, 0.38), "occasional": (0.30, 0.45), "premium": (0.38, 0.55)}
    accept_rate = np.array([np.random.uniform(*base_accept[s]) for s in segments])

    order_freq = np.array([
        {"budget": np.random.uniform(1, 4), "occasional": np.random.uniform(2, 8),
         "premium": np.random.uniform(6, 20)}[s] for s in segments
    ])
    avg_order_value = np.array([
        {"budget": np.random.uniform(150, 300), "occasional": np.random.uniform(250, 500),
         "premium": np.random.uniform(400, 900)}[s] for s in segments
    ])
    preferred_price_range = np.array([
        {"budget": "budget", "occasional": "moderate", "premium": "premium"}[s] for s in segments
    ])

    df = pd.DataFrame({
        "user_id": [f"U{str(i).zfill(6)}" for i in range(n)],
        "city": cities,
        "segment": segments,
        "dietary_pref": dietary,
        "csao_acceptance_rate": accept_rate.round(3),
        "order_frequency_per_month": order_freq.round(2),
        "avg_order_value": avg_order_value.round(2),
        "preferred_price_range": preferred_price_range,
        "account_age_days": account_age,
    })
    return df


def gen_restaurants(n=N_RESTAURANTS):
    cities = np.random.choice(CITIES, size=n)
    cuisines = np.random.choice(CUISINES, size=n)
    price_band = np.random.choice(PRICE_BANDS, size=n, p=PRICE_BAND_PROBS)
    is_chain = np.random.choice([1, 0], size=n, p=[0.35, 0.65])
    rating = np.round(np.random.uniform(2.5, 4.9, size=n), 1)
    delivery_rating = np.round(np.random.uniform(4.0, 4.8, size=n), 1)
    avg_delivery_time = np.random.randint(25, 51, size=n)
    total_orders = np.random.randint(100, 5001, size=n)

    df = pd.DataFrame({
        "restaurant_id": [f"R{str(i).zfill(5)}" for i in range(n)],
        "city": cities,
        "primary_cuisine": cuisines,
        "price_band": price_band,
        "is_chain": is_chain,
        "rating": rating,
        "delivery_rating": delivery_rating,
        "avg_delivery_time_min": avg_delivery_time,
        "total_orders": total_orders,
    })
    return df


def gen_menu_items(restaurants: pd.DataFrame, target_n=7914):
    rows = []
    items_per_restaurant = max(1, target_n // len(restaurants))
    item_id = 0
    cat_weights = [0.35, 0.20, 0.15, 0.20, 0.10]  # main, side, dessert, beverage, condiment

    for _, r in restaurants.iterrows():
        n_items = np.random.poisson(items_per_restaurant)
        n_items = max(3, n_items)
        cats = np.random.choice(ITEM_CATEGORIES, size=n_items, p=cat_weights)
        for cat in cats:
            lo, hi = PRICE_RANGES[cat]
            price = np.random.randint(lo, hi + 1)
            is_veg = np.random.choice([1, 0], p=[0.55, 0.45])
            is_popular = np.random.choice([1, 0], p=[0.30, 0.70])
            item_rating = round(np.random.uniform(3.5, 4.9), 1)
            order_count = np.random.randint(10, 1001)
            rows.append({
                "item_id": f"I{str(item_id).zfill(6)}",
                "restaurant_id": r["restaurant_id"],
                "category": cat,
                "price": price,
                "is_veg": is_veg,
                "is_popular": is_popular,
                "rating": item_rating,
                "order_count": order_count,
            })
            item_id += 1

    df = pd.DataFrame(rows)
    # trim/pad to roughly match target_n
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def gen_sessions(users, restaurants, n=N_SESSIONS):
    user_ids = np.random.choice(users["user_id"], size=n)
    user_city_map = users.set_index("user_id")["city"].to_dict()
    restaurants_by_city = {c: restaurants[restaurants.city == c]["restaurant_id"].values for c in CITIES}

    rest_ids = []
    for uid in user_ids:
        city = user_city_map[uid]
        pool = restaurants_by_city.get(city)
        if pool is None or len(pool) == 0:
            pool = restaurants["restaurant_id"].values
        rest_ids.append(np.random.choice(pool))

    meal_times = np.random.choice(MEAL_TIMES, size=n, p=MEAL_TIME_PROBS)
    devices = np.random.choice(DEVICE_TYPES, size=n, p=DEVICE_PROBS)
    day_of_week = np.random.randint(0, 7, size=n)

    df = pd.DataFrame({
        "session_id": [f"S{str(i).zfill(7)}" for i in range(n)],
        "user_id": user_ids,
        "restaurant_id": rest_ids,
        "meal_time": meal_times,
        "device_type": devices,
        "day_of_week": day_of_week,
    })
    return df


def _mba_probability(main_item, cat, restaurant_cuisine):
    """Look up an acceptance probability for showing `cat` after a main item."""
    key = "biryani_main" if restaurant_cuisine == "Biryani" else (
        "tandoor_main" if restaurant_cuisine in ("North Indian", "Mughlai") else "main"
    )
    return MBA_PRIORS.get((key, cat), MBA_PRIORS.get(("main", cat), 0.4))


def gen_csao_events(sessions, restaurants, menu_items, target_n=92680):
    rest_lookup = restaurants.set_index("restaurant_id")
    items_by_rest = menu_items.groupby("restaurant_id")

    rows = []
    event_id = 0
    per_session = max(1, target_n // len(sessions))

    for _, s in sessions.iterrows():
        rest = rest_lookup.loc[s["restaurant_id"]]
        try:
            rest_items = items_by_rest.get_group(s["restaurant_id"])
        except KeyError:
            continue
        mains = rest_items[rest_items.category == "main"]
        if mains.empty:
            continue
        main_item = mains.sample(1).iloc[0]

        cart_state = [main_item["category"]]
        cart_value = main_item["price"]
        n_steps = np.random.randint(1, per_session + 2)

        for step in range(n_steps):
            candidates = rest_items[~rest_items["item_id"].isin([])]
            candidates = candidates[candidates.category != "main"]
            if candidates.empty:
                break
            candidate = candidates.sample(1).iloc[0]
            accept_prob = _mba_probability(main_item, candidate["category"], rest["primary_cuisine"])
            accepted = np.random.rand() < accept_prob
            rank_score = round(accept_prob * (0.9 ** (step // 3)), 4)

            rows.append({
                "event_id": f"E{str(event_id).zfill(7)}",
                "session_id": s["session_id"],
                "user_id": s["user_id"],
                "restaurant_id": s["restaurant_id"],
                "main_item_id": main_item["item_id"],
                "recommended_item_id": candidate["item_id"],
                "recommended_item_category": candidate["category"],
                "position_in_rail": step,
                "rank_score": rank_score,
                "cart_value_at_show": round(cart_value, 2),
                "meal_time": s["meal_time"],
                "device_type": s["device_type"],
                "accepted": int(accepted),
            })
            event_id += 1
            if accepted:
                cart_value += candidate["price"]
                cart_state.append(candidate["category"])

            if event_id >= target_n:
                break
        if event_id >= target_n:
            break

    df = pd.DataFrame(rows)
    return df


def main():
    print("Generating users...")
    users = gen_users()
    users.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)

    print("Generating restaurants...")
    restaurants = gen_restaurants()
    restaurants.to_csv(os.path.join(OUT_DIR, "restaurants.csv"), index=False)

    print("Generating menu items...")
    menu_items = gen_menu_items(restaurants)
    menu_items.to_csv(os.path.join(OUT_DIR, "menu_items.csv"), index=False)

    print("Generating sessions...")
    sessions = gen_sessions(users, restaurants)
    sessions.to_csv(os.path.join(OUT_DIR, "sessions.csv"), index=False)

    print("Generating CSAO events...")
    events = gen_csao_events(sessions, restaurants, menu_items)
    events.to_csv(os.path.join(OUT_DIR, "csao_events.csv"), index=False)

    summary = {
        "users": len(users),
        "restaurants": len(restaurants),
        "menu_items": len(menu_items),
        "sessions": len(sessions),
        "csao_events": len(events),
        "accepted_rate": round(events["accepted"].mean(), 4) if len(events) else None,
    }
    with open(os.path.join(OUT_DIR, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.", summary)


if __name__ == "__main__":
    main()
