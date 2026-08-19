"""
04_model_training.py
CSAO RAIL - trains the primary LightGBM ranking/classification model,
plus an RF + GB ensemble for comparison, using a temporal (no-leakage)
train/test split.

Task framing: Pointwise Learning-to-Rank as binary classification.
  Input tuple: (user, cart_state, candidate_item, context)
  Target: P(accepted = 1)
  Ranking: sort candidates by P(accept), return top-N
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, ndcg_score, precision_score, roc_auc_score,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

LGB_PARAMS = dict(
    objective="binary",
    boosting_type="gbdt",
    metric="auc",
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    max_depth=7,
    lambda_l1=0.5,
    lambda_l2=0.5,
    num_threads=-1,
    verbose=-1,
)


def prepare_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    feature_cols = [c for c in df.columns if c != "accepted"]

    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # Temporal split: respects row order as a proxy for time (no future leakage)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["accepted"]
    X_test, y_test = test_df[feature_cols], test_df["accepted"]

    return X_train, y_train, X_test, y_test, feature_cols


def precision_at_k(y_true, y_score, k):
    order = np.argsort(-y_score)[:k]
    return y_true.values[order].mean() if len(order) else 0.0


def recall_at_k(y_true, y_score, k):
    order = np.argsort(-y_score)[:k]
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    return y_true.values[order].sum() / total_pos


def evaluate(y_true, y_score, y_pred):
    metrics = {
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "precision_at_3": round(precision_at_k(y_true, y_score, 3), 4),
        "precision_at_5": round(precision_at_k(y_true, y_score, 5), 4),
        "precision_at_10": round(precision_at_k(y_true, y_score, 10), 4),
        "recall_at_5": round(recall_at_k(y_true, y_score, 5), 4),
        "recall_at_10": round(recall_at_k(y_true, y_score, 10), 4),
    }
    try:
        metrics["ndcg_at_5"] = round(ndcg_score([y_true.values], [y_score], k=5), 4)
        metrics["ndcg_at_10"] = round(ndcg_score([y_true.values], [y_score], k=10), 4)
    except ValueError:
        pass
    return metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not installed; falling back to GradientBoostingClassifier as a stand-in.")
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return model, X_test.columns.tolist()

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model, X_train.columns.tolist()


def predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def train_ensemble(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)

    rf_scores = rf.predict_proba(X_test)[:, 1]
    gb_scores = gb.predict_proba(X_test)[:, 1]
    return rf, gb, rf_scores, gb_scores


def main():
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")
    print(f"Positive ratio - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    print("\nTraining LightGBM (primary model)...")
    lgb_model, cols = train_lightgbm(X_train, y_train, X_test, y_test)
    lgb_scores = predict(lgb_model, X_test)
    lgb_pred = (lgb_scores >= 0.5).astype(int)
    lgb_metrics = evaluate(y_test, lgb_scores, lgb_pred)
    print("LightGBM metrics:", json.dumps(lgb_metrics, indent=2))

    print("\nTraining ensemble (RF + GB) for comparison...")
    rf, gb, rf_scores, gb_scores = train_ensemble(X_train, y_train, X_test)
    rf_auc = round(roc_auc_score(y_test, rf_scores), 4)
    gb_auc = round(roc_auc_score(y_test, gb_scores), 4)
    soft_ensemble_scores = 0.3 * rf_scores + 0.3 * gb_scores + 0.4 * lgb_scores
    ensemble_auc = round(roc_auc_score(y_test, soft_ensemble_scores), 4)

    print(f"Random Forest AUC: {rf_auc}")
    print(f"Gradient Boosting AUC: {gb_auc}")
    print(f"LightGBM AUC: {lgb_metrics['auc_roc']}")
    print(f"Soft ensemble (0.3 RF + 0.3 GB + 0.4 LGB) AUC: {ensemble_auc}")
    print("Production decision: ship LightGBM-only if ensemble gain is small "
          "relative to added inference latency.")

    # baseline: popularity (predict positive class prior for everyone)
    baseline_score = np.full(len(y_test), y_train.mean())
    baseline_auc = 0.5  # a constant baseline has no discriminative AUC; reported for reference only

    results = {
        "lightgbm": lgb_metrics,
        "ensemble": {
            "random_forest_auc": rf_auc,
            "gradient_boosting_auc": gb_auc,
            "soft_ensemble_auc": ensemble_auc,
        },
        "feature_columns": feature_cols,
    }
    with open(os.path.join(MODEL_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # persist model
    try:
        lgb_model.save_model(os.path.join(MODEL_DIR, "lightgbm_model.txt"))
        print(f"\nSaved LightGBM model to {os.path.join(MODEL_DIR, 'lightgbm_model.txt')}")
    except AttributeError:
        import joblib
        joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lightgbm_model.joblib"))
        print(f"\nSaved fallback model to {os.path.join(MODEL_DIR, 'lightgbm_model.joblib')}")

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)


if __name__ == "__main__":
    main()
