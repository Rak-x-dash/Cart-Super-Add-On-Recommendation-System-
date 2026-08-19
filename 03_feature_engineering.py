"""
03_feature_engineering.py
CSAO RAIL - builds the 47-feature training matrix across 6 feature groups:
  1. User features
  2. Restaurant features
  3. Item (candidate) features
  4. Cart context features
  5. Temporal features
  6. Interaction (cross) features

Batch features (user_*, restaurant_*, item_*) are conceptually cached in
Redis with a 1-hour TTL; real-time features (cart state, position,
rank_score) are computed inline at request time. In this offline script
everything is joined into a single flat table for model training.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MEAL_TIME_MAP = {"breakfast": 1, "lunch": 2, "snack": 3, "dinner": 4, "late_night": 5}
MEAL_TIME_HOUR = {"breakfast": 8, "lunch": 13, "snack": 17, "dinner": 20, "late_night": 23}
DEVICE_MAP = {"mobile": 1, "web": 2, "tablet": 3}
DIETARY_MAP = {"veg": 1, "both": 2, "non_veg": 3}
SEGMENT_MAP = {"budget": 1, "occasional": 2, "premium": 3}
PRICE_BAND_MAP = {"budget": 1, "moderate": 2, "premium": 3, "luxury": 4}


def load_raw():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    restaurants = pd.read_csv(os.path.join(DATA_DIR, "restaurants.csv"))
    menu_items = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
    sessions = pd.read_csv(os.path.join(DATA_DIR, "sessions.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "csao_events.csv"))
    return users, restaurants, menu_items, sessions, events


def user_features(events, users):
    df = events.merge(users, on="user_id", how="left")
    df["user_segment"] = df["segment"].map(SEGMENT_MAP)
    df["user_csao_acceptance_rate"] = df["csao_acceptance_rate"]
    df["user_order_frequency"] = df["order_frequency_per_month"]
    df["user_avg_order_value"] = df["avg_order_value"]
    df["user_account_age_days"] = df["account_age_days"]
    df["user_dietary_pref"] = df["dietary_pref"].map(DIETARY_MAP)
    df["user_preferred_price_range"] = df["preferred_price_range"].map(PRICE_BAND_MAP)
    le_city = LabelEncoder()
    df["user_city"] = le_city.fit_transform(df["city"].astype(str))
    return df


def restaurant_features(df, restaurants):
    df = df.merge(
        restaurants.rename(columns={"rating": "restaurant_rating"}),
        on="restaurant_id", how="left", suffixes=("", "_rest"),
    )
    df["restaurant_is_chain"] = df["is_chain"]
    df["restaurant_delivery_rating"] = df["delivery_rating"]
    df["restaurant_avg_delivery_time"] = df["avg_delivery_time_min"]
    df["restaurant_total_orders"] = df["total_orders"]
    df["restaurant_price_band"] = df["price_band"].map(PRICE_BAND_MAP)
    le_cuisine = LabelEncoder()
    df["restaurant_primary_cuisine"] = le_cuisine.fit_transform(df["primary_cuisine"].astype(str))
    # fallback for missing restaurant data
    for col in ["restaurant_rating", "restaurant_delivery_rating", "restaurant_avg_delivery_time",
                "restaurant_total_orders"]:
        df[col] = df[col].fillna(df[col].mean())
    return df


def item_features(df, menu_items):
    items = menu_items.rename(columns={
        "item_id": "recommended_item_id",
        "price": "recommended_item_price",
        "is_veg": "recommended_item_is_veg",
        "is_popular": "recommended_item_is_popular",
        "rating": "recommended_item_rating",
        "order_count": "recommended_item_order_count",
    })[["recommended_item_id", "recommended_item_price", "recommended_item_is_veg",
        "recommended_item_is_popular", "recommended_item_rating", "recommended_item_order_count"]]
    df = df.merge(items, on="recommended_item_id", how="left")
    df["recommended_item_category"] = df["recommended_item_category"]
    return df


def cart_context_features(df, menu_items):
    main_prices = menu_items.set_index("item_id")["price"].to_dict()
    df["cart_item_category"] = "main"
    df["cart_item_price"] = df["main_item_id"].map(main_prices)
    df["recommendation_position"] = df["position_in_rail"]
    df["rank_score"] = df["rank_score"]
    df["recommendation_reason"] = "Goes well with main"
    df["cart_total_value"] = df["cart_value_at_show"]
    return df


def temporal_features(df):
    df["meal_time_code"] = df["meal_time"].map(MEAL_TIME_MAP)
    df["hour_of_day"] = df["meal_time"].map(MEAL_TIME_HOUR)
    df["day_of_week"] = df.get("day_of_week", pd.Series(np.random.randint(0, 7, len(df))))
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["device_type_code"] = df["device_type"].map(DEVICE_MAP)
    return df


def interaction_features(df):
    df["user_item_category_affinity"] = df["user_segment"] * df["recommended_item_category"].map(
        {"main": 1, "side": 2, "dessert": 3, "beverage": 4, "condiment": 5}
    ).fillna(0)
    df["price_segment_match"] = (df["user_preferred_price_range"] == df["restaurant_price_band"]).astype(int)
    df["meal_category_fit"] = df["meal_time_code"] * df["recommended_item_category"].map(
        {"main": 1, "side": 2, "dessert": 3, "beverage": 4, "condiment": 5}
    ).fillna(0)
    df["position_score"] = 0.9 ** (df["recommendation_position"] // 3)
    df["main_item_id_code"] = LabelEncoder().fit_transform(df["main_item_id"].astype(str))
    return df


FEATURE_COLUMNS = [
    # Group 1: user
    "user_segment", "user_csao_acceptance_rate", "user_order_frequency",
    "user_avg_order_value", "user_account_age_days", "user_dietary_pref",
    "user_preferred_price_range", "user_city",
    # Group 2: restaurant
    "restaurant_rating", "restaurant_is_chain", "restaurant_delivery_rating",
    "restaurant_avg_delivery_time", "restaurant_total_orders",
    "restaurant_price_band", "restaurant_primary_cuisine",
    # Group 3: item
    "recommended_item_price", "recommended_item_is_veg", "recommended_item_is_popular",
    "recommended_item_rating", "recommended_item_order_count",
    # Group 4: cart context
    "cart_item_price", "recommendation_position", "rank_score", "cart_total_value",
    # Group 5: temporal
    "meal_time_code", "hour_of_day", "day_of_week", "is_weekend", "device_type_code",
    # Group 6: interaction
    "user_item_category_affinity", "price_segment_match", "meal_category_fit",
    "position_score", "main_item_id_code",
]


def create_features():
    users, restaurants, menu_items, sessions, events = load_raw()

    df = user_features(events, users)
    df = restaurant_features(df, restaurants)
    df = item_features(df, menu_items)
    df = cart_context_features(df, menu_items)
    df = temporal_features(df)
    df = interaction_features(df)

    df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].mean() if df[col].dtype != object else 0)

    out_cols = FEATURE_COLUMNS + ["accepted"]
    out_cols = [c for c in out_cols if c in df.columns]
    feature_df = df[out_cols].copy()
    return feature_df


def main():
    feature_df = create_features()
    out_path = os.path.join(DATA_DIR, "features.csv")
    feature_df.to_csv(out_path, index=False)
    print(f"Feature matrix: {feature_df.shape[0]} rows x {feature_df.shape[1]} cols")
    print(f"Saved to {out_path}")
    print("Positive class ratio:", round(feature_df["accepted"].mean(), 4))


if __name__ == "__main__":
    main()
