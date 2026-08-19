"""
02_market_basket_analysis.py
CSAO RAIL - Apriori algorithm + association rule mining over accepted
CSAO interactions, grouped into session-level transactions.

Config:
  min_support    = 0.02
  min_confidence = 0.30
  min_lift       = 1.0
  max_itemset_size = 3
"""

import json
import os
from itertools import combinations

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "association_rules.json")

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.0
MAX_ITEMSET_SIZE = 3
RANDOM_STATE = 42


def load_transactions():
    """Build session -> set(item categories) transactions from accepted events."""
    events = pd.read_csv(os.path.join(DATA_DIR, "csao_events.csv"))
    accepted = events[events["accepted"] == 1].copy()

    # main item's category is always 'main'; append recommended categories
    tx = accepted.groupby("session_id")["recommended_item_category"].apply(
        lambda cats: frozenset(list(cats) + ["main"])
    )
    return list(tx.values)


def support(itemset, transactions):
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)


def apriori_gen(prev_freq_itemsets, k):
    """Generate candidate itemsets of size k from frequent itemsets of size k-1."""
    items = sorted({i for itemset in prev_freq_itemsets for i in itemset})
    candidates = set()
    for combo in combinations(items, k):
        candidate = frozenset(combo)
        # prune: all (k-1)-subsets must be frequent
        if all(frozenset(sub) in prev_freq_itemsets for sub in combinations(combo, k - 1)):
            candidates.add(candidate)
    return candidates


def run_apriori(transactions):
    n = len(transactions)
    all_items = sorted({i for t in transactions for i in t})

    # 1-itemsets
    freq_itemsets = {}
    current = {}
    for item in all_items:
        s = support(frozenset([item]), transactions)
        if s >= MIN_SUPPORT:
            current[frozenset([item])] = s
    freq_itemsets[1] = current

    k = 2
    while current and k <= MAX_ITEMSET_SIZE:
        candidates = apriori_gen(set(current.keys()), k)
        next_level = {}
        for c in candidates:
            s = support(c, transactions)
            if s >= MIN_SUPPORT:
                next_level[c] = s
        if not next_level:
            break
        freq_itemsets[k] = next_level
        current = next_level
        k += 1

    return freq_itemsets


def mine_rules(freq_itemsets, transactions):
    support_lookup = {}
    for level in freq_itemsets.values():
        support_lookup.update(level)

    rules = []
    for k, level in freq_itemsets.items():
        if k < 2:
            continue
        for itemset, itemset_support in level.items():
            items = list(itemset)
            for r in range(1, len(items)):
                for antecedent in combinations(items, r):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if antecedent not in support_lookup or not consequent:
                        continue
                    ant_support = support_lookup[antecedent]
                    cons_support = support(consequent, transactions)
                    if ant_support == 0 or cons_support == 0:
                        continue
                    confidence = itemset_support / ant_support
                    lift = confidence / cons_support
                    if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                        rules.append({
                            "antecedent": sorted(antecedent),
                            "consequent": sorted(consequent),
                            "support": round(itemset_support, 4),
                            "confidence": round(confidence, 4),
                            "lift": round(lift, 4),
                        })
    rules.sort(key=lambda r: r["confidence"], reverse=True)
    return rules


def get_recommendations(cart_categories, rules, top_k=5):
    """
    cart_categories: list of item categories currently in the cart (e.g. ['main'])
    Returns top-K recommended categories ranked by confidence, deduplicated.
    """
    cart_set = set(cart_categories)
    matches = []
    for rule in rules:
        if set(rule["antecedent"]).issubset(cart_set):
            matches.append(rule)

    seen = set()
    recs = []
    for rule in sorted(matches, key=lambda r: r["confidence"], reverse=True):
        for cat in rule["consequent"]:
            if cat not in cart_set and cat not in seen:
                seen.add(cat)
                recs.append({
                    "category": cat,
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                    "reason": f"Goes well with {', '.join(rule['antecedent'])}",
                })
    return recs[:top_k]


def main():
    transactions = load_transactions()
    print(f"Transactions: {len(transactions)}")

    freq_itemsets = run_apriori(transactions)
    for k, level in freq_itemsets.items():
        print(f"Frequent {k}-itemsets: {len(level)}")

    rules = mine_rules(freq_itemsets, transactions)
    print(f"Total rules generated: {len(rules)}")
    if rules:
        avg_conf = sum(r["confidence"] for r in rules) / len(rules)
        max_lift = max(r["lift"] for r in rules)
        print(f"Avg confidence: {avg_conf:.2f} | Max lift: {max_lift:.2f}")

    with open(OUT_PATH, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"Saved rules to {OUT_PATH}")

    # demo
    demo = get_recommendations(["main"], rules)
    print("Sample recommendations for cart=['main']:", demo)


if __name__ == "__main__":
    main()
