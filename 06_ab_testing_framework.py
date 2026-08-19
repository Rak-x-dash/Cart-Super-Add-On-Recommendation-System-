"""
06_ab_testing_framework.py
CSAO RAIL - statistical framework for validating the CSAO Rail model
against control in a live A/B test.

Test design:
  alpha = 0.05, min sample = 100,000 sessions/arm, MDE = 5% relative lift,
  duration ~7 days @ 10% traffic, deterministic user-hash bucketing,
  Bonferroni correction across the metric family.
"""

import json
import math
import os

import numpy as np
from scipy import stats

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

ALPHA = 0.05
MIN_SAMPLE_PER_ARM = 100_000
MDE = 0.05


def bonferroni_alpha(num_tests, alpha=ALPHA):
    return alpha / num_tests


def welch_t_test(control_values, treatment_values):
    t_stat, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False)
    pooled_sd = np.sqrt((np.var(control_values, ddof=1) + np.var(treatment_values, ddof=1)) / 2)
    cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_sd if pooled_sd > 0 else 0.0
    return {
        "test": "welch_t_test",
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(float(cohens_d), 4),
        "control_mean": round(float(np.mean(control_values)), 4),
        "treatment_mean": round(float(np.mean(treatment_values)), 4),
    }


def z_test_proportions(control_successes, control_n, treatment_successes, treatment_n):
    p1 = control_successes / control_n
    p2 = treatment_successes / treatment_n
    p_pool = (control_successes + treatment_successes) / (control_n + treatment_n)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low, ci_high = (p2 - p1) - 1.96 * se, (p2 - p1) + 1.96 * se
    return {
        "test": "z_test_proportions",
        "z_stat": round(z, 4),
        "p_value": round(p_value, 6),
        "control_rate": round(p1, 4),
        "treatment_rate": round(p2, 4),
        "diff_ci_95": [round(ci_low, 4), round(ci_high, 4)],
    }


def decide(primary_p_value, guardrail_ok, alpha_corrected):
    if primary_p_value < alpha_corrected and guardrail_ok:
        return "DEPLOY"
    if primary_p_value < alpha_corrected and not guardrail_ok:
        return "INVESTIGATE"
    return "KEEP_CONTROL"


def run_ab_test(control_n=100_000, treatment_n=100_000, seed=42):
    """
    Simulates an A/B test comparing the CSAO Rail model (treatment) against
    the current popularity-based rail (control), using the same directional
    effect sizes referenced in the project deck. Replace the simulated
    arrays with real logged metrics per session in production.
    """
    rng = np.random.default_rng(seed)

    # AOV (continuous) - control ~N(310, 90), treatment ~N(418, 100)
    control_aov = rng.normal(310, 90, control_n)
    treatment_aov = rng.normal(418, 100, treatment_n)
    aov_result = welch_t_test(control_aov, treatment_aov)

    # CSAO acceptance rate (binary/proportion)
    control_accept = int(0.22 * control_n)
    treatment_accept = int(0.41 * treatment_n)
    accept_result = z_test_proportions(control_accept, control_n, treatment_accept, treatment_n)

    # Cart-to-order ratio (guardrail, continuous)
    control_c2o = rng.normal(2.9, 0.6, control_n)
    treatment_c2o = rng.normal(3.4, 0.6, treatment_n)
    c2o_result = welch_t_test(control_c2o, treatment_c2o)
    guardrail_ok = c2o_result["treatment_mean"] >= c2o_result["control_mean"]

    # CTR (proportion, secondary)
    control_ctr = int(0.15 * control_n)
    treatment_ctr = int(0.18 * treatment_n)
    ctr_result = z_test_proportions(control_ctr, control_n, treatment_ctr, treatment_n)

    num_tests = 4
    alpha_corrected = bonferroni_alpha(num_tests)

    decision = decide(accept_result["p_value"], guardrail_ok, alpha_corrected)

    report = {
        "config": {
            "alpha": ALPHA,
            "bonferroni_corrected_alpha": round(alpha_corrected, 6),
            "min_sample_per_arm": MIN_SAMPLE_PER_ARM,
            "mde": MDE,
            "control_n": control_n,
            "treatment_n": treatment_n,
        },
        "metrics": {
            "average_order_value": aov_result,
            "csao_acceptance_rate": accept_result,
            "cart_to_order_ratio_guardrail": c2o_result,
            "click_through_rate": ctr_result,
        },
        "guardrail_ok": guardrail_ok,
        "decision": decision,
    }
    return report


def main():
    report = run_ab_test()
    out_path = os.path.join(REPORT_DIR, "ab_test_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
