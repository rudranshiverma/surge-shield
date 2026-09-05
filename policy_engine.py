"""Deterministic decision layer: turns a scored transaction into an action
(auto-clear, step-up, human review, urgent review). No model calls."""

from dataclasses import dataclass, field
from enum import Enum

class Action(str, Enum):
    AUTO_CLEAR = "AUTO_CLEAR"
    STEP_UP = "STEP_UP"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    URGENT_REVIEW = "URGENT_REVIEW"


PROB_AUTO_CLEAR_MAX = 0.15
PROB_HUMAN_REVIEW_MIN = 0.40
EXPOSURE_URGENT_INR = 100_000.0

# distributed_low_and_slow attacks have weak single-window evidence by design;
# this floors AUTO_CLEAR to STEP_UP when cusum is elevated, as a conservative
# safety net rather than a claim that cusum alone separates attack from festive.
CUSUM_WATCH_THRESHOLD = 60.0

# Typical values a normal window's feature should sit near
# heuristics, not learned or re-derived from a formal baseline fit.
SIGNAL_REFERENCES = {
    "decline_rate": 0.05,
    "device_concentration": 1.5,
    "ip_concentration": 1.5,
    "pct_txn_below_threshold": 0.05,
}

RESPONSE_HINTS = {
    "DECLINE_PATTERN": "High decline rate relative to typical. Consistent with probing/testing. "
                        "Suggested: throttle or hold the originating device/IP for further attempts, "
                        "not just this one transaction.",
    "FAN_OUT_DEVICE": "One device transacting across an unusually high number of distinct identities. "
                       "Suggested: entity-level hold on this device; review its other recent transactions.",
    "FAN_OUT_IP": "One network/IP transacting across an unusually high number of distinct identities. "
                  "Suggested: entity-level hold on this IP; check for a coordinated device-farm pattern.",
    "MICRO_TXN_PATTERN": "Cluster of unusually small transaction amounts -- classic testing signature. "
                          "Suggested: block further attempts below the probe-amount threshold from this source.",
    "NONE_ELEVATED": "No single signal is strongly elevated relative to typical. Probability is driven "
                      "by a combination of moderate factors -- review the full feature breakdown.",
}

@dataclass
class Decision:
    action: Action
    dominant_signal: str
    response_hint: str
    estimated_exposure_inr: float
    calibrated_prob: float
    cusum_override_triggered: bool
    watchlist_override_triggered: bool
    reasoning: list = field(default_factory=list)

def estimate_exposure_inr(avg_amount: float, txn_count: float, decline_rate: float) -> float:
    return float(avg_amount) * float(txn_count) * (1.0 - float(decline_rate))

def determine_dominant_signal(features: dict) -> str:
    candidates = {}
    for name, typical in SIGNAL_REFERENCES.items():
        value = float(features.get(name, 0.0))
        candidates[name] = value / typical if typical > 0 else 0.0

    best_name = max(candidates, key=candidates.get)
    best_ratio = candidates[best_name]

    if best_ratio < 2.0:
        return "NONE_ELEVATED"
    mapping = {
        "decline_rate": "DECLINE_PATTERN",
        "device_concentration": "FAN_OUT_DEVICE",
        "ip_concentration": "FAN_OUT_IP",
        "pct_txn_below_threshold": "MICRO_TXN_PATTERN",
    }
    return mapping[best_name]

def evaluate(features: dict, is_watchlisted: bool = False) -> Decision:
    reasoning = []
    prob = float(features["calibrated_prob"])
    exposure = estimate_exposure_inr(features["avg_amount"], features["txn_count"], features["decline_rate"])

    if prob < PROB_AUTO_CLEAR_MAX:
        action = Action.AUTO_CLEAR
        reasoning.append(f"calibrated_prob={prob:.3f} < AUTO_CLEAR_MAX({PROB_AUTO_CLEAR_MAX}) -> auto-clear")
    elif prob < PROB_HUMAN_REVIEW_MIN:
        action = Action.STEP_UP
        reasoning.append(f"{PROB_AUTO_CLEAR_MAX} <= calibrated_prob={prob:.3f} < HUMAN_REVIEW_MIN({PROB_HUMAN_REVIEW_MIN}) -> step-up verification")
    else:
        action = Action.HUMAN_REVIEW
        reasoning.append(f"calibrated_prob={prob:.3f} >= HUMAN_REVIEW_MIN({PROB_HUMAN_REVIEW_MIN}) -> human review")
        if exposure >= EXPOSURE_URGENT_INR:
            action = Action.URGENT_REVIEW
            reasoning.append(f"estimated_exposure=Rs.{exposure:,.0f} >= URGENT threshold(Rs.{EXPOSURE_URGENT_INR:,.0f}) -> escalated to urgent")

    watchlist_override = False
    if is_watchlisted and action == Action.AUTO_CLEAR:
        action = Action.STEP_UP
        watchlist_override = True
        reasoning.append("OVERRIDE: slice is on the analyst watchlist -- not auto-clearing until re-reviewed.")

    cusum_override = False
    cusum = float(features.get("cusum_statistic", 0.0))
    if action == Action.AUTO_CLEAR and cusum >= CUSUM_WATCH_THRESHOLD:
        action = Action.STEP_UP
        cusum_override = True
        reasoning.append(f"OVERRIDE: cusum_statistic={cusum:.1f} >= watch threshold({CUSUM_WATCH_THRESHOLD}) "
                          f"despite low single-window probability -- not auto-clearing on this signal alone.")

    dominant_signal = determine_dominant_signal(features)
    reasoning.append(f"dominant_signal={dominant_signal}")

    return Decision(
        action=action,
        dominant_signal=dominant_signal,
        response_hint=RESPONSE_HINTS[dominant_signal],
        estimated_exposure_inr=exposure,
        calibrated_prob=prob,
        cusum_override_triggered=cusum_override,
        watchlist_override_triggered=watchlist_override,
        reasoning=reasoning,
    )

if __name__ == "__main__":
    cases = [
        dict(name="clearly normal", calibrated_prob=0.03, avg_amount=800, txn_count=5, decline_rate=0.02,
             device_concentration=1.1, ip_concentration=1.2, pct_txn_below_threshold=0.0, cusum_statistic=2.0),
        dict(name="ambiguous", calibrated_prob=0.25, avg_amount=800, txn_count=5, decline_rate=0.10,
             device_concentration=1.5, ip_concentration=1.5, pct_txn_below_threshold=0.02, cusum_statistic=10.0),
        dict(name="clear fan-out attack, high exposure", calibrated_prob=0.85, avg_amount=15000, txn_count=40,
             decline_rate=0.12, device_concentration=12.0, ip_concentration=2.0, pct_txn_below_threshold=0.1,
             cusum_statistic=90.0),
        dict(name="low prob but high cusum", calibrated_prob=0.08, avg_amount=500,
             txn_count=6, decline_rate=0.15, device_concentration=1.8, ip_concentration=1.6,
             pct_txn_below_threshold=0.05, cusum_statistic=75.0),
        dict(name="low prob, watchlisted slice", calibrated_prob=0.05, avg_amount=500,
             txn_count=6, decline_rate=0.02, device_concentration=1.1, ip_concentration=1.2,
             pct_txn_below_threshold=0.0, cusum_statistic=2.0, is_watchlisted=True),
    ]
    for c in cases:
        name = c.pop("name")
        watchlisted = c.pop("is_watchlisted", False)
        d = evaluate(c, is_watchlisted=watchlisted)
        print(f"\n[{name}]")
        print(f"  action={d.action.value}  dominant_signal={d.dominant_signal}  "
              f"exposure=Rs.{d.estimated_exposure_inr:,.0f}  cusum_override={d.cusum_override_triggered}  "
              f"watchlist_override={d.watchlist_override_triggered}")
        for r in d.reasoning:
            print(f"    - {r}")