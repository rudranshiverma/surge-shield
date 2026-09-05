"""
generate_dataset.py
Generates synthetic UPI e-commerce transaction data for training a festive-surge-vs-attack fraud detector. No public dataset records
UPI-native fields (VPA identity, PSP handle, collect vs pay, device/IP fan-out) or real festival-timed fraud, so both are simulated here,
calibrated to published aggregate patterns (NPCI festive volume, known card-testing/ATO seasonality).

Design choices:
- New-customer rate rises during real festive windows, so VPA novelty
  alone can't separate attack from legitimate growth.
- Festive windows get infra-driven decline codes (timeouts) distinct
  from attack-driven ones (invalid VPA), so decline rate alone isn't
  an attack signal either.
- Regular customers in a slice share IP blocks (~8 per block), modeling
  CGNAT, so IP concentration is a weaker signal than device concentration.
- Attack intensity varies continuously (0.3-1.0) so attacks aren't all
  maximally separable from normal traffic.
- The seasonal baseline is computed walk-forward from history only; the
  generator's true multipliers are never exposed to the feature table.
- Train/test split is by slice and by festival identity (three festivals
  held out entirely), never by row shuffle.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# CONFIG
SIM_START = datetime(2026, 7, 15)
WINDOW_MINUTES = 15
WINDOWS_PER_DAY = (24*60)//WINDOW_MINUTES  

CATEGORIES = ["electronics", "fashion", "grocery", "general"]
REGIONS = ["north", "south", "east", "west", "central", "northeast"]

PSP_HANDLES = {
    "@okhdfcbank": 0.16, "@ybl": 0.24, "@paytm": 0.16,
    "@okaxis": 0.12, "@ibl": 0.10, "@oksbi": 0.12, "@axl": 0.10,
}

TEST_AMOUNT_THRESHOLD = 10.0   # "probe-sized" transactions
BASE_DECLINE_RATE = 0.03       # ordinary day-to-day decline rate

# category -> (lognormal mu, sigma) for transaction amount in INR
CATEGORY_AMOUNT_PARAMS = {
    "electronics": (8.6, 0.55),   
    "fashion":(7.0, 0.6),    
    "grocery":(6.0, 0.5),    
    "general":(6.6, 0.6),    
}

# Festival calendar: each entry ramps up to peak_day then decays.
FESTIVALS = [
    # TRAIN festivals
    {"name": "Independence Day Sale", "peak_day": (datetime(2026, 8, 15) - SIM_START).days,
     "ramp_days": 2, "decay_days": 1, "is_test_festival": False,
     "category_mult": {"electronics": 3.2, "fashion": 2.4, "grocery": 1.1, "general": 1.8}},
    {"name": "Raksha Bandhan", "peak_day": (datetime(2026, 8, 28) - SIM_START).days,
     "ramp_days": 1, "decay_days": 1, "is_test_festival": False,
     "category_mult": {"electronics": 1.8, "fashion": 3.0, "grocery": 1.2, "general": 1.5}},
    {"name": "Diwali Mega Sale", "peak_day": (datetime(2026, 11, 8) - SIM_START).days,
     "ramp_days": 4, "decay_days": 2, "is_test_festival": False,
     "category_mult": {"electronics": 6.0, "fashion": 4.2, "grocery": 1.3, "general": 2.6}},
    # TEST festivals (never seen in training)
    {"name": "New Year's Day Sale", "peak_day": (datetime(2027, 1, 1) - SIM_START).days,
     "ramp_days": 1, "decay_days": 1, "is_test_festival": True,
     "category_mult": {"electronics": 2.0, "fashion": 1.8, "grocery": 1.1, "general": 1.6}},
    {"name": "Republic Day Sale", "peak_day": (datetime(2027, 1, 26) - SIM_START).days,
     "ramp_days": 2, "decay_days": 1, "is_test_festival": True,
     "category_mult": {"electronics": 2.5, "fashion": 1.6, "grocery": 1.1, "general": 1.7}},
    {"name": "Holi Sale", "peak_day": (datetime(2027, 3, 22) - SIM_START).days,
     "ramp_days": 1, "decay_days": 1, "is_test_festival": True,
     "category_mult": {"electronics": 1.3, "fashion": 2.2, "grocery": 1.2, "general": 1.5}},
]
TEST_FESTIVAL_NAMES = {f["name"] for f in FESTIVALS if f.get("is_test_festival")}

ATTACK_FAMILIES = [
    "sudden_burst", "gradual_ramp", "low_value_testing",
    "device_to_many_vpas", "ip_to_many_vpas",
    "high_decline_campaign", "distributed_low_and_slow", "festive_attack",
]

# (min, max) ranges scaled by each episode's intensity in [0,1]
FAMILY_RANGES = {
    "sudden_burst":dict(duration=(1, 4),volume_mult=(3, 14),decline=(0.30, 0.65), conc=(0.5, 0.95)),
    "gradual_ramp":dict(duration=(8, 20),volume_mult=(2, 6),decline=(0.15, 0.40), conc=(0.3, 0.75)),
    "low_value_testing":dict(duration=(4, 12),volume_mult=(2, 8),decline=(0.45, 0.80), conc=(0.6, 0.95)),
    "device_to_many_vpas":dict(duration=(3, 10),volume_mult=(2, 7),decline=(0.25, 0.55), conc=(0.7, 0.98)),
    "ip_to_many_vpas":dict(duration=(3, 10),  volume_mult=(2, 7),decline=(0.20, 0.50), conc=(0.6, 0.95)),
    "high_decline_campaign":dict(duration=(4, 8),volume_mult=(1.5, 5), decline=(0.55, 0.90), conc=(0.3, 0.7)),
    "distributed_low_and_slow": dict(duration=(40, 80), volume_mult=(1.2, 1.8), decline=(0.12, 0.25), conc=(0.2, 0.5)),
    "festive_attack":dict(duration=(2, 8),volume_mult=(2, 6),decline=(0.25, 0.55), conc=(0.4, 0.85)),
}

HOURLY_SHAPE = np.array([
    0.20, 0.15, 0.10, 0.10, 0.10, 0.15, 0.30, 0.50, 0.70, 0.90, 1.10, 1.30,
    1.40, 1.30, 1.20, 1.10, 1.20, 1.40, 1.60, 1.70, 1.50, 1.10, 0.60, 0.30,
])
HOURLY_SHAPE = HOURLY_SHAPE / HOURLY_SHAPE.mean()

def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]

def dow_multiplier(dow: int) -> float:
    return 1.15 if dow >= 5 else 1.0

def festival_multiplier(day_offset: float, category: str):
    mult, name, phase = 1.0, None, None
    for f in FESTIVALS:
        start = f["peak_day"] - f["ramp_days"]
        end = f["peak_day"] + f["decay_days"]
        if start <= day_offset <= end:
            if day_offset <= f["peak_day"]:
                frac = (day_offset - start) / max(f["ramp_days"], 1)
            else:
                frac = 1 - (day_offset - f["peak_day"]) / max(f["decay_days"], 1)
            frac = float(np.clip(frac, 0, 1))
            m = f["category_mult"][category]
            effective = 1 + (m - 1) * frac
            if effective > mult:
                mult, name = effective, f["name"]
                if day_offset == f["peak_day"]:
                    phase = "peak"
                elif day_offset < f["peak_day"]:
                    phase = "ramp"
                else:
                    phase = "decay"
    return mult, name, phase

# SLICES (merchant_id x geo_region) AND CUSTOMER POOLS
def build_slices(n_slices: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n_slices):
        category = CATEGORIES[i % len(CATEGORIES)]
        region = REGIONS[rng.integers(len(REGIONS))]
        merchant_id = f"M{i:04d}"
        slice_id = f"{merchant_id}_{region}"
        # baseline avg transactions per 15-min window at an "average" hour
        size_factor = rng.lognormal(mean=1.0, sigma=0.5)
        base_lambda = float(np.clip(size_factor, 0.5, 12.0))
        rows.append(dict(slice_id=slice_id, merchant_id=merchant_id,
                          geo_region=region, category=category,
                          base_lambda=base_lambda))
    return pd.DataFrame(rows)

def build_customer_pool(slice_id: str, n: int) -> list:
    pool = []
    for i in range(n):
        pool.append(dict(
            vpa=h(f"vpa-{slice_id}-{i}"),
            device=h(f"dev-{slice_id}-{i}"),
            ip=h(f"ip-{slice_id}-{i // 8}"),  
        ))
    return pool

def new_customer(slice_id: str, counter: int) -> dict:
    return dict(
        vpa=h(f"newvpa-{slice_id}-{counter}"),
        device=h(f"newdev-{slice_id}-{counter}"),
        ip=h(f"newip-{slice_id}-{counter // 5}"),
    )

# EPISODE GENERATION (attack campaigns + standalone non-calendar flash sales)
def build_episodes(slices_df: pd.DataFrame, sim_days: int,
                    n_attack_episodes: int, n_flash_sales: int,
                    rng: np.random.Generator) -> list:
    episodes = []
    eid_counter = 0
    slice_ids = slices_df["slice_id"].tolist()
    slice_category = dict(zip(slices_df["slice_id"], slices_df["category"]))

    #attack episodes
    for _ in range(n_attack_episodes):
        family = ATTACK_FAMILIES[rng.integers(len(ATTACK_FAMILIES))]
        ranges = FAMILY_RANGES[family]
        slice_id = slice_ids[rng.integers(len(slice_ids))]
        intensity = float(rng.uniform(0.3, 1.0))

        def scale(rng_pair):
            lo, hi = rng_pair
            return lo + (hi-lo) * intensity

        duration = int(rng.integers(ranges["duration"][0], ranges["duration"][1] + 1))

        if family == "festive_attack":
            category = slice_category[slice_id]
            f = FESTIVALS[rng.integers(len(FESTIVALS))]
            start_day = f["peak_day"]
            for _try in range(15):
                cand = f["peak_day"] + int(rng.integers(-f["ramp_days"], f["decay_days"] + 1))
                cand = max(cand, 0)
                mult, _name, _phase = festival_multiplier(cand, category)
                if mult > 1.05:
                    start_day = cand
                    break
        else:
            start_day = int(rng.integers(0, max(sim_days - 1, 1)))

        start_window_idx = int(rng.integers(0, WINDOWS_PER_DAY))
        start_ts = SIM_START + timedelta(days=start_day, minutes=start_window_idx * WINDOW_MINUTES)
        end_ts = start_ts + timedelta(minutes=duration * WINDOW_MINUTES)

        n_atk_devices = max(1, int(round(2 + 8 * (1 - scale(ranges["conc"])))))
        n_atk_ips = max(1, int(round(1 + 5 * (1 - scale(ranges["conc"])))))
        if family == "device_to_many_vpas":
            n_atk_devices = int(rng.integers(1, 3))
        if family == "ip_to_many_vpas":
            n_atk_ips = 1

        eid_counter += 1
        episodes.append(dict(
            id=f"ATK{eid_counter:05d}", etype="attack", family=family,
            slice_id=slice_id, start_ts=start_ts, end_ts=end_ts,
            intensity=intensity,
            volume_mult=scale(ranges["volume_mult"]),
            decline_rate=scale(ranges["decline"]),
            concentration_intensity=scale(ranges["conc"]),
            attacker_devices=[h(f"atkdev-{eid_counter}-{i}") for i in range(n_atk_devices)],
            attacker_ips=[h(f"atkip-{eid_counter}-{i}") for i in range(n_atk_ips)],
        ))

    for _ in range(n_flash_sales):
        slice_id = slice_ids[rng.integers(len(slice_ids))]
        start_day = int(rng.integers(0, max(sim_days - 1, 1)))
        start_window_idx = int(rng.integers(0, WINDOWS_PER_DAY))
        duration = int(rng.integers(4, 16))
        start_ts = SIM_START + timedelta(days=start_day, minutes=start_window_idx * WINDOW_MINUTES)
        end_ts = start_ts + timedelta(minutes=duration * WINDOW_MINUTES)
        eid_counter += 1
        episodes.append(dict(
            id=f"FLASH{eid_counter:05d}", etype="festive_spike_standalone", family=None,
            slice_id=slice_id, start_ts=start_ts, end_ts=end_ts,
            intensity=None,
            volume_mult=float(rng.uniform(1.5, 3.0)),
            amount_mult=float(rng.uniform(1.15, 1.5)),
        ))

    return episodes


def index_episodes_by_slice(episodes: list) -> dict:
    idx = {}
    for ep in episodes:
        idx.setdefault(ep["slice_id"], []).append(ep)
    for k in idx:
        idx[k].sort(key=lambda e: e["start_ts"])
    return idx

# TRANSACTION-LEVEL GENERATION
def gen_txn(rng, category, pool, new_counter, amount_shift, active_attack,
            decline_base_rate, infra_decline_boost, slice_id, window_start,
            merchant_id, geo_region, txn_id):
    """Generate a single raw transaction row given the window's context."""

    if active_attack is not None:
        fam = active_attack["family"]
        if fam == "device_to_many_vpas":
            device = active_attack["attacker_devices"][rng.integers(len(active_attack["attacker_devices"]))]
            ip = h(f"atkip-{active_attack['id']}-{rng.integers(50)}")
            vpa = h(f"stolen-{active_attack['id']}-{rng.integers(1_000_000)}")
        elif fam == "ip_to_many_vpas":
            ip = active_attack["attacker_ips"][0]
            device = h(f"atkdev-{active_attack['id']}-{rng.integers(30)}")
            vpa = h(f"stolen-{active_attack['id']}-{rng.integers(1_000_000)}")
        elif fam in ("low_value_testing", "high_decline_campaign"):
            device = active_attack["attacker_devices"][rng.integers(len(active_attack["attacker_devices"]))]
            ip = active_attack["attacker_ips"][rng.integers(len(active_attack["attacker_ips"]))]
            vpa = h(f"guess-{active_attack['id']}-{rng.integers(1_000_000)}")
        else:  # sudden_burst, gradual_ramp, distributed_low_and_slow, festive_attack
            if rng.random() < active_attack["concentration_intensity"]:
                device = active_attack["attacker_devices"][rng.integers(len(active_attack["attacker_devices"]))]
                ip = active_attack["attacker_ips"][rng.integers(len(active_attack["attacker_ips"]))]
            else:
                device = h(f"mixdev-{active_attack['id']}-{rng.integers(500)}")
                ip = h(f"mixip-{active_attack['id']}-{rng.integers(200)}")
            vpa = h(f"mixvpa-{active_attack['id']}-{rng.integers(1_000_000)}")
    else:
        p_new = 0.05 if amount_shift <= 1.05 else 0.18  # more first-time buyers during real festive surges
        if rng.random() < p_new:
            c = new_customer(slice_id, new_counter())
        else:
            c = pool[rng.integers(len(pool))]
        device, ip, vpa = c["device"], c["ip"], c["vpa"]

    mu, sigma = CATEGORY_AMOUNT_PARAMS[category]
    if active_attack is not None and active_attack["family"] == "low_value_testing":
        amount = float(rng.uniform(1.0, TEST_AMOUNT_THRESHOLD * 0.9))
    else:
        amount = float(max(rng.lognormal(mu, sigma) * amount_shift, 1.0))

    txn_type_probs = [0.55, 0.25, 0.20]
    if active_attack is not None and active_attack["family"] in ("low_value_testing", "high_decline_campaign"):
        txn_type = "collect_request"
    else:
        txn_type = rng.choice(["pay", "collect_request", "collect_accept"], p=txn_type_probs)

    decline_rate = decline_base_rate + infra_decline_boost
    declined = rng.random() < decline_rate
    if declined:
        if active_attack is not None:
            reason = rng.choice(["invalid_vpa", "vpa_not_found", "collect_request_expired"], p=[0.5, 0.35, 0.15])
        elif infra_decline_boost > 0 and rng.random() < 0.6:
            reason = rng.choice(["bank_server_timeout", "npci_timeout"])
        else:
            reason = rng.choice(["insufficient_funds", "incorrect_pin", "bank_server_error"], p=[0.6, 0.25, 0.15])
        status = "declined"
    else:
        reason, status = None, "success"

    psp = rng.choice(list(PSP_HANDLES.keys()), p=list(PSP_HANDLES.values()))

    return dict(
        transaction_id=txn_id, timestamp=window_start, slice_id=slice_id, merchant_id=merchant_id,
        payer_vpa_id=vpa, vpa_psp_handle=psp, txn_type=txn_type, amount=round(amount, 2),
        status=status, decline_reason_code=reason, device_id_hash=device, ip_hash=ip,
        geo_region=geo_region,
    )

# MAIN SIMULATION LOOP
def simulate(sim_days: int, n_slices: int, n_attack_episodes: int,
             n_flash_sales: int, pool_size: int, seed: int):
    rng = np.random.default_rng(seed)
    slices_df = build_slices(n_slices, rng)
    episodes = build_episodes(slices_df, sim_days, n_attack_episodes, n_flash_sales, rng)
    ep_by_slice = index_episodes_by_slice(episodes)

    raw_rows = []
    window_meta_rows = []
    txn_id_counter = 0

    for _, srow in slices_df.iterrows():
        slice_id, category = srow["slice_id"], srow["category"]
        merchant_id, geo_region = srow["merchant_id"], srow["geo_region"]
        base_lambda = srow["base_lambda"]
        pool = build_customer_pool(slice_id, pool_size)
        new_counter_state = [0]

        def next_counter():
            new_counter_state[0] += 1
            return new_counter_state[0]

        slice_episodes = ep_by_slice.get(slice_id, [])

        for day_offset in range(sim_days):
            day_dt = SIM_START + timedelta(days=day_offset)
            dow = day_dt.weekday()
            for w in range(WINDOWS_PER_DAY):
                window_start = day_dt + timedelta(minutes=w * WINDOW_MINUTES)
                hour = window_start.hour
                fest_mult, fest_name, fest_phase = festival_multiplier(day_offset, category)
                is_festival = fest_mult > 1.05

                lam = base_lambda * HOURLY_SHAPE[hour] * dow_multiplier(dow) * fest_mult
                lam *= float(rng.lognormal(0.0, 0.15))  
                lam_legit = lam  # cached pre-episode lambda: legitimate demand, untouched by any attack volume_mult

                infra_decline_boost = 0.0
                amount_shift = 1.0
                if is_festival:
                    infra_decline_boost = 0.01 + 0.03 * min((fest_mult - 1) / 5, 1)
                    amount_shift *= 1 + 0.3 * min((fest_mult - 1) / 5, 1)

                scenario_type = "festive_spike" if is_festival else "normal"
                is_attack = 0
                episode_id = None
                attack_family = None
                decline_base_rate = BASE_DECLINE_RATE
                active_attack = None

                for ep in slice_episodes:
                    if ep["start_ts"] <= window_start < ep["end_ts"]:
                        if ep["etype"] == "festive_spike_standalone":
                            lam *= ep["volume_mult"]
                            lam_legit *= ep["volume_mult"]  # flash-sale volume is genuine demand, stays legit
                            amount_shift *= ep["amount_mult"]
                            if scenario_type == "normal":
                                scenario_type = "festive_spike"
                        else:  # attack
                            lam *= ep["volume_mult"]
                            if not is_attack:
                                is_attack = 1
                                episode_id = ep["id"]
                                attack_family = ep["family"]
                                decline_base_rate = ep["decline_rate"]
                                active_attack = ep
                                scenario_type = "festive_attack" if scenario_type == "festive_spike" else "attack"

                n_txn_legit = int(rng.poisson(max(lam_legit, 0.01)))
                n_txn_attack = int(rng.poisson(max(lam - lam_legit, 0.01))) if active_attack is not None else 0

                for _ in range(n_txn_legit):
                    txn_id_counter += 1
                    raw_rows.append(gen_txn(
                        rng, category, pool, next_counter, amount_shift, None,
                        BASE_DECLINE_RATE, infra_decline_boost, slice_id, window_start,
                        merchant_id, geo_region, txn_id_counter,
                    ))
                for _ in range(n_txn_attack):
                    txn_id_counter += 1
                    raw_rows.append(gen_txn(
                        rng, category, pool, next_counter, amount_shift, active_attack,
                        decline_base_rate, infra_decline_boost, slice_id, window_start,
                        merchant_id, geo_region, txn_id_counter,
                    ))

                window_meta_rows.append(dict(
                    slice_id=slice_id, window_start_ts=window_start,
                    day_of_week=dow, hour_of_day=hour,
                    is_festival_window=is_festival, festival_name=fest_name,
                    festival_phase=fest_phase if is_festival else None,
                    scenario_type=scenario_type, is_attack=is_attack,
                    episode_id=episode_id, attack_family=attack_family,
                    category=category, geo_region=geo_region,
                ))

    raw_df = pd.DataFrame(raw_rows)
    meta_df = pd.DataFrame(window_meta_rows)
    return raw_df, meta_df, slices_df, episodes

# AGGREGATION: raw events -> feature table
def aggregate_features(raw_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    if len(raw_df) == 0:
        agg = pd.DataFrame(columns=["slice_id", "window_start_ts"])
    else:
        raw_df = raw_df.copy()
        raw_df["declined"] = (raw_df["status"] == "declined").astype(int)
        raw_df["below_thresh"] = (raw_df["amount"] < TEST_AMOUNT_THRESHOLD).astype(int)

        def top_reason_share(s):
            s = s.dropna()
            if len(s) == 0:
                return 0.0
            return s.value_counts(normalize=True).iloc[0]

        grp = raw_df.groupby(["slice_id", "timestamp"])
        agg = grp.agg(
            txn_count=("transaction_id", "count"),
            decline_rate=("declined", "mean"),
            avg_amount=("amount", "mean"),
            amount_std=("amount", "std"),
            pct_txn_below_threshold=("below_thresh", "mean"),
            unique_vpa_count=("payer_vpa_id", "nunique"),
            unique_device_count=("device_id_hash", "nunique"),
            unique_ip_count=("ip_hash", "nunique"),
        ).reset_index().rename(columns={"timestamp": "window_start_ts"})
        reason_share = grp["decline_reason_code"].apply(top_reason_share).reset_index()
        reason_share.columns = ["slice_id", "window_start_ts", "top_decline_reason_share"]
        agg = agg.merge(reason_share, on=["slice_id", "window_start_ts"], how="left")

    full = meta_df.merge(agg, on=["slice_id", "window_start_ts"], how="left")
    for col, fill in [("txn_count", 0), ("decline_rate", 0.0), ("avg_amount", 0.0),
                      ("amount_std", 0.0), ("pct_txn_below_threshold", 0.0),
                      ("unique_vpa_count", 0), ("unique_device_count", 0),
                      ("unique_ip_count", 0), ("top_decline_reason_share", 0.0)]:
        full[col] = full[col].fillna(fill)

    eps = 1e-6
    full["vpa_reuse_ratio"] = 1 - (full["unique_vpa_count"] / (full["txn_count"] + eps))
    full["vpa_reuse_ratio"] = full["vpa_reuse_ratio"].where(full["txn_count"] > 0, 0.0)
    full["device_concentration"] = full["txn_count"] / (full["unique_device_count"] + eps)
    full["device_concentration"] = full["device_concentration"].where(full["txn_count"] > 0, 0.0)
    full["ip_concentration"] = full["txn_count"] / (full["unique_ip_count"] + eps)
    full["ip_concentration"] = full["ip_concentration"].where(full["txn_count"] > 0, 0.0)

    full = full.sort_values(["slice_id", "window_start_ts"]).reset_index(drop=True)
    prev_txn = full.groupby("slice_id")["txn_count"].shift(1)
    raw_roc = (full["txn_count"] - prev_txn) / prev_txn.replace(0, np.nan)
    zero_to_n = (prev_txn == 0) & (full["txn_count"] > 0)
    raw_roc[zero_to_n] = full.loc[zero_to_n, "txn_count"].astype(float)
    full["rate_of_change"] = raw_roc.fillna(0.0)
    return full

# WALK-FORWARD SEASONAL BASELINE + CUSUM/EWMA 
def add_seasonal_baseline_and_stats(df: pd.DataFrame, cusum_k: float = 0.5,
                                     ewma_alpha: float = 0.3) -> pd.DataFrame:
    """
    Two-layer walk-forward baseline:

    Layer 1: per-slice, per-hour trailing average of non-festival transaction
    counts. Correct for ordinary days, blind to festival magnitude.
    Layer 2: a per-(category, festival_phase) running multiplier, learned from
    prior clean festive_spike windows only (never attack windows), and only from
    split == 'train' rows, so a held-out slice's or festival's own behavior can
    never leak into the shared estimate used for other slices' training features.
"""
    df = df.sort_values(["window_start_ts", "slice_id"]).reset_index(drop=True)
    n = len(df)

    baseline_naive = np.zeros(n)
    baseline_adj = np.zeros(n)
    resid_naive = np.zeros(n)
    resid_adj = np.zeros(n)
    fest_mult_est = np.ones(n)
    cusum = np.zeros(n)
    ewma = np.zeros(n)
    cusum_naive = np.zeros(n)
    ewma_naive = np.zeros(n)

    slice_hour_hist = {}       # slice_id -> hour -> non-festival txn counts
    slice_overall_hist = {}    # slice_id -> non-festival txn counts, fallback
    cat_phase_ratio_hist = {}  # (category, phase) -> realized ratios, clean train-split festival windows only

    slice_id_arr = df["slice_id"].values
    category_arr = df["category"].values
    hour_arr = df["hour_of_day"].values
    is_fest_arr = df["is_festival_window"].values
    phase_arr = df["festival_phase"].values
    scenario_arr = df["scenario_type"].values
    txn_arr = df["txn_count"].values
    split_arr = df["split"].values

    s_prev_adj = {}
    e_prev_adj = {}
    s_prev_naive = {}
    e_prev_naive = {}

    for i in range(n):
        sid, cat, hour = slice_id_arr[i], category_arr[i], hour_arr[i]
        is_fest, phase, scenario, txn = is_fest_arr[i], phase_arr[i], scenario_arr[i], txn_arr[i]
        row_split = split_arr[i]

        hour_hist = slice_hour_hist.setdefault(sid, {}).get(hour, [])
        overall_hist = slice_overall_hist.setdefault(sid, [])
        if len(hour_hist) >= 3:
            b_nonfest = float(np.mean(hour_hist[-56:]))
        elif len(overall_hist) > 0:
            b_nonfest = float(np.mean(overall_hist[-200:]))
        else:
            b_nonfest = max(txn, 1.0)
        baseline_naive[i] = b_nonfest

        key = (cat, phase)
        ratio_hist = cat_phase_ratio_hist.get(key, [])
        mult_est = float(np.mean(ratio_hist)) if len(ratio_hist) > 0 else 1.0
        fest_mult_est[i] = mult_est if is_fest else 1.0
        b_adj = b_nonfest * mult_est if is_fest else b_nonfest
        baseline_adj[i] = b_adj

        r_naive = txn / max(b_nonfest, 0.5)
        r_adj = txn / max(b_adj, 0.5)
        resid_naive[i] = r_naive
        resid_adj[i] = r_adj

        sp = s_prev_adj.get(sid, 0.0)
        sp = max(0.0, sp + (r_adj - 1.0) - cusum_k)
        s_prev_adj[sid] = sp
        cusum[i] = sp
        ep = e_prev_adj.get(sid)
        ep = (r_adj - 1.0) if ep is None else ewma_alpha * (r_adj - 1.0) + (1 - ewma_alpha) * ep
        e_prev_adj[sid] = ep
        ewma[i] = ep

        spn = s_prev_naive.get(sid, 0.0)
        spn = max(0.0, spn + (r_naive - 1.0) - cusum_k)
        s_prev_naive[sid] = spn
        cusum_naive[i] = spn
        epn = e_prev_naive.get(sid)
        epn = (r_naive - 1.0) if epn is None else ewma_alpha * (r_naive - 1.0) + (1 - ewma_alpha) * epn
        e_prev_naive[sid] = epn
        ewma_naive[i] = epn

        # Update history after using it
        if scenario == "normal":
            slice_hour_hist[sid].setdefault(hour, []).append(txn)
            overall_hist.append(txn)
        elif scenario == "festive_spike" and row_split == "train":
            cat_phase_ratio_hist.setdefault(key, []).append(r_naive)

    df["baseline_nonfestival_hourly"] = baseline_naive
    df["festival_multiplier_estimate"] = fest_mult_est
    df["seasonal_baseline_expected"] = baseline_adj
    df["seasonal_residual"] = resid_adj
    df["cusum_statistic"] = cusum
    df["ewma_statistic"] = ewma
    df["seasonal_residual_naive"] = resid_naive
    df["cusum_statistic_naive"] = cusum_naive
    df["ewma_statistic_naive"] = ewma_naive

    df = df.sort_values(["slice_id", "window_start_ts"]).reset_index(drop=True)
    return df

# TRAIN/TEST SPLIT: by slice AND by time, never by row shuffle
def assign_split(df: pd.DataFrame, sim_days: int, test_slice_frac: float = 0.2,
                  family_holdout: str = None, seed: int = 7) -> pd.DataFrame:
    """
    Three generalization axes, no row shuffling:
      1. Unseen slice: test_slice_frac of merchants held out entirely.
      2. Unseen festival: any window in New Year's Day Sale, Republic Day Sale,
         or Holi Sale is always test, regardless of slice or time.
      3. Optional unseen attack family.
    """
    rng = np.random.default_rng(seed)
    all_slices = df["slice_id"].unique()
    n_test_slices = max(1, int(len(all_slices) * test_slice_frac))
    test_slices = set(rng.choice(all_slices, size=n_test_slices, replace=False))

    is_test_slice = df["slice_id"].isin(test_slices)
    is_test_festival = df["festival_name"].isin(TEST_FESTIVAL_NAMES)
    is_holdout_family = df["attack_family"] == family_holdout if family_holdout else False

    df = df.copy()
    df["split"] = np.where(is_test_slice | is_test_festival | is_holdout_family, "test", "train")
    return df

# MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-days", type=int, default=256)
    ap.add_argument("--n-slices", type=int, default=30)
    ap.add_argument("--n-attack-episodes", type=int, default=800)
    ap.add_argument("--n-flash-sales", type=int, default=250)
    ap.add_argument("--pool-size", type=int, default=250)
    ap.add_argument("--family-holdout", type=str, default=None,
                     help="e.g. ip_to_many_vpas -- forces this attack family test-only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="./data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw_df, meta_df, slices_df, episodes = simulate(
        sim_days=args.sim_days, n_slices=args.n_slices,
        n_attack_episodes=args.n_attack_episodes, n_flash_sales=args.n_flash_sales,
        pool_size=args.pool_size, seed=args.seed,
    )

    features = aggregate_features(raw_df, meta_df)
    features = assign_split(features, sim_days=args.sim_days, family_holdout=args.family_holdout)
    features = add_seasonal_baseline_and_stats(features)

    raw_path = os.path.join(args.outdir, "raw_events.csv")
    feat_path = os.path.join(args.outdir, "features.csv")
    slices_path = os.path.join(args.outdir, "slices.csv")
    episodes_path = os.path.join(args.outdir, "episodes_manifest.csv")

    raw_df.to_csv(raw_path, index=False)
    features.to_csv(feat_path, index=False)
    slices_df.to_csv(slices_path, index=False)
    pd.DataFrame(episodes).to_csv(episodes_path, index=False)
    try:
        raw_df.to_parquet(raw_path.replace(".csv", ".parquet"), index=False)
        features.to_parquet(feat_path.replace(".csv", ".parquet"), index=False)
    except Exception as e:
        print(f"(parquet export skipped: {e})")

    print("=== GENERATION SUMMARY ===")
    print(f"raw_events rows:  {len(raw_df):,}")
    print(f"features rows:    {len(features):,}  "
          f"(expected ~ {args.n_slices} slices x {args.sim_days*WINDOWS_PER_DAY:,} windows "
          f"= {args.n_slices*args.sim_days*WINDOWS_PER_DAY:,})")
    print("\nscenario_type counts:")
    print(features["scenario_type"].value_counts())
    print("\nis_attack balance:")
    print(features["is_attack"].value_counts(normalize=True).round(4))
    print("\nattack_family counts (windows, not episodes):")
    print(features.loc[features["is_attack"] == 1, "attack_family"].value_counts())
    print("\ndistinct attack episodes injected:", len([e for e in episodes if e["etype"] == "attack"]))
    print("distinct flash-sale episodes injected:", len([e for e in episodes if e["etype"] == "festive_spike_standalone"]))
    print("\nsplit counts:")
    print(features["split"].value_counts())
    print("\nfestive false-positive risk check -- decline_rate stats within festive_spike (non-attack) windows only:")
    fs = features[features["scenario_type"] == "festive_spike"]
    print(fs["decline_rate"].describe()[["mean", "50%", "max"]])

    manifest = dict(
        config=dict(sim_days=args.sim_days, n_slices=args.n_slices,
                    n_attack_episodes=args.n_attack_episodes, n_flash_sales=args.n_flash_sales,
                    pool_size=args.pool_size, seed=args.seed, family_holdout=args.family_holdout),
        outputs=dict(raw_events=raw_path, features=feat_path, slices=slices_path, episodes=episodes_path),
    )
    with open(os.path.join(args.outdir, "generation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nWrote outputs to {args.outdir}")

if __name__ == "__main__":
    main()