"""FastAPI backend: serves the static frontend and the live inference API from the same origin"""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from xgboost import XGBClassifier

from feature_pipeline import assemble_single_row, apply_categorical_encoder, assemble_features, add_tier1_naive_residual
from train_evaluate import (
    get_non_straddling_test_episodes, compute_episode_exposure, sweep_cost_by_threshold,
    confusion_counts, COST_PER_FALSE_ALARM,
)
from generate_dataset import FESTIVALS, SIM_START
import policy_engine as pe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DASHBOARD_DATA_DIR", os.path.join(BASE_DIR, "data"))
EVAL_DIR = os.path.join(DATA_DIR, "eval")
DB_PATH = os.path.join(BASE_DIR, "audit_trail.db")

STATE = {}
def require_file(path: str):
    if not os.path.exists(path):
        raise RuntimeError(
            f"Required file not found: {path}\n"
            f"Run generate_dataset.py then train_evaluate.py first, or set DASHBOARD_DATA_DIR "
            f"to the folder containing features.parquet and an eval/ subfolder."
        )
    return path

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT, slice_id TEXT, window_start_ts TEXT,
            calibrated_prob REAL, system_action TEXT, dominant_signal TEXT,
            estimated_exposure_inr REAL, analyst_action TEXT, ground_truth_is_attack INTEGER,
            note TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            slice_id TEXT PRIMARY KEY,
            added_at TEXT, dominant_signal TEXT, calibrated_prob REAL
        )
    """)
    conn.commit()
    conn.close()

def load_watchlist() -> set:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT slice_id FROM watchlist").fetchall()
    conn.close()
    return {r[0] for r in rows}

def festival_date_ranges() -> list:
    out = []
    for f in FESTIVALS:
        start = SIM_START + timedelta(days=f["peak_day"] - f["ramp_days"])
        end = SIM_START + timedelta(days=f["peak_day"] + f["decay_days"], hours=23, minutes=59)
        out.append(dict(name=f["name"], start_ts=start, end_ts=end, is_test_festival=bool(f.get("is_test_festival"))))
    return out

def score_batch(df: pd.DataFrame, tier: int):
    if tier == 1:
        df = add_tier1_naive_residual(df)
    cat_encoded = apply_categorical_encoder(df, STATE["cat_encoder"]) if tier == 2 else None
    X = assemble_features(df, tier, cat_encoded)
    if tier == 2:
        return STATE["calibrator"].predict_proba(X)[:, 1], X
    return STATE["models"][tier].predict_proba(X)[:, 1], X

def select_live_stream_for_festival(name: str, pad: int = 8, max_len: int = 40, min_len: int = 16):
    """Real chronological data for one merchant in one festival's date range. Finds up
    to 3 real escalations, trims to a contiguous window around them, and prefers a
    slice whose escalations span at least 2 different signals."""
    ranges = {r["name"]: r for r in festival_date_ranges()}
    r = ranges[name]
    sub = STATE["test_df"][(STATE["test_df"]["window_start_ts"] >= r["start_ts"]) &
                            (STATE["test_df"]["window_start_ts"] <= r["end_ts"])]
    if len(sub) == 0:
        return None, [], []

    best = None
    fallback = None
    for slice_id, grp in sub.groupby("slice_id"):
        grp = grp.sort_values("window_start_ts").reset_index(drop=True)
        probs, _ = score_batch(grp.copy(), 2)
        positions, signals = [], []
        for i, row in grp.iterrows():
            feats = dict(calibrated_prob=probs[i], avg_amount=row["avg_amount"], txn_count=row["txn_count"],
                         decline_rate=row["decline_rate"], device_concentration=row["device_concentration"],
                         ip_concentration=row["ip_concentration"], pct_txn_below_threshold=row["pct_txn_below_threshold"],
                         cusum_statistic=row["cusum_statistic"])
            d = pe.evaluate(feats, is_watchlisted=slice_id in STATE["watchlisted_slices"])
            if d.action in (pe.Action.HUMAN_REVIEW, pe.Action.URGENT_REVIEW):
                positions.append(i)
                signals.append(d.dominant_signal)
        if not positions:
            if fallback is None:
                fallback = (slice_id, grp)
            continue
        focus_signals = signals[:3]
        diversity_rank = 0 if len(set(focus_signals)) >= 2 else 1
        score = (diversity_rank, abs(len(positions) - 2), -len(grp), slice_id)
        if best is None or score < best[0]:
            best = (score, slice_id, grp, positions, signals)

    if best is None:
        if fallback is None:
            return None, [], []
        slice_id, grp = fallback
        window = grp.iloc[: min(len(grp), max_len)]
        return slice_id, window["window_start_ts"].astype(str).tolist(), []

    _, slice_id, grp, positions, signals = best
    focus = positions[:3]
    hi_ceiling = (positions[3] - 1) if len(positions) > 3 else (len(grp) - 1)
    lo = max(0, focus[0] - pad)
    hi = min(hi_ceiling, focus[-1] + pad)
    target_min = min(len(grp), min_len)
    while (hi - lo + 1) < target_min and (lo > 0 or hi < hi_ceiling):
        if lo > 0: lo -= 1
        if hi < hi_ceiling: hi += 1
    if (hi - lo + 1) > max_len:
        extra = (hi - lo + 1) - max_len
        lo += extra // 2
        hi -= extra - extra // 2
    hi = max(hi, lo)

    window = grp.iloc[lo:hi + 1]
    included_signals = [sig for pos, sig in zip(positions, signals) if lo <= pos <= hi]
    return slice_id, window["window_start_ts"].astype(str).tolist(), included_signals

def select_best_demo_stream():
    """Runs the festival selector above across all six festivals and keeps the one
    with the most diverse escalation signals, closest to 2-3 escalations."""
    best = None
    for f in FESTIVALS:
        slice_id, timestamps, signals = select_live_stream_for_festival(f["name"])
        if slice_id is None:
            continue
        diversity_rank = 0 if len(set(signals[:3])) >= 2 else 1
        score = (diversity_rank, abs(len(signals[:3]) - 2), f["name"])
        if best is None or score < best[0]:
            best = (score, f["name"], slice_id, timestamps, signals)
    if best is None:
        return None
    _, festival_name, slice_id, timestamps, signals = best
    return dict(festival=festival_name, slice_id=slice_id,
                rows=[dict(slice_id=slice_id, window_start_ts=ts) for ts in timestamps],
                escalation_signals=signals)

@asynccontextmanager
async def lifespan(app: FastAPI):
    require_file(os.path.join(DATA_DIR, "features.parquet"))
    for fname in ["tier0_model.json", "tier1_model.json", "tier2_model.json", "tier2_calibrator.joblib", "cat_encoder.joblib"]:
        require_file(os.path.join(EVAL_DIR, fname))

    df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    STATE["test_df"] = df[df["split"] == "test"].reset_index(drop=True)

    models = {}
    for t in (0, 1):
        m = XGBClassifier()
        m.load_model(os.path.join(EVAL_DIR, f"tier{t}_model.json"))
        models[t] = m
    STATE["models"] = models
    STATE["calibrator"] = joblib.load(os.path.join(EVAL_DIR, "tier2_calibrator.joblib"))
    STATE["cat_encoder"] = joblib.load(os.path.join(EVAL_DIR, "cat_encoder.joblib"))

    tier2_base = XGBClassifier()
    tier2_base.load_model(os.path.join(EVAL_DIR, "tier2_model.json"))
    STATE["tier2_base_model"] = tier2_base
    STATE["explainer"] = shap.TreeExplainer(tier2_base)

    STATE["valid_episode_ids"] = get_non_straddling_test_episodes(df)
    STATE["episode_exposure"] = compute_episode_exposure(STATE["test_df"], STATE["valid_episode_ids"])

    metrics_path = os.path.join(EVAL_DIR, "metrics_summary.json")
    tier_best = {0: dict(threshold=0.5), 1: dict(threshold=0.5), 2: dict(threshold=0.5)}
    if os.path.exists(metrics_path):
        saved = json.load(open(metrics_path))
        if "cost_model" in saved and "tier_best" in saved["cost_model"]:
            tier_best = {int(k): v for k, v in saved["cost_model"]["tier_best"].items()}
    STATE["tier_best"] = tier_best

    init_db()
    # Selecting the demo stream before loading the real watchlist keeps it reproducible across restarts, regardless of leftover watchlist state from a prior run.
    STATE["watchlisted_slices"] = set()
    STATE["demo_stream"] = select_best_demo_stream()
    STATE["watchlisted_slices"] = load_watchlist()
    if STATE["demo_stream"]:
        print(f"[startup] demo stream: {STATE['demo_stream']['festival']} / {STATE['demo_stream']['slice_id']} "
              f"({len(STATE['demo_stream']['rows'])} rows, signals={STATE['demo_stream']['escalation_signals']})")
    else:
        print("[startup] WARNING: no demo stream could be selected across any festival.")
    yield


app = FastAPI(lifespan=lifespan)

class ActionBody(BaseModel):
    slice_id: str
    window_start_ts: str
    action: str
    note: str = None
    escalated_to: str = None

@app.get("/api/festivals")
def get_festivals():
    ranges = festival_date_ranges()
    return dict(
        overall_start=SIM_START.isoformat(),
        overall_end=(SIM_START + timedelta(days=256)).isoformat(),
        festivals=[dict(name=r["name"], start_ts=r["start_ts"].isoformat(), end_ts=r["end_ts"].isoformat(),
                         is_test_festival=r["is_test_festival"]) for r in ranges],
    )

@app.get("/api/festival/{name}/metrics")
def festival_metrics(name: str):
    ranges = {r["name"]: r for r in festival_date_ranges()}
    if name not in ranges:
        raise HTTPException(404, f"Unknown festival: {name}")
    r = ranges[name]

    sub = STATE["test_df"][(STATE["test_df"]["window_start_ts"] >= r["start_ts"]) &
                            (STATE["test_df"]["window_start_ts"] <= r["end_ts"])].copy()
    if len(sub) == 0:
        raise HTTPException(404, f"No test rows in range for {name}")

    y_true = sub["is_attack"].values
    range_ep_ids = set(sub.loc[sub["episode_id"].notna(), "episode_id"].unique()) & STATE["valid_episode_ids"]
    range_exposure = STATE["episode_exposure"].reindex(list(range_ep_ids)).fillna(0)
    no_detection_floor = float(range_exposure.sum())

    tiers_out = {}
    for tier in (0, 1, 2):
        prob, _ = score_batch(sub.copy(), tier)
        threshold = STATE["tier_best"][tier]["threshold"]
        pred = (prob >= threshold).astype(int)
        overall = confusion_counts(y_true, pred)

        fs_mask = (sub["scenario_type"] == "festive_spike").values
        festive_fpr = float(pred[fs_mask].mean()) if fs_mask.sum() > 0 else None

        cost_sweep = sweep_cost_by_threshold(prob, sub, range_ep_ids, range_exposure, np.array([threshold]))
        total_cost = float(cost_sweep.iloc[0]["total_cost"])

        tiers_out[tier] = dict(threshold=threshold, precision=overall["precision"], recall=overall["recall"],
                                fpr=overall["fpr"], festive_fpr=festive_fpr, total_cost_inr=total_cost,
                                n_rows=len(sub), n_attack_rows=int(y_true.sum()))

    return dict(festival=name, is_test_festival=r["is_test_festival"], n_rows=len(sub),
                no_detection_floor_cost_inr=no_detection_floor, tiers=tiers_out)

def find_row(slice_id: str, window_start_ts: str) -> pd.Series:
    ts = pd.Timestamp(window_start_ts)
    match = STATE["test_df"][(STATE["test_df"]["slice_id"] == slice_id) & (STATE["test_df"]["window_start_ts"] == ts)]
    if len(match) != 1:
        raise HTTPException(404, f"No unique test row for slice_id={slice_id} window_start_ts={window_start_ts} "
                                  f"(found {len(match)})")
    return match.iloc[0]

@app.get("/api/live-stream")
def live_stream():
    if STATE["demo_stream"] is None:
        raise HTTPException(404, "No demo stream available")
    return STATE["demo_stream"]

@app.get("/api/case")
def case_detail(slice_id: str, window_start_ts: str):
    row = find_row(slice_id, window_start_ts)

    X = assemble_single_row(row, 2, STATE["cat_encoder"])
    calibrated_prob = float(STATE["calibrator"].predict_proba(X)[:, 1][0])

    features = dict(
        calibrated_prob=calibrated_prob, avg_amount=row["avg_amount"], txn_count=row["txn_count"],
        decline_rate=row["decline_rate"], device_concentration=row["device_concentration"],
        ip_concentration=row["ip_concentration"], pct_txn_below_threshold=row["pct_txn_below_threshold"],
        cusum_statistic=row["cusum_statistic"],
    )
    decision = pe.evaluate(features, is_watchlisted=(row["slice_id"] in STATE["watchlisted_slices"]))

    shap_values = STATE["explainer"].shap_values(X)
    contrib = pd.Series(shap_values[0], index=X.columns).sort_values(key=abs, ascending=False).head(6)
    shap_out = [dict(feature=k, value=round(float(v), 4)) for k, v in contrib.items()]

    def clean(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v

    return dict(
        slice_id=row["slice_id"], window_start_ts=str(row["window_start_ts"]),
        category=row["category"], geo_region=row["geo_region"],
        is_festival_window=bool(row["is_festival_window"]), festival_name=clean(row["festival_name"]),
        festival_phase=clean(row["festival_phase"]), txn_count=int(row["txn_count"]),
        decline_rate=float(row["decline_rate"]), avg_amount=float(row["avg_amount"]),
        calibrated_prob=calibrated_prob, action=decision.action.value,
        dominant_signal=decision.dominant_signal, response_hint=decision.response_hint,
        estimated_exposure_inr=decision.estimated_exposure_inr,
        cusum_override_triggered=decision.cusum_override_triggered,
        watchlist_override_triggered=decision.watchlist_override_triggered,
        is_slice_watchlisted=(row["slice_id"] in STATE["watchlisted_slices"]),
        reasoning=decision.reasoning, shap_top_features=shap_out,
        ground_truth_is_attack=int(row["is_attack"]), ground_truth_attack_family=clean(row["attack_family"]),
    )

@app.post("/api/case/action")
def log_action(body: ActionBody):
    if body.action not in ("Escalate", "Watchlist", "Dismiss"):
        raise HTTPException(400, "action must be Escalate, Watchlist, or Dismiss")
    detail = case_detail(body.slice_id, body.window_start_ts)

    note = body.note
    if body.escalated_to:
        note = f"Escalated to: {body.escalated_to}" + (f" | {note}" if note else "")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO audit_log (logged_at, slice_id, window_start_ts, calibrated_prob, system_action, "
        "dominant_signal, estimated_exposure_inr, analyst_action, ground_truth_is_attack, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), detail["slice_id"], detail["window_start_ts"],
         detail["calibrated_prob"], detail["action"], detail["dominant_signal"],
         detail["estimated_exposure_inr"], body.action, detail["ground_truth_is_attack"], note),
    )

    if body.action == "Watchlist":
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (slice_id, added_at, dominant_signal, calibrated_prob) "
            "VALUES (?, ?, ?, ?)",
            (body.slice_id, datetime.now().isoformat(timespec="seconds"),
             detail["dominant_signal"], detail["calibrated_prob"]),
        )
        STATE["watchlisted_slices"].add(body.slice_id)

    conn.commit()
    conn.close()
    return dict(status="logged")

@app.get("/api/audit-trail")
def audit_trail():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")