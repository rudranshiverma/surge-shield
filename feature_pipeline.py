"""
feature_pipeline.py
Single source of truth for turning a raw features.parquet row into model inputs, for all three tiers. 
Imported by both train_evaluate.py (batch, training time) and server.py (single-row, live serving time).
"""
import numpy as np
import pandas as pd

TEST_FESTIVAL_NAMES = {"New Year's Day Sale", "Republic Day Sale", "Holi Sale"}
TIER1_FIXED_FESTIVAL_MULT = 3.0

TIER0_FEATURES = ["txn_count", "rate_of_change"]
TIER2_NUMERIC_FEATURES = [
    "txn_count", "rate_of_change", "decline_rate", "top_decline_reason_share",
    "avg_amount", "amount_std", "pct_txn_below_threshold",
    "unique_vpa_count", "vpa_reuse_ratio", "unique_device_count", "device_concentration",
    "unique_ip_count", "ip_concentration", "seasonal_baseline_expected", "seasonal_residual",
    "cusum_statistic", "ewma_statistic", "festival_multiplier_estimate",
]
TIER2_CATEGORICAL_FEATURES = ["category", "geo_region", "festival_phase", "day_of_week", "hour_of_day"]

def add_tier1_naive_residual(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fixed_baseline = df["baseline_nonfestival_hourly"].clip(lower=0.5) * np.where(
        df["is_festival_window"], TIER1_FIXED_FESTIVAL_MULT, 1.0
    )
    df["tier1_naive_residual"] = df["txn_count"] / fixed_baseline
    return df

def fit_categorical_encoder(train_df: pd.DataFrame, cols: list) -> dict:
    return {col: sorted(train_df[col].fillna("none").astype(str).unique()) for col in cols}

def apply_categorical_encoder(df: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, cats in encoder.items():
        col_str = df[col].fillna("none").astype(str)
        for c in cats:
            out[f"{col}__{c}"] = (col_str == c).astype(int)
    return out

def assemble_features(df: pd.DataFrame, tier: int, cat_encoded: pd.DataFrame) -> pd.DataFrame:
    """df must already have tier1_naive_residual (via add_tier1_naive_residual) if tier==1.
    cat_encoded must already be computed (via apply_categorical_encoder) if tier==2
    passed in rather than computed here because the ENCODER must be fit on train only,
    which this stateless function has no way to know about on its own."""
    if tier == 0:
        return df[TIER0_FEATURES].copy()
    if tier == 1:
        X = df[TIER0_FEATURES].copy()
        X["is_festival_window"] = df["is_festival_window"].astype(int)
        X["tier1_naive_residual"] = df["tier1_naive_residual"]
        return X
    if tier == 2:
        X = df[TIER2_NUMERIC_FEATURES].copy()
        X["is_festival_window"] = df["is_festival_window"].astype(int)
        return pd.concat([X, cat_encoded.loc[df.index]], axis=1)
    raise ValueError(tier)

def assemble_single_row(row: pd.Series, tier: int, cat_encoder: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    if tier == 1:
        df = add_tier1_naive_residual(df)
    cat_encoded = apply_categorical_encoder(df, cat_encoder) if tier == 2 else None
    return assemble_features(df, tier, cat_encoded)