"""Trains and evaluates the three-tier ablation (naive / rule-based / full)
on the dataset from generate_dataset.py. See README for methodology."""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

from feature_pipeline import (
    TEST_FESTIVAL_NAMES, TIER1_FIXED_FESTIVAL_MULT, TIER0_FEATURES,
    TIER2_NUMERIC_FEATURES, TIER2_CATEGORICAL_FEATURES,
    add_tier1_naive_residual, fit_categorical_encoder, apply_categorical_encoder, assemble_features,
)

DECISION_THRESHOLD = 0.5
COST_PER_FALSE_ALARM = 50.0
CALIB_SLICE_FRAC = 0.35
CALIB_SPLIT_SEED = 99

def train_base_model(X_train, y_train, scale_pos_weight) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model

def confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, fpr=fpr)

def get_non_straddling_test_episodes(full_df: pd.DataFrame) -> set:
    atk = full_df[full_df["episode_id"].notna()]
    split_counts = atk.groupby("episode_id")["split"].agg(lambda s: set(s))
    return set(split_counts[split_counts.apply(lambda s: s == {"test"})].index)

def episode_level_detection(test_df: pd.DataFrame, pred: np.ndarray, valid_episode_ids: set) -> pd.DataFrame:
    tdf = test_df.copy()
    tdf["pred"] = pred
    tdf = tdf[tdf["episode_id"].isin(valid_episode_ids)].sort_values(["episode_id", "window_start_ts"])
    rows = []
    for ep_id, grp in tdf.groupby("episode_id"):
        flagged_positions = np.where(grp["pred"].values == 1)[0]
        caught = len(flagged_positions) > 0
        time_to_detect = int(flagged_positions[0]) if caught else None
        rows.append(dict(episode_id=ep_id, attack_family=grp["attack_family"].iloc[0],
                          n_windows=len(grp), caught=caught, time_to_detect_windows=time_to_detect))
    return pd.DataFrame(rows)

def compute_episode_exposure(test_df: pd.DataFrame, valid_episode_ids: set) -> pd.Series:
    atk = test_df[test_df["episode_id"].isin(valid_episode_ids)].copy()
    atk["window_exposure"] = atk["avg_amount"] * atk["txn_count"] * (1 - atk["decline_rate"])
    return atk.groupby("episode_id")["window_exposure"].sum()

def sweep_cost_by_threshold(prob: np.ndarray, test_df: pd.DataFrame, valid_episode_ids: set,
                             episode_exposure: pd.Series, thresholds: np.ndarray) -> pd.DataFrame:
    is_attack = (test_df["is_attack"] == 1).values
    non_attack_mask = ~is_attack
    ep_ids = test_df["episode_id"].values
    valid_mask = np.isin(ep_ids, list(valid_episode_ids))

    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        fp_count = int((pred[non_attack_mask] == 1).sum())
        fp_cost = fp_count * COST_PER_FALSE_ALARM

        caught_any = pd.Series(pred[valid_mask], index=ep_ids[valid_mask]).groupby(level=0).max()
        uncaught_episodes = caught_any[caught_any == 0].index
        fn_cost = float(episode_exposure.reindex(uncaught_episodes).fillna(0).sum())

        rows.append(dict(threshold=t, fp_count=fp_count, fp_cost=fp_cost,
                          n_uncaught_episodes=len(uncaught_episodes), fn_cost=fn_cost,
                          total_cost=fp_cost + fn_cost))
    return pd.DataFrame(rows)

def max_recall_at_precision(probs, y_true, min_precision=0.95):
    best = None
    for t in np.linspace(0.01, 0.99, 500):
        pred_t = (probs >= t).astype(int)
        tp_ = int(((y_true == 1) & (pred_t == 1)).sum())
        fp_ = int(((y_true == 0) & (pred_t == 1)).sum())
        fn_ = int(((y_true == 1) & (pred_t == 0)).sum())
        prec_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        rec_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        if prec_ >= min_precision and (best is None or rec_ > best[1]):
            best = (t, rec_, prec_)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--outdir", type=str, default="./data/eval")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(os.path.join(args.data_dir, "features.parquet"))
    assert {"train", "test"}.issubset(set(df["split"].unique()))
    df = add_tier1_naive_residual(df)

    full_train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    rng = np.random.default_rng(CALIB_SPLIT_SEED)
    train_slices = full_train_df["slice_id"].unique()
    n_calib = max(1, int(len(train_slices) * CALIB_SLICE_FRAC))
    calib_slices = set(rng.choice(train_slices, size=n_calib, replace=False))
    train_fit_df = full_train_df[~full_train_df["slice_id"].isin(calib_slices)].reset_index(drop=True)
    train_calib_df = full_train_df[full_train_df["slice_id"].isin(calib_slices)].reset_index(drop=True)
    print(f"train_fit: {len(train_fit_df):,} rows ({train_fit_df['slice_id'].nunique()} slices) | "
          f"train_calib: {len(train_calib_df):,} rows ({train_calib_df['slice_id'].nunique()} slices) | "
          f"calib positives: {int((train_calib_df['is_attack']==1).sum())}")

    y_train_fit = train_fit_df["is_attack"].values
    y_calib = train_calib_df["is_attack"].values
    y_test = test_df["is_attack"].values

    cat_encoder = fit_categorical_encoder(train_fit_df, TIER2_CATEGORICAL_FEATURES)
    fit_cat = apply_categorical_encoder(train_fit_df, cat_encoder)
    calib_cat = apply_categorical_encoder(train_calib_df, cat_encoder)
    test_cat = apply_categorical_encoder(test_df, cat_encoder)

    scale_pos_weight = (y_train_fit == 0).sum() / max((y_train_fit == 1).sum(), 1)

    results = {}
    test_probs = {}
    all_base_models = {}
    for tier in (0, 1, 2):
        X_fit = assemble_features(train_fit_df, tier, fit_cat)
        X_test = assemble_features(test_df, tier, test_cat)
        assert X_fit.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0

        base_model = train_base_model(X_fit, y_train_fit, scale_pos_weight)
        all_base_models[tier] = base_model

        if tier == 2:
            X_calib = assemble_features(train_calib_df, tier, calib_cat)
            calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="isotonic")
            calibrator.fit(X_calib, y_calib)
            prob = calibrator.predict_proba(X_test)[:, 1]
            tier2_base_model = base_model
            tier2_calibrator = calibrator
            tier2_X_test = X_test
        else:
            prob = base_model.predict_proba(X_test)[:, 1]

        pred = (prob >= DECISION_THRESHOLD).astype(int)
        test_probs[tier] = prob

        overall = confusion_counts(y_test, pred)
        overall["pr_auc"] = float(average_precision_score(y_test, prob))

        fs_mask = (test_df["scenario_type"] == "festive_spike").values
        unseen_fest_mask = fs_mask & test_df["festival_name"].isin(TEST_FESTIVAL_NAMES).values
        unseen_slice_mask = fs_mask & ~test_df["festival_name"].isin(TEST_FESTIVAL_NAMES).values
        festive_fpr_overall = float(pred[fs_mask].mean()) if fs_mask.sum() > 0 else float("nan")
        festive_fpr_unseen_festival = float(pred[unseen_fest_mask].mean()) if unseen_fest_mask.sum() > 0 else float("nan")
        festive_fpr_unseen_slice = float(pred[unseen_slice_mask].mean()) if unseen_slice_mask.sum() > 0 else float("nan")

        results[tier] = dict(
            overall=overall, festive_fpr_overall=festive_fpr_overall,
            festive_fpr_unseen_festival=festive_fpr_unseen_festival,
            festive_fpr_unseen_festival_n=int(unseen_fest_mask.sum()),
            festive_fpr_unseen_slice=festive_fpr_unseen_slice,
            festive_fpr_unseen_slice_n=int(unseen_slice_mask.sum()),
            calibrated=(tier == 2),
        )
        print(f"\n=== Tier {tier} {'(calibrated)' if tier==2 else '(raw)'} ===")
        print(f"  PR-AUC: {overall['pr_auc']:.4f}  precision={overall['precision']:.4f} "
              f"recall={overall['recall']:.4f} f1={overall['f1']:.4f} fpr={overall['fpr']:.4f}")
        print(f"  festive-surge FPR: overall={festive_fpr_overall:.4f}  "
              f"unseen-festival(n={unseen_fest_mask.sum()})={festive_fpr_unseen_festival:.4f}  "
              f"unseen-slice(n={unseen_slice_mask.sum()})={festive_fpr_unseen_slice:.4f}")

    valid_ep_ids = get_non_straddling_test_episodes(df)
    atk_all = df[df["episode_id"].notna()]
    split_sets = atk_all.groupby("episode_id")["split"].agg(lambda s: frozenset(s))
    n_straddling = int((split_sets == frozenset({"train", "test"})).sum())
    n_pure_train = int((split_sets == frozenset({"train"})).sum())
    n_pure_test = int((split_sets == frozenset({"test"})).sum())
    print(f"\nEpisode split breakdown: {n_pure_train} pure-train, {n_pure_test} pure-test, "
          f"{n_straddling} straddling (excluded).")

    tier2_prob = test_probs[2]
    thresholds = np.linspace(0.01, 0.99, 197)
    episode_exposure = compute_episode_exposure(test_df, valid_ep_ids)

    calib_ep_ids = set(train_calib_df.loc[train_calib_df["episode_id"].notna(), "episode_id"].unique())
    calib_episode_exposure = compute_episode_exposure(train_calib_df, calib_ep_ids)

    calib_probs = {}
    for tier in (0, 1, 2):
        X_calib_t = assemble_features(train_calib_df, tier, calib_cat)
        if tier == 2:
            calib_probs[tier] = tier2_calibrator.predict_proba(X_calib_t)[:, 1]
        else:
            calib_probs[tier] = all_base_models[tier].predict_proba(X_calib_t)[:, 1]

    no_detection_cost = float(episode_exposure.reindex(list(valid_ep_ids)).fillna(0).sum())
    tier_best = {}
    for tier in (0, 1, 2):
        calib_sweep = sweep_cost_by_threshold(calib_probs[tier], train_calib_df, calib_ep_ids,
                                               calib_episode_exposure, thresholds)
        chosen_threshold = float(calib_sweep.loc[calib_sweep["total_cost"].idxmin(), "threshold"])
        test_sweep_at_chosen = sweep_cost_by_threshold(test_probs[tier], test_df, valid_ep_ids,
                                                        episode_exposure, np.array([chosen_threshold]))
        row = test_sweep_at_chosen.iloc[0]
        tier_best[tier] = dict(threshold=chosen_threshold, total_cost=float(row["total_cost"]),
                                fp_cost=float(row["fp_cost"]), fn_cost=float(row["fn_cost"]))
        print(f"Tier {tier} threshold (selected on calib): {chosen_threshold:.3f} "
              f"-> test cost Rs.{row['total_cost']:,.0f}")
        print(tier, tier_best[tier]["fp_cost"], tier_best[tier]["fn_cost"])

    calib_y = train_calib_df["is_attack"].values
    pr_constrained = max_recall_at_precision(calib_probs[2], calib_y, min_precision=0.95)
    if pr_constrained is not None:
        pr_threshold, _, _ = pr_constrained
        pred_pr_test = (tier2_prob >= pr_threshold).astype(int)
        pr_overall_test = confusion_counts(y_test, pred_pr_test)
        print(f"Tier 2 precision-constrained threshold: {pr_threshold:.3f} "
              f"-> test precision={pr_overall_test['precision']:.3f} recall={pr_overall_test['recall']:.3f}")
    else:
        pr_threshold, pr_overall_test = None, None

    print(f"\nNo-detection floor: Rs.{no_detection_cost:,.0f}")
    print("Waterfall: " + " -> ".join(
        [f"No detection Rs.{no_detection_cost:,.0f}"] +
        [f"Tier {t} Rs.{tier_best[t]['total_cost']:,.0f}" for t in (0, 1, 2)]
    ))

    best_threshold = tier_best[2]["threshold"]
    tier2_pred_at_best = (tier2_prob >= best_threshold).astype(int)
    tier2_overall_at_best = confusion_counts(y_test, tier2_pred_at_best)
    print(f"\nTier 2 at cost-optimal threshold {best_threshold:.3f}: "
          f"precision={tier2_overall_at_best['precision']:.3f} recall={tier2_overall_at_best['recall']:.3f}")

    cost_df = sweep_cost_by_threshold(tier2_prob, test_df, valid_ep_ids, episode_exposure, thresholds)
    tier2_pred_default = (tier2_prob >= DECISION_THRESHOLD).astype(int)
    default_row = cost_df.iloc[(cost_df["threshold"] - DECISION_THRESHOLD).abs().idxmin()]
    tier2_pred_optimal = (tier2_prob >= best_threshold).astype(int)

    family_rows = []
    for fam in sorted(test_df.loc[test_df["is_attack"] == 1, "attack_family"].dropna().unique()):
        mask = (test_df["attack_family"] == fam).values
        n_windows = int(mask.sum())
        n_episodes_all = test_df.loc[mask, "episode_id"].nunique()
        window_recall_default = float(tier2_pred_default[mask].mean()) if n_windows > 0 else float("nan")
        window_recall_optimal = float(tier2_pred_optimal[mask].mean()) if n_windows > 0 else float("nan")
        family_rows.append(dict(attack_family=fam, test_windows=n_windows, test_episodes=n_episodes_all,
                                 window_recall_at_0_5=window_recall_default,
                                 window_recall_at_optimal=window_recall_optimal))
    family_df = pd.DataFrame(family_rows).sort_values("test_windows", ascending=False)

    ep_detect_optimal = episode_level_detection(test_df, tier2_pred_optimal, valid_ep_ids)
    ep_summary = (ep_detect_optimal.groupby("attack_family")
                  .agg(n_test_episodes=("episode_id", "count"),
                       episode_detection_rate=("caught", "mean"),
                       median_time_to_detect_windows=("time_to_detect_windows", "median"))
                  .reset_index())
    print("\n=== Tier 2 window-level recall by family ===")
    print(family_df.to_string(index=False))
    print("\n=== Tier 2 episode-level detection by family ===")
    print(ep_summary.to_string(index=False))

    tier2_base_model.save_model(os.path.join(args.outdir, "tier2_model.json"))
    all_base_models[0].save_model(os.path.join(args.outdir, "tier0_model.json"))
    all_base_models[1].save_model(os.path.join(args.outdir, "tier1_model.json"))
    joblib.dump(tier2_calibrator, os.path.join(args.outdir, "tier2_calibrator.joblib"))
    joblib.dump(cat_encoder, os.path.join(args.outdir, "cat_encoder.joblib"))

    sample_n = min(3000, len(tier2_X_test))
    sample_idx = np.random.default_rng(42).choice(len(tier2_X_test), size=sample_n, replace=False)
    X_shap = tier2_X_test.iloc[sample_idx]
    explainer = shap.TreeExplainer(tier2_base_model)
    shap_values = explainer.shap_values(X_shap)
    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "shap_summary_tier2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    results["tier2_calibration"] = dict(
        calib_slices=int(train_calib_df["slice_id"].nunique()),
        calib_rows=int(len(train_calib_df)), calib_positives=int((y_calib == 1).sum()),
    )
    results["cost_model"] = dict(
        cost_per_false_alarm_inr=COST_PER_FALSE_ALARM,
        exposure_proxy="avg_amount * txn_count * (1 - decline_rate) per episode, estimated not measured",
        no_detection_floor_cost_inr=no_detection_cost,
        threshold_selection_note="thresholds selected on the calibration split, reported on test",
        tier_best=tier_best,
        tier2_at_cost_optimal_threshold=dict(
            threshold=best_threshold, precision=tier2_overall_at_best["precision"],
            recall=tier2_overall_at_best["recall"], fpr=tier2_overall_at_best["fpr"],
        ),
        tier2_precision_constrained=(
            dict(min_precision=0.95, threshold=pr_threshold, precision=pr_overall_test["precision"],
                 recall=pr_overall_test["recall"], fpr=pr_overall_test["fpr"])
            if pr_threshold is not None else None
        ),
        default_threshold=DECISION_THRESHOLD, tier2_default_threshold_cost_inr=float(default_row["total_cost"]),
        n_straddling_episodes_excluded=n_straddling,
    )
    with open(os.path.join(args.outdir, "metrics_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    family_df.to_csv(os.path.join(args.outdir, "recall_by_attack_family.csv"), index=False)
    ep_summary.to_csv(os.path.join(args.outdir, "episode_level_detection_by_family.csv"), index=False)
    cost_df.to_csv(os.path.join(args.outdir, "cost_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [results[t]["festive_fpr_overall"] for t in (0, 1, 2)]
    ax.bar(["Tier 0\n(naive)", "Tier 1\n(rule-based)", "Tier 2\n(full, calibrated)"], vals,
           color=["#c0392b", "#e67e22", "#27ae60"])
    ax.set_ylabel("False positive rate on legitimate festive surges")
    ax.set_title("Festive-surge false alarm rate by tier")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "festive_fpr_by_tier.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35; x = np.arange(3)
    ax.bar(x - width/2, [results[t]["festive_fpr_unseen_festival"] for t in (0,1,2)], width, label="Unseen festival")
    ax.bar(x + width/2, [results[t]["festive_fpr_unseen_slice"] for t in (0,1,2)], width, label="Unseen slice, known festival")
    ax.set_xticks(x); ax.set_xticklabels(["Tier 0", "Tier 1", "Tier 2"])
    ax.set_ylabel("Festive-surge FPR"); ax.set_title("Festive FPR by generalization axis"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "festive_fpr_by_tier_and_axis.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for t, label, color in [(0,"Tier 0 (naive)","#c0392b"), (1,"Tier 1 (rule-based)","#e67e22"), (2,"Tier 2 (calibrated)","#27ae60")]:
        prec, rec, _ = precision_recall_curve(y_test, test_probs[t])
        ax.plot(rec, prec, label=f"{label} (AP={results[t]['overall']['pr_auc']:.3f})", color=color)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve by tier")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "pr_curve_by_tier.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(family_df["attack_family"], family_df["window_recall_at_optimal"], color="#2980b9")
    for bar, (_, row) in zip(bars, family_df.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{row['window_recall_at_optimal']:.2f} (n={row['test_windows']} win / {row['test_episodes']} ep)",
                va="center", fontsize=8)
    ax.set_xlabel(f"Window-level recall (threshold={best_threshold:.2f})")
    ax.set_title("Window-level recall by attack family")
    ax.set_xlim(0, 1.2)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "recall_by_attack_family.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(ep_summary["attack_family"], ep_summary["episode_detection_rate"], color="#8e44ad")
    for bar, (_, row) in zip(bars, ep_summary.iterrows()):
        ttd = row["median_time_to_detect_windows"]
        ttd_str = f"{ttd:.0f}w" if pd.notna(ttd) else "n/a"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{row['episode_detection_rate']:.2f} (n={row['n_test_episodes']} ep, median TTD={ttd_str})",
                va="center", fontsize=8)
    ax.set_xlabel("Episode-level detection rate")
    ax.set_title("Episode-level detection by attack family")
    ax.set_xlim(0, 1.55)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "episode_detection_by_family.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["No detection", "Tier 0", "Tier 1", "Tier 2"]
    costs = [no_detection_cost, tier_best[0]["total_cost"], tier_best[1]["total_cost"], tier_best[2]["total_cost"]]
    colors = ["#7f8c8d", "#c0392b", "#e67e22", "#27ae60"]
    bars = ax.bar(labels, costs, color=colors)
    ax.set_yscale("log")
    for bar, c in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, c * 1.15, f"Rs.{c:,.0f}", ha="center", fontsize=8)
    ax.set_ylabel("Total expected cost (log scale)")
    ax.set_title("Cost by tier, each at its own best threshold")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "cost_waterfall_by_tier.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cost_df["threshold"], cost_df["total_cost"], color="#2c3e50")
    ax.axvline(best_threshold, color="#27ae60", linestyle="--", label=f"optimal t={best_threshold:.3f}")
    ax.axvline(DECISION_THRESHOLD, color="#c0392b", linestyle=":", label=f"default t={DECISION_THRESHOLD}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Total expected cost (Rs.)")
    ax.set_title("Tier 2 cost vs. threshold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "cost_vs_threshold_tier2_reference.png"), dpi=150); plt.close(fig)
    print(f"\nWrote metrics, models, and charts to {args.outdir}")
if __name__ == "__main__":
    main()