import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(results: pd.DataFrame, ground_truth_path: str = "data/labeled_ground_truth.csv"):
    """
    Evaluates model output against ground truth bulk_deal labels.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain columns: symbol, date, final_risk_score
        This is the output of calculate_final_risk() from risk_scoring.py

    ground_truth_path : str
        Path to labeled_ground_truth.csv produced by ground_truth.py
        Must contain columns: symbol, date, is_fraud

    Returns
    -------
    pd.DataFrame
        Merged dataframe with ground truth + predictions (useful for inspection)
    """

    gt_path = Path(ground_truth_path)
    if not gt_path.is_absolute():
        gt_path = Path(__file__).resolve().parent.parent / gt_path

    # ── Load ground truth ────────────────────────────────────────────────────
    try:
        gt = pd.read_csv(gt_path, parse_dates=["date"])
    except FileNotFoundError:
        print(f"ERROR: Ground truth file not found at '{ground_truth_path}'")
        print("       Run ground_truth.py first to generate it.")
        return None

    # ── Align dtypes before merge ────────────────────────────────────────────
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    gt["date"] = pd.to_datetime(gt["date"])

    # ── Merge on symbol + date ───────────────────────────────────────────────
    merged = df.merge(gt, on=["symbol", "date"], how="left")
    merged["is_fraud"] = merged["is_fraud"].fillna(0).astype(int)

    if merged["is_fraud"].sum() == 0:
        print("WARNING: No ground truth fraud days matched in this dataset window.")
        print("         Check that your data dates overlap with labeled_ground_truth.csv")
        return merged

    y_true = merged["is_fraud"]
    y_score = merged["final_risk_score"]

    # ── Find optimal threshold (maximizes F1) ────────────────────────────────
    best_f1, best_threshold = 0, 0
    for pct in range(70, 100):
        thresh = np.percentile(y_score, pct)
        y_pred_temp = (y_score >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_temp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    merged["predicted_fraud"] = (y_score >= best_threshold).astype(int)
    y_pred = merged["predicted_fraud"]

    # ── Confusion matrix breakdown ───────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── Metrics ─────────────────────────────────────────────────────────────
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auprc     = average_precision_score(y_true, y_score)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = None

    # ── Print report ─────────────────────────────────────────────────────────
    _print_report(
        total=len(merged),
        fraud_count=int(y_true.sum()),
        flagged_count=int(y_pred.sum()),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        auprc=auprc,
        roc_auc=roc_auc,
        threshold=best_threshold,
    )

    # ── Print missed fraud days ──────────────────────────────────────────────
    missed = merged[(merged["is_fraud"] == 1) & (merged["predicted_fraud"] == 0)]
    if not missed.empty:
        print("\n  Missed fraud days (false negatives):")
        print(
            missed[["date", "symbol", "final_risk_score", "risk_category"]]
            .sort_values("final_risk_score", ascending=False)
            .to_string(index=False)
        )

    false_alarms = merged[(merged["is_fraud"] == 0) & (merged["predicted_fraud"] == 1)]
    if not false_alarms.empty:
        print(f"\n  False alarms (flagged but clean): {len(false_alarms)} days")

    # ── Return metrics dict for UI ───────────────────────────────────────────
    metrics = {
        "total_days" : len(merged),
        "fraud_days" : int(y_true.sum()),
        "flagged"    : int(y_pred.sum()),
        "tp"         : int(tp),
        "fp"         : int(fp),
        "fn"         : int(fn),
        "tn"         : int(tn),
        "precision"  : round(precision, 3),
        "recall"     : round(recall, 3),
        "f1"         : round(f1, 3),
        "auprc"      : round(auprc, 3),
        "roc_auc"    : round(roc_auc, 3) if roc_auc is not None else None,
        "baseline"   : round(int(y_true.sum()) / len(merged), 3),
    }

    return merged, metrics


def _print_report(total, fraud_count, flagged_count,
                  tp, fp, fn, tn,
                  precision, recall, f1, auprc, roc_auc, threshold):

    bar = "═" * 48

    print(f"\n{bar}")
    print("  MODEL EVALUATION REPORT")
    print(bar)
    print(f"  Total days evaluated : {total}")
    print(f"  Known fraud days     : {fraud_count}  (ground truth)")
    print(f"  Flagged by model     : {flagged_count}  (threshold = {threshold:.2f})")
    print()
    print(f"  {'True Positives':<28} {tp:>4}  ← caught fraud")
    print(f"  {'False Positives':<28} {fp:>4}  ← false alarms")
    print(f"  {'False Negatives':<28} {fn:>4}  ← missed fraud  ⚠")
    print(f"  {'True Negatives':<28} {tn:>4}  ← correctly clean")
    print()
    print(f"  {'Precision':<28} {precision:.3f}")
    print(f"    → of all flags raised, {precision*100:.1f}% were real fraud")
    print()
    print(f"  {'Recall':<28} {recall:.3f}")
    print(f"    → of all real fraud days, caught {recall*100:.1f}%")
    print()
    print(f"  {'F1 Score':<28} {f1:.3f}")
    print()
    print(f"  {'AUPRC':<28} {auprc:.3f}  ← KEY metric for fraud")
    print(f"    → random baseline ≈ {fraud_count/total:.3f}  |  perfect = 1.000")
    if roc_auc is not None:
        print(f"  {'ROC-AUC':<28} {roc_auc:.3f}")
    print(bar)
    _print_score_interpretation(auprc, recall, precision)


def _print_score_interpretation(auprc, recall, precision):
    print()
    if auprc >= 0.7:
        print("  ✓ Strong signal — model is meaningfully detecting fraud patterns.")
    elif auprc >= 0.4:
        print("  ~ Moderate signal — model is learning something but needs more data")
        print("    or better features to be reliable.")
    else:
        print("  ✗ Weak signal — model is close to random for fraud detection.")
        print("    Consider: more labeled data, better features, or tuning contamination.")

    if recall < 0.5:
        print("  ⚠ Low recall — model is missing more than half of real fraud days.")
        print("    For a fraud detector, missing fraud is costly. Prioritize recall.")
    if precision < 0.3:
        print("  ⚠ Low precision — most flags are false alarms.")
        print("    Consider raising the contamination threshold in IsolationForest.")
    print()