import pandas as pd
from pathlib import Path
from data_loader import load_all_data
from feature import rename_columns, add_manipulation_features, add_market_cap, get_yfinance_ticker
from train import train_model, MODEL_FEATURES
from risk_scoring import calculate_final_risk
from ground_truth import build_ground_truth
from evaluate import evaluate_model
from ui import show_results
from datetime import datetime
import os

GROUND_TRUTH_PATH = "data/labeled_ground_truth.csv"
timestamp = datetime.now().strftime("%Y_%m_%d")
MODEL_PATH = f"models/isolation_forest_{timestamp}.pkl"


def build_summary(results):
    """
    Aggregates per-day results into one row per stock.
    Risk category is driven by a weighted blend:
      - 60% max_risk_score  (peak danger — how bad did it get?)
      - 40% manipulation_rate  (frequency — how often was it suspicious?)
    This prevents a stock with one extreme day from being dismissed,
    and prevents a mildly elevated stock from being over-flagged.
    """
    summary = (
        results.groupby("symbol")
        .agg(
            avg_risk_score   = ("final_risk_score", "mean"),
            max_risk_score   = ("final_risk_score", "max"),
            manipulated_days = ("is_manipulated",   "sum"),
            total_days       = ("is_manipulated",   "count"),
        )
        .reset_index()
    )

    summary["manipulation_rate"] = (
        summary["manipulated_days"] / summary["total_days"] * 100
    ).round(1)

    summary["avg_risk_score"] = summary["avg_risk_score"].round(2)
    summary["max_risk_score"] = summary["max_risk_score"].round(2)

    # Weighted blend score — drives the risk category label
    # max_risk_score is already 0–100
    # manipulation_rate is 0–100 (it's a percentage)
    summary["category_score"] = (
        0.6 * summary["max_risk_score"] +
        0.4 * summary["manipulation_rate"]
    ).round(2)

    def risk_label(score):
        if score >= 70:
            return "High Risk"
        elif score >= 45:
            return "Moderate Risk"
        elif score >= 20:
            return "Low Risk"
        else:
            return "Normal"

    summary["risk_category"]  = summary["category_score"].apply(risk_label)
    summary["is_manipulated"] = summary["manipulated_days"] > 0

    return summary


if __name__ == "__main__":

    # ── 1. Load & prepare data ───────────────────────────────────────────────
    print("Loading data...")
    df = load_all_data()
    df = rename_columns(df)
    symbol = get_yfinance_ticker(df)
    df = add_market_cap(df, symbol)
    df = add_manipulation_features(df)

    # ── 2. Train / load model ────────────────────────────────────────────────
    model = train_model(df, MODEL_PATH, force_retrain=False)

    # ── 3. Score every row ───────────────────────────────────────────────────
    X = df[MODEL_FEATURES].fillna(0)
    ml_scores = model.decision_function(X)

    # ── 4. Build per-stock summary ───────────────────────────────────────────
    results = calculate_final_risk(df, ml_scores)
    summary = build_summary(results)

    # Terminal: full detail
    print("\n── Stock Manipulation Summary ──────────────────────────────────")
    display_cols = ["symbol", "is_manipulated", "avg_risk_score",
                    "max_risk_score", "manipulation_rate", "risk_category"]
    print(summary[display_cols].sort_values("max_risk_score", ascending=False).to_string(index=False))

    # ── 5. Ground truth + evaluation (terminal only) ─────────────────────────
    print("\n── Building ground truth labels ────────────────────────────────")
    build_ground_truth(data_folder="data", output_path=GROUND_TRUTH_PATH)

    eval_result = evaluate_model(results, ground_truth_path=GROUND_TRUTH_PATH)
    eval_metrics = None
    if eval_result is not None:
        _, eval_metrics = eval_result

    # ── 6. Launch UI window ──────────────────────────────────────────────────
    print("\nLaunching results window...")
    show_results(
        summary_df=summary[display_cols].sort_values("max_risk_score", ascending=False),
        eval_metrics=eval_metrics,
    )