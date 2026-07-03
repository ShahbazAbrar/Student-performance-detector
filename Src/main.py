"""
Step 10 — Final Integration & System Summary
=============================================
Loads artifacts from every pipeline stage (ML prediction, BFS search
agent, CSP solver, knowledge-based inference) and prints/saves a
consolidated end-to-end summary.

Pipeline workflow:
  Dataset → Preprocessing → ML Prediction → Risk Mapping →
  BFS Search Agent → CSP Validation → Forward-Chaining Knowledge
  System → Final Recommendation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# ── Paths to all pipeline outputs ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "Output"

PREDICTION_PATH    = OUTPUT_DIR / "prediction_output.json"   # Step 6: ML prediction
SEARCH_PATH        = OUTPUT_DIR / "search_path.json"         # Step 7: BFS search agent
CSP_PATH           = OUTPUT_DIR / "csp_solution.json"        # Step 8: CSP solver
KNOWLEDGE_PATH     = OUTPUT_DIR / "knowledge_inference.json" # Step 9: knowledge base
MODEL_RESULTS_PATH = OUTPUT_DIR / "model_results.csv"        # ML model comparison
SUMMARY_PATH       = OUTPUT_DIR / "final_system_summary.txt" # final output


# ── Helpers ──────────────────────────────────────────────────────

def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_model_results(path: Path) -> pd.DataFrame:
    """Read the ML model comparison CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def build_best_model_name(model_results: pd.DataFrame) -> str:
    """Pick the best model by F1-score (tie-break: accuracy)."""
    if model_results.empty:
        return "Unknown"
    if "f1_score" in model_results.columns:
        best = model_results.sort_values(["f1_score", "accuracy"], ascending=False).iloc[0]
        return str(best["model"])
    return str(model_results.iloc[0].get("model", "Unknown"))


def format_workflow() -> str:
    """Return a human-readable pipeline string."""
    return (
        "Dataset -> Preprocessing -> ML Prediction -> Risk Mapping -> BFS Search Agent -> "
        "CSP Validation -> Forward Chaining Knowledge System -> Final Recommendation"
    )


# ── Combine recommendations from all AI components ───────────────

def build_final_combined_recommendation(
    knowledge_data: Dict[str, Any], csp_data: Dict[str, Any], search_data: Dict[str, Any]
) -> str:
    """Merge search-agent path, CSP plan, and knowledge-base guidance."""
    kb_rec  = str(knowledge_data.get("final_recommendation", ""))
    csp_ok  = bool(csp_data.get("csp_plan_found", False))
    status  = str(csp_data.get("final_plan_status", "unknown"))
    plan    = csp_data.get("csp_solution", {})
    path    = search_data.get("search_path", [])

    if csp_ok:
        return (
            f"Aligned recommendation ({status}): follow the search path {path}, "
            f"implement the CSP plan {plan}, and apply the knowledge-base guidance: {kb_rec}"
        )
    return (
        f"Use the knowledge-base recommendation {kb_rec}, then refine the plan to satisfy the "
        f"CSP constraints in {plan}; the original search path {path} needs adjustment. "
        f"(CSP status={status})"
    )


# ── Main: end-to-end summary (Step 10) ──────────────────────────

def main() -> None:
    """Load all outputs, print summary, and write final report."""
    print("Student Performance Prediction and Academic Advising System")
    print()
    print(format_workflow())
    print()

    # Load outputs from every pipeline stage
    prediction_data = read_json(PREDICTION_PATH)
    search_data     = read_json(SEARCH_PATH)
    csp_data        = read_json(CSP_PATH)
    knowledge_data  = read_json(KNOWLEDGE_PATH)
    model_results   = read_model_results(MODEL_RESULTS_PATH)

    # Extract key fields
    best_model  = build_best_model_name(model_results)
    pred_class  = str(prediction_data.get("predicted_class", "Unknown"))
    risk_level  = str(search_data.get("risk_level", "Unknown"))
    search_path = search_data.get("search_path", [])
    csp_plan    = csp_data.get("csp_solution", {})
    kb_rec      = str(knowledge_data.get("final_recommendation", "Unknown"))
    combined    = build_final_combined_recommendation(knowledge_data, csp_data, search_data)

    # Print consolidated summary
    print(f"Best model name: {best_model}")
    print()
    print("Model comparison table:")
    print(model_results.to_string(index=False))
    print()
    print(f"Predicted class: {pred_class}")
    print(f"Risk level: {risk_level}")
    print(f"Search path: {search_path}")
    print(f"CSP feasible plan: {csp_plan}")
    print(f"Knowledge inference final recommendation: {kb_rec}")
    print(f"Final combined recommendation: {combined}")

    # Write summary file
    summary_lines = [
        "Student Performance Prediction and Academic Advising System",
        "",
        format_workflow(),
        "",
        f"Best model name: {best_model}",
        f"Predicted class: {pred_class}",
        f"Risk level: {risk_level}",
        f"Search path: {search_path}",
        f"CSP feasible plan: {csp_plan}",
        f"Knowledge inference final recommendation: {kb_rec}",
        f"Final combined recommendation: {combined}",
        "",
        "Model comparison table:",
        model_results.to_string(index=False),
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print()
    print(f"Saved final summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()