"""
Step 8 — Constraint Satisfaction Problem (CSP) Solver
=====================================================
Validates and refines the search agent's recommendation by solving a
CSP with backtracking.  Ensures the recommended academic plan is
FEASIBLE (practical and doable for the student).

AI concepts covered:
  • CSP formulation  (Variables, Domains, Constraints)
  • Backtracking Search  (systematic trial with pruning)
  • Constraint Checking  (is_consistent — early pruning of invalid branches)
  • Feasibility Validation  (search-path vs. expected path comparison)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATH = PROJECT_ROOT / "Output" / "prediction_output.json"
SEARCH_PATH     = PROJECT_ROOT / "Output" / "search_path.json"
OUTPUT_PATH     = PROJECT_ROOT / "Output" / "csp_solution.json"

# ── CSP Variables (Step 8) ───────────────────────────────────────
# Variables = the decisions we need to make for the student's plan
VARIABLES = [
    "study_hours",              # daily study hours
    "resource_sessions",        # library / resource sessions per week
    "tutoring_days",            # tutoring days per week
    "participation_sessions",   # class-participation sessions
    "monitoring_sessions",      # weekly check-ins
    "rest_hours",               # daily rest / sleep hours
]

# ── CSP Domains (Step 8) ────────────────────────────────────────
# Domains = finite set of allowed values for each variable
DOMAINS: Dict[str, List[int]] = {
    "study_hours":            [1, 2, 3, 4, 5],
    "resource_sessions":      [1, 2, 3, 4],
    "tutoring_days":          [0, 1, 2, 3],
    "participation_sessions": [1, 2, 3],
    "monitoring_sessions":    [1, 2, 3],
    "rest_hours":             [6, 7, 8],
}

# ── CSP Constraints (Step 8) ────────────────────────────────────
# 1. Risk-based minimums — higher risk → stricter minimums
# 2. Global workload cap  — study + resources + tutoring + participation ≤ 8
# 3. Rest requirement     — rest_hours ≥ 7

RISK_CONSTRAINTS = {
    "High Risk": {
        "study_hours": 3, "resource_sessions": 3,
        "tutoring_days": 2, "monitoring_sessions": 2, "rest_hours": 7,
    },
    "Medium Risk": {
        "study_hours": 2, "resource_sessions": 2,
        "tutoring_days": 1, "monitoring_sessions": 1, "rest_hours": 7,
    },
    "Low Risk": {
        "study_hours": 1, "resource_sessions": 1,
        "tutoring_days": 0, "monitoring_sessions": 1, "rest_hours": 7,
    },
}

# Expected BFS paths per risk level (from search_agent.py)
RECOMMENDED_PATHS = {
    "High Risk": [
        "High Risk", "Improve Attendance", "Increase Resource Usage",
        "Join Tutoring", "Weekly Monitoring", "Academic Safe Zone",
    ],
    "Medium Risk": [
        "Medium Risk", "Increase Resource Usage",
        "Increase Class Participation", "Weekly Monitoring", "Academic Safe Zone",
    ],
    "Low Risk": ["Low Risk", "Maintain Current Plan", "Academic Safe Zone"],
}


# ── Helpers ──────────────────────────────────────────────────────

def read_json(path: Path) -> Any:
    """Read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_value(payload: Any, keys: List[str]) -> Optional[Any]:
    """Recursively search JSON for a value matching any of *keys*."""
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                return payload[key]
        for value in payload.values():
            found = extract_value(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = extract_value(item, keys)
            if found is not None:
                return found
    return None


def normalize_class(value: Any) -> Optional[str]:
    """Convert prediction value to H, M, or L."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text in {"H", "M", "L"}:
        return text
    if "HIGH" in text:
        return "H"
    if "MEDIUM" in text:
        return "M"
    if "LOW" in text:
        return "L"
    return None


def map_risk(predicted_class: str) -> str:
    """Map H/M/L → risk level (reversed: H = Low Risk, L = High Risk)."""
    return {"H": "Low Risk", "M": "Medium Risk", "L": "High Risk"}.get(
        predicted_class, "Unknown"
    )


def path_is_feasible(search_path: List[str], risk_level: str) -> bool:
    """True when the search path matches the expected path for this risk."""
    return search_path == RECOMMENDED_PATHS.get(risk_level, [])


# ── Constraint Checking (Step 8) ────────────────────────────────
# A COMPLETE assignment must satisfy ALL constraints at once.

def satisfies_constraints(
    assignment: Dict[str, int], risk_level: str, search_path: List[str]
) -> bool:
    """Return True if a full assignment satisfies every constraint."""
    # 1. Risk-based minimum values
    for var, minimum in RISK_CONSTRAINTS[risk_level].items():
        if assignment[var] < minimum:
            return False

    # 2. Path-specific: participation must be ≥ 1 if path includes it
    if "Increase Class Participation" in search_path and assignment["participation_sessions"] < 1:
        return False

    # 3. Global workload cap
    workload = (
        assignment["study_hours"] + assignment["resource_sessions"]
        + assignment["tutoring_days"] + assignment["participation_sessions"]
    )
    if workload > 8:
        return False

    # 4. Rest requirement
    if assignment["rest_hours"] < 7:
        return False

    return True


# ── Backtracking Search (Step 8) ────────────────────────────────
# Backtracking systematically explores all possible assignments:
#   1. Pick the next unassigned variable
#   2. Try each value in its domain
#   3. Check consistency (partial constraint check → early pruning)
#   4. If consistent → recurse to the next variable
#   5. If inconsistent → UNDO assignment and try the next value
#   6. All variables assigned + constraints met → solution found

def backtracking_search(
    variables: List[str],
    domains: Dict[str, List[int]],
    risk_level: str,
    search_path: List[str],
    partial_assignment: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, int]]:
    """Solve the CSP via backtracking. Returns a valid assignment or None."""
    assignment = dict(partial_assignment or {})

    # -- Partial consistency check (prunes branches early) ---------
    def is_consistent(current: Dict[str, int]) -> bool:
        # Check minimums for already-assigned variables
        for var, minimum in RISK_CONSTRAINTS[risk_level].items():
            if var in current and current[var] < minimum:
                return False                   # violated → prune

        # Check workload cap when all activity vars are assigned
        activity = ["study_hours", "resource_sessions", "tutoring_days", "participation_sessions"]
        if all(v in current for v in activity):
            if sum(current[v] for v in activity) > 8:
                return False                   # overloaded → prune

        # Check rest when assigned
        if "rest_hours" in current and current["rest_hours"] < 7:
            return False

        return True                            # no violation so far

    # -- Recursive backtracker ------------------------------------
    def backtrack(idx: int) -> Optional[Dict[str, int]]:
        # Base case: all variables assigned → full constraint check
        if idx == len(variables):
            return dict(assignment) if satisfies_constraints(assignment, risk_level, search_path) else None

        var = variables[idx]
        for value in domains[var]:
            assignment[var] = value            # assign
            if is_consistent(assignment):      # check partial constraints
                result = backtrack(idx + 1)    # recurse
                if result is not None:
                    return result              # solution found!
            assignment.pop(var, None)          # BACKTRACK: undo

        return None                            # dead end

    return backtrack(0)


# ── Refine path if original was infeasible ───────────────────────

def refine_search_path(risk_level: str, search_path: List[str], feasible: bool) -> List[str]:
    """Fall back to the recommended path if the original is infeasible."""
    if feasible and search_path:
        return search_path
    return RECOMMENDED_PATHS.get(risk_level, search_path)


# ── Main: run the CSP solver pipeline ────────────────────────────

def main() -> None:
    """Read inputs → determine risk → check feasibility → solve CSP → save."""
    # 1. Read outputs from previous pipeline stages
    prediction_data = read_json(PREDICTION_PATH)
    search_data     = read_json(SEARCH_PATH)

    # 2. Extract predicted class and derive risk level
    raw = extract_value(prediction_data, ["predicted_class", "prediction", "class_prediction", "Class"])
    predicted_class = normalize_class(raw)
    if predicted_class is None:
        raise ValueError("Could not extract predicted class from prediction_output.json")
    risk_level = map_risk(predicted_class)

    # 3. Get the search path produced by search_agent.py
    search_path_val = extract_value(search_data, ["search_path"])
    if not isinstance(search_path_val, list):
        raise ValueError("search_path.json does not contain a valid search_path list")
    search_path = [str(s) for s in search_path_val]

    # 4. Check whether the search path matches the expected pattern
    original_feasible = path_is_feasible(search_path, risk_level)

    # 5. Solve CSP using backtracking
    csp_solution = backtracking_search(VARIABLES, DOMAINS, risk_level, search_path)
    if csp_solution is None:
        raise RuntimeError(f"No feasible CSP solution for risk level: {risk_level}")

    # 6. Refine path if needed
    refined_path = refine_search_path(risk_level, search_path, original_feasible)

    # 7. Determine status and explanation
    if original_feasible:
        status = "aligned_feasible"
        explanation = (
            f"The search-agent recommendation is feasible for {risk_level}. "
            f"The CSP found a valid academic support plan."
        )
    else:
        status = "refined_feasible"
        explanation = "The original search path needed adjustment. CSP generated a feasible refined plan."

    # 8. Print results
    print(f"Risk level: {risk_level}")
    print(f"Search path: {search_path}")
    print("CSP variables:", VARIABLES)
    print(f"CSP solution: {csp_solution}")
    print(f"Original path feasible: {original_feasible}")
    print(f"Plan status: {status}")

    # 9. Save output
    output = {
        "risk_level": risk_level,
        "search_path": refined_path,
        "csp_solution": csp_solution,
        "original_search_path_feasible": original_feasible,
        "csp_plan_found": True,
        "final_plan_status": status,
        "explanation": explanation,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved CSP solution to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
