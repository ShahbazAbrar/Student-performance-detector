"""
Step 9 — Knowledge-Based System (Forward Chaining)
===================================================
Implements a rule-based inference engine using forward chaining.
Reads ML, search-agent, and CSP outputs, constructs initial facts,
fires IF-THEN rules until no new facts appear, and produces an
inference trace and final recommendation.

AI concepts covered:
  • IF-THEN Rules  (condition → derived fact)
  • Forward Chaining  (data-driven: start with facts, apply rules)
  • Inference Trace  (log of which rules fired and why)
  • Fact Propagation  (new facts can trigger further rules)
  • Contrasting Examples  (same engine, different inputs → different conclusions)

Forward-chaining overview (viva):
  1. Start with known facts
  2. Scan all rules — if a rule's condition matches current facts, FIRE it
  3. Firing a rule adds a NEW fact to the knowledge base
  4. Repeat until no new facts are produced (fixed point)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "Output"
PREDICTION_PATH = OUTPUT_DIR / "prediction_output.json"
SEARCH_PATH     = OUTPUT_DIR / "search_path.json"
CSP_PATH        = OUTPUT_DIR / "csp_solution.json"
OUTPUT_PATH     = OUTPUT_DIR / "knowledge_inference.json"


# ── Load pipeline outputs ────────────────────────────────────────

def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_extract(data: Any, keys: List[str]) -> Optional[Any]:
    """Recursively search nested JSON for any of *keys*."""
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        for value in data.values():
            extracted = safe_extract(value, keys)
            if extracted is not None:
                return extracted
    elif isinstance(data, list):
        for item in data:
            extracted = safe_extract(item, keys)
            if extracted is not None:
                return extracted
    return None


def normalize_predicted_class(value: Any) -> Optional[str]:
    """Convert any prediction value to H, M, or L."""
    if value is None:
        return None
    text = str(value).strip().upper()
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
    """H → Low Risk, M → Medium Risk, L → High Risk."""
    return {"H": "Low Risk", "M": "Medium Risk", "L": "High Risk"}.get(
        predicted_class, "Unknown"
    )


def extract_context() -> Dict[str, Any]:
    """Read ML, search, and CSP outputs into a single context dict."""
    prediction_data = read_json(PREDICTION_PATH)
    search_data = read_json(SEARCH_PATH)
    csp_data = read_json(CSP_PATH)

    raw = safe_extract(prediction_data, ["predicted_class", "prediction", "class_prediction", "Class"])
    predicted_class = normalize_predicted_class(raw)
    if predicted_class is None:
        raise ValueError("Could not extract a valid predicted class")

    return {
        "predicted_class": predicted_class,
        "risk_level": safe_extract(search_data, ["risk_level"]) or map_risk(predicted_class),
        "search_path": safe_extract(search_data, ["search_path"]),
        "csp_solution": safe_extract(csp_data, ["csp_solution"]) or {},
    }


# ── Build feature profiles for the two examples ─────────────────
# Primary example  → uses the actual risk level (medium or high risk)
# Contrasting case → always low-risk with positive features

def build_feature_profile(risk_level: str, contrasting: bool = False) -> Dict[str, Any]:
    """Return a simulated student feature profile."""
    if not contrasting and risk_level == "Medium Risk":
        return {
            "StudentAbsenceDays": "Above-7", "raisedhands": "low",
            "VisITedResources": "low", "AnnouncementsView": "moderate",
            "Discussion": "low", "ParentAnsweringSurvey": "No",
            "ParentschoolSatisfaction": "Bad",
        }
    if not contrasting and risk_level == "High Risk":
        return {
            "StudentAbsenceDays": "Above-7", "raisedhands": "very-low",
            "VisITedResources": "very-low", "AnnouncementsView": "low",
            "Discussion": "low", "ParentAnsweringSurvey": "No",
            "ParentschoolSatisfaction": "Bad",
        }
    # Low Risk / contrasting — positive engagement across the board
    return {
        "StudentAbsenceDays": "Under-7", "raisedhands": "high",
        "VisITedResources": "high", "AnnouncementsView": "high",
        "Discussion": "high", "ParentAnsweringSurvey": "Yes",
        "ParentschoolSatisfaction": "Good",
    }


# ── Create initial facts (Step 9) ───────────────────────────────
# Facts = the starting knowledge the inference engine works with.
# They combine ML prediction, search path, CSP plan, and features.

def initial_facts_from_context(context: Dict[str, Any], contrasting: bool = False) -> Dict[str, Any]:
    """Build the initial fact set for forward chaining."""
    feature_profile = build_feature_profile(context["risk_level"], contrasting=contrasting)

    # Contrasting example overrides to Low Risk with ideal CSP values
    csp_solution = context["csp_solution"] if not contrasting else {
        "study_hours": 4, "resource_sessions": 3, "tutoring_days": 0,
        "participation_sessions": 2, "monitoring_sessions": 1, "rest_hours": 8,
    }

    return {
        "predicted_class": context["predicted_class"] if not contrasting else "H",
        "risk_level": context["risk_level"] if not contrasting else "Low Risk",
        "search_path": context["search_path"] if not contrasting else [
            "Low Risk", "Maintain Current Plan", "Academic Safe Zone"
        ],
        "csp_solution": csp_solution,
        **feature_profile,
    }


# ── Rule data structure ─────────────────────────────────────────

RuleCondition = Callable[[Dict[str, Any]], bool]
RuleAction    = Callable[[Dict[str, Any]], Tuple[Optional[str], List[str]]]

@dataclass(frozen=True)
class Rule:
    name: str
    condition: RuleCondition          # IF this is true …
    action: RuleAction                # THEN derive this fact


def value_at_least(value: Any, threshold: int) -> bool:
    """Safely check if a numeric value meets a threshold."""
    try:
        return int(value) >= threshold
    except (TypeError, ValueError):
        return False


# ── IF-THEN Rules (Step 9 — 12 rules + 8 derived-fact rules) ────
# Rules R1–R10  : derive basic facts from raw features
# Rules R11–R13 : check CSP plan adequacy
# Rules R14–R20 : chain derived facts into recommendations
#
# Viva note: rules that depend on facts produced by OTHER rules
# demonstrate forward-chaining's propagation behaviour.

def build_rules() -> List[Rule]:
    """Return the complete list of IF-THEN rules."""
    return [
        # ── Risk classification rules ────────────────────────────
        Rule("R1_predicted_low_risk",
             lambda f: f.get("predicted_class") == "H",
             lambda f: ("student_is_low_risk", ["predicted_class = H -> student_is_low_risk"])),
        Rule("R2_predicted_medium_risk",
             lambda f: f.get("predicted_class") == "M",
             lambda f: ("student_is_medium_risk", ["predicted_class = M -> student_is_medium_risk"])),
        Rule("R3_predicted_high_risk",
             lambda f: f.get("predicted_class") == "L",
             lambda f: ("student_is_high_risk", ["predicted_class = L -> student_is_high_risk"])),

        # ── Feature-based rules ──────────────────────────────────
        Rule("R4_attendance_problem",
             lambda f: f.get("StudentAbsenceDays") == "Above-7",
             lambda f: ("attendance_problem", ["StudentAbsenceDays = Above-7 -> attendance_problem"])),
        Rule("R5_low_participation",
             lambda f: str(f.get("raisedhands", "")).lower() in {"low", "very-low"},
             lambda f: ("low_participation", ["raisedhands is low -> low_participation"])),
        Rule("R6_low_resource_usage",
             lambda f: str(f.get("VisITedResources", "")).lower() in {"low", "very-low"},
             lambda f: ("low_resource_usage", ["VisITedResources is low -> low_resource_usage"])),
        Rule("R7_low_announcement_engagement",
             lambda f: str(f.get("AnnouncementsView", "")).lower() == "low",
             lambda f: ("low_announcement_engagement", ["AnnouncementsView = low -> low_announcement_engagement"])),
        Rule("R8_low_discussion_engagement",
             lambda f: str(f.get("Discussion", "")).lower() == "low",
             lambda f: ("low_discussion_engagement", ["Discussion = low -> low_discussion_engagement"])),
        Rule("R9_low_parent_involvement",
             lambda f: str(f.get("ParentAnsweringSurvey", "")).strip().lower() == "no",
             lambda f: ("low_parent_involvement", ["ParentAnsweringSurvey = No -> low_parent_involvement"])),
        Rule("R10_home_support_gap",
             lambda f: str(f.get("ParentschoolSatisfaction", "")).strip().lower() == "bad",
             lambda f: ("home_support_gap", ["ParentschoolSatisfaction = Bad -> home_support_gap"])),

        # ── CSP-plan adequacy rules ──────────────────────────────
        Rule("R11_adequate_study_plan",
             lambda f: value_at_least(f.get("study_hours"), 3),
             lambda f: ("adequate_study_plan", ["study_hours >= 3 -> adequate_study_plan"])),
        Rule("R12_tutoring_support_active",
             lambda f: value_at_least(f.get("tutoring_days"), 1),
             lambda f: ("tutoring_support_active", ["tutoring_days >= 1 -> tutoring_support_active"])),
        Rule("R13_weekly_monitoring_active",
             lambda f: value_at_least(f.get("monitoring_sessions"), 1),
             lambda f: ("weekly_monitoring_active", ["monitoring_sessions >= 1 -> weekly_monitoring_active"])),

        # ── Derived / chaining rules ─────────────────────────────
        # R14 fires if ANY risk signal exists → shows propagation
        Rule("R14_intervention_required",
             lambda f: any(lbl in f for lbl in [
                 "student_is_high_risk", "attendance_problem",
                 "low_participation", "low_resource_usage",
                 "low_parent_involvement", "home_support_gap",
             ]),
             lambda f: ("intervention_required", ["risk or engagement signals -> intervention_required"])),
        Rule("R15_recommend_attendance_plan",
             lambda f: "attendance_problem" in f,
             lambda f: ("recommend_attendance_plan", ["attendance_problem -> recommend_attendance_plan"])),
        Rule("R16_recommend_resource_usage",
             lambda f: "low_resource_usage" in f,
             lambda f: ("recommend_resource_usage", ["low_resource_usage -> recommend_resource_usage"])),

        # R17 chains R14 + R12 → plan ready
        Rule("R17_support_plan_ready",
             lambda f: "intervention_required" in f and "tutoring_support_active" in f,
             lambda f: ("academic_support_plan_ready",
                        ["intervention_required and tutoring_support_active -> academic_support_plan_ready"])),
        # R18 chains R17 + R13 → final recommendation
        Rule("R18_final_recommendation_ready",
             lambda f: "academic_support_plan_ready" in f and "weekly_monitoring_active" in f,
             lambda f: ("final_recommendation_ready",
                        ["academic_support_plan_ready and weekly_monitoring_active -> final_recommendation_ready"])),

        # ── Low-risk stability rules (contrasting case) ──────────
        Rule("R19_low_risk_stability",
             lambda f: f.get("risk_level") == "Low Risk" and f.get("StudentAbsenceDays") == "Under-7",
             lambda f: ("stable_performance", ["Low Risk and good attendance -> stable_performance"])),
        Rule("R20_maintain_current_plan",
             lambda f: "stable_performance" in f and str(f.get("raisedhands", "")).lower() == "high",
             lambda f: ("maintain_current_plan",
                        ["stable_performance and high participation -> maintain_current_plan"])),
    ]


# ── Forward-Chaining Engine (Step 9) ────────────────────────────
# Repeatedly scans rules and fires any whose condition is met.
# Stops when a full pass produces NO new facts (fixed point).
# Each fired rule is recorded in the inference trace.

def forward_chain(
    initial_facts: Dict[str, Any], rules: List[Rule]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run forward chaining. Returns (final_facts, inference_trace)."""
    facts = dict(initial_facts)
    trace: List[Dict[str, Any]] = []        # inference trace for output
    fired: set = set()                       # rules that already fired

    changed = True
    while changed:                           # keep looping until stable
        changed = False
        for rule in rules:
            if rule.name in fired:
                continue                     # each rule fires at most once
            if rule.condition(facts):        # IF condition is satisfied …
                new_fact, justification = rule.action(facts)   # … THEN act
                fired.add(rule.name)
                trace.append({"rule": rule.name, "justification": justification, "new_fact": new_fact})
                if new_fact and new_fact not in facts:
                    facts[new_fact] = True    # add derived fact
                    changed = True            # new fact may trigger more rules

    return facts, trace


# ── Final recommendation ─────────────────────────────────────────

def derive_final_recommendation(facts: Dict[str, Any]) -> str:
    """Select a recommendation based on what facts have been inferred."""
    if "final_recommendation_ready" in facts:
        return (
            "Structured intervention plan: improve attendance, increase resource usage, "
            "join tutoring, and keep weekly monitoring until the student reaches the academic safe zone."
        )
    if "maintain_current_plan" in facts or facts.get("risk_level") == "Low Risk":
        return "Maintain the current plan with light monitoring and sustain good academic habits."
    if "intervention_required" in facts:
        return "Immediate academic intervention required with attendance, resource, and tutoring support."
    return "No additional intervention required; continue monitoring and preserve effective study habits."


# ── Run a single inference case ──────────────────────────────────

def run_case(label: str, context: Dict[str, Any], contrasting: bool = False) -> Dict[str, Any]:
    """Build facts, run forward chaining, and return the results dict."""
    initial_facts = initial_facts_from_context(context, contrasting=contrasting)
    inferred_facts, trace = forward_chain(initial_facts, build_rules())
    recommendation = derive_final_recommendation(inferred_facts)

    # Print inference trace (useful for viva demonstration)
    print(f"Example case: {label}")
    print("Initial facts:")
    print(json.dumps(initial_facts, indent=2))
    print("Inference trace:")
    for step in trace:
        print(f"- {step['rule']}: {', '.join(step['justification'])}")
    print(f"Final recommendation: {recommendation}")

    return {
        "example_case_label": label,
        "initial_facts": initial_facts,
        "rules_fired": [s["rule"] for s in trace],
        "inferred_facts": sorted([k for k, v in inferred_facts.items() if v is True]),
        "final_recommendation": recommendation,
        "inference_trace": trace,
    }


# ── Main: two contrasting examples (Step 9) ─────────────────────
# Example 1 (primary)     : uses actual ML prediction (likely medium/high risk)
# Example 2 (contrasting) : forced low-risk student with ideal features
# The SAME rules and engine produce DIFFERENT conclusions — showing
# how forward chaining adapts to different initial facts.

def main() -> None:
    """Run inference on the actual case and a contrasting case, then save."""
    context = extract_context()

    primary_case     = run_case("medium_risk_actual_case",    context, contrasting=False)
    contrasting_case = run_case("low_risk_contrasting_case",  context, contrasting=True)

    output = {
        "example_case_label": primary_case["example_case_label"],
        "initial_facts": primary_case["initial_facts"],
        "rules_fired": primary_case["rules_fired"],
        "inferred_facts": primary_case["inferred_facts"],
        "final_recommendation": primary_case["final_recommendation"],
        "comparison_examples": [primary_case, contrasting_case],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Saved knowledge inference to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()