"""
Step 7 — Intelligent Search Agent (BFS)
========================================
Reads the ML prediction, maps it to a risk level, builds a state-space
graph, and runs Breadth-First Search to find the shortest action path
from the student's current risk state to the "Academic Safe Zone".

AI concepts covered:
  • PEAS Framework  (Performance, Environment, Actuators, Sensors)
  • State-Space Representation  (states, actions, transitions, goal)
  • BFS Search Algorithm  (queue-based, level-by-level, shortest path)
"""

import json
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATH = PROJECT_ROOT / "Output" / "prediction_output.json"
SEARCH_OUTPUT_PATH = PROJECT_ROOT / "Output" / "search_path.json"

# ── PEAS Framework (Step 7 — Agent Specification) ────────────────
# P - Performance : Reduce academic risk and improve student outcomes
# E - Environment : Academic advising system using student data
# A - Actuators   : Improve Attendance, Increase Resource Usage,
#                   Join Tutoring, Weekly Monitoring, Maintain Plan
# S - Sensors     : ML prediction (H/M/L), absence days, raised
#                   hands, visited resources, discussions, etc.

# ── Risk Mapping ─────────────────────────────────────────────────
# ML predicts performance class → we reverse-map to risk:
#   H (high performance)  → Low Risk
#   M (medium)            → Medium Risk
#   L (low performance)   → High Risk
RISK_MAP = {"H": "Low Risk", "M": "Medium Risk", "L": "High Risk"}

# Keys that might hold the predicted class in the ML output JSON
PREDICTION_KEYS = ["predicted_class", "prediction", "class_prediction", "Class"]


# ── Helper: read JSON ────────────────────────────────────────────
def _read_json(path: Path) -> Any:
    """Read a JSON file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Helper: extract predicted class from nested JSON ─────────────
def _find_predicted_class(data: Any) -> Optional[str]:
    """Recursively search the JSON for the predicted class value."""
    if isinstance(data, dict):
        for key in PREDICTION_KEYS:          # check known keys first
            if key in data:
                return data[key]
        for v in data.values():              # recurse into nested dicts
            found = _find_predicted_class(v)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:                    # recurse into lists
            found = _find_predicted_class(item)
            if found is not None:
                return found
    return None


# ── Helper: normalise any prediction value to H / M / L ─────────
def _normalize_class(value: Any) -> Optional[str]:
    """Convert any prediction value to a single letter: H, M, or L."""
    if value is None:
        return None
    if isinstance(value, list) and value:
        value = value[0]
    upper = str(value).strip().upper()
    if not upper:
        return None
    if upper in ("H", "M", "L"):             # direct match
        return upper
    for word, letter in {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}.items():
        if word in upper:                     # word match
            return letter
    if upper.startswith("H"):
        return "H"
    if upper.startswith("M"):
        return "M"
    if upper.startswith("L"):
        return "L"
    return None


def map_predicted_to_risk(letter: str) -> str:
    """Map prediction letter (H/M/L) to risk level string."""
    return RISK_MAP.get(letter, "Unknown")


# ── State-Space Graph (Step 7 — states, actions, goal) ───────────
# State-Space formulation:
#   Initial state : student's risk level  (e.g. "High Risk")
#   Goal state    : "Academic Safe Zone"
#   Actions       : edges in the directed graph
#   Transition    : each action moves one step closer to the goal
# Higher-risk students require more intermediate actions.

def build_graph(risk_level: str) -> Dict[str, List[str]]:
    """Build a directed graph of actions for the given risk level."""
    graphs: Dict[str, Dict[str, List[str]]] = {
        "High Risk": {                        # most actions needed
            "High Risk": ["Improve Attendance"],
            "Improve Attendance": ["Increase Resource Usage"],
            "Increase Resource Usage": ["Join Tutoring"],
            "Join Tutoring": ["Weekly Monitoring"],
            "Weekly Monitoring": ["Academic Safe Zone"],
            "Academic Safe Zone": [],
        },
        "Medium Risk": {
            "Medium Risk": ["Increase Resource Usage"],
            "Increase Resource Usage": ["Increase Class Participation"],
            "Increase Class Participation": ["Weekly Monitoring"],
            "Weekly Monitoring": ["Academic Safe Zone"],
            "Academic Safe Zone": [],
        },
        "Low Risk": {                         # fewest actions
            "Low Risk": ["Maintain Current Plan"],
            "Maintain Current Plan": ["Academic Safe Zone"],
            "Academic Safe Zone": [],
        },
    }
    return graphs.get(risk_level, graphs["Medium Risk"])


# ── BFS — Breadth-First Search (Step 7) ──────────────────────────
# BFS guarantees the SHORTEST path (fewest actions) because it
# explores all nodes at depth d before moving to depth d+1.
#
# Data structure : FIFO queue (deque) — oldest path is expanded first
# Each queue element is a COMPLETE path (list of nodes visited so far)
# Visited set prevents revisiting the same node (avoids cycles)
#
# Why BFS over DFS?
#   BFS → optimal (shortest) for uniform-cost edges
#   DFS → may find a longer path first; not optimal here

def bfs(start: str, goal: str, graph: Dict[str, List[str]]) -> List[str]:
    """Return the shortest path from *start* to *goal* using BFS."""
    queue = deque([[start]])       # initialise queue with start path
    visited = set([start])         # track visited nodes

    while queue:
        path = queue.popleft()     # FIFO: dequeue oldest path
        node = path[-1]            # current node = last in path

        if node == goal:           # goal test
            return path

        for neighbor in graph.get(node, []):        # expand neighbours
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])     # enqueue extended path

    return []                      # no path found


# ── Main: run the full search-agent pipeline ─────────────────────
def main():
    """Read ML prediction → map risk → build graph → BFS → save."""
    # 1. Read ML prediction
    try:
        data = _read_json(PREDICTION_PATH)
    except Exception as e:
        print(f"Error reading prediction file: {e}")
        return

    # 2. Extract and normalise predicted class
    raw_pred = _find_predicted_class(data)
    pred_letter = _normalize_class(raw_pred)
    if pred_letter is None:
        print("Could not extract predicted class from prediction JSON.")
        return

    # 3. Map to risk level
    risk_level = map_predicted_to_risk(pred_letter)

    # 4. Build state-space graph and run BFS
    graph = build_graph(risk_level)
    search_path = bfs(risk_level, "Academic Safe Zone", graph)

    # 5. Format recommendation
    final_recommendation = " -> ".join(search_path) if search_path else "No path found"

    # 6. Print results
    print(f"Predicted class: {pred_letter}")
    print(f"Risk level: {risk_level}")
    print(f"Search path: {search_path}")
    print(f"Final recommendation: {final_recommendation}")

    # 7. Save output JSON
    output = {
        "predicted_class": pred_letter,
        "risk_level": risk_level,
        "search_path": search_path,
        "final_recommendation": final_recommendation,
    }
    try:
        with SEARCH_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved search path to: {SEARCH_OUTPUT_PATH}")
    except Exception as e:
        print(f"Failed to save search path: {e}")


if __name__ == "__main__":
    main()
