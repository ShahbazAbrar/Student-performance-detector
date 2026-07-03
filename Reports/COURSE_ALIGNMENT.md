# Course Alignment — Student Performance AI ML Project

This document maps course topics to where they appear in the project.

Course Topic | Used in Project | File / Evidence
---|---|---
Python data structures | Yes | `Src/knowledge_base.py` (dict facts), `Output/*.json` (JSON outputs)
dictionaries / JSON-like outputs | Yes | `Output/prediction_output.json`, `Output/search_path.json`
loops | Yes | loops appear across all scripts, e.g., `for` in `ml_pipeline.py`
Pandas | Yes | `Src/ml_pipeline.py` (data loading, EDA, preprocessing)
NumPy | Yes | `Src/ml_pipeline.py` (numeric arrays, scaling)
BFS | Yes | `Src/search_agent.py` (bfs function)
state-space search | Yes | `Src/search_agent.py` (graph representation)
backtracking | Yes | `Src/csp_solver.py` (backtracking_search)
rule-based reasoning | Yes | `Src/knowledge_base.py` (IF-THEN rules)
forward chaining | Yes | `Src/knowledge_base.py` (forward_chain engine)

dataset exploration | Yes | `Src/ml_pipeline.py` (perform_eda, print_dataset_summary)
Pandas data handling | Yes | `Src/ml_pipeline.py` (dataframes, missing values)
missing values | Yes | `Src/ml_pipeline.py` (SimpleImputer in preprocessor)
feature scaling | Yes | `Src/ml_pipeline.py` (StandardScaler)
train/test split | Yes | `Src/ml_pipeline.py` (train_test_split)
EDA | Yes | `Src/ml_pipeline.py` (plots saved to Output/Figures)
Logistic Regression | Yes | `Src/ml_pipeline.py` (baseline model)
Decision Tree | Yes | `Src/ml_pipeline.py` (intermediate model)
Random Forest | Yes | `Src/ml_pipeline.py` (advanced model)
model comparison | Yes | `Src/ml_pipeline.py` (model_results.csv)
classification metrics | Yes | `Src/ml_pipeline.py` (accuracy, precision, recall, F1)
ROC-AUC | Yes | `Src/ml_pipeline.py` (ROC plotting & auc)
K-Means clustering | Yes | `Src/ml_pipeline.py` (run_kmeans on engagement features)

Notes:
- The project avoids advanced algorithms outside the course scope and focuses on explainable, course-aligned implementations.
