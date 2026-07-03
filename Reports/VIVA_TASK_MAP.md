# 📋 VIVA TASK MAP — Where Is Everything?

> **Use this file to quickly find which task is done where.**
> ML instructor and AI instructor ask about different things.
> This guide tells you exactly which file to open for each question.

---

## 🔄 PROJECT WORKFLOW (How Everything Connects)

```
Dataset (xAPI-Edu-Data.csv)
    │
    ▼
┌──────────────────────┐
│  ml_pipeline.py      │  ← ML Part (Run First)
│  Load → Preprocess   │
│  → Train → Evaluate  │
│  → Cluster → Save    │
└──────────┬───────────┘
           │  Outputs: prediction_output.json, model_results.csv
           ▼
┌──────────────────────┐
│  search_agent.py     │  ← AI Part 1 (Run Second)
│  Risk Mapping → BFS  │
│  → Action Path       │
└──────────┬───────────┘
           │  Outputs: search_path.json
           ▼
┌──────────────────────┐
│  csp_solver.py       │  ← AI Part 2 (Run Third)
│  CSP Variables →     │
│  Backtracking →      │
│  Feasible Plan       │
└──────────┬───────────┘
           │  Outputs: csp_solution.json
           ▼
┌──────────────────────┐
│  knowledge_base.py   │  ← AI Part 3 (Run Fourth)
│  Facts → Rules →     │
│  Forward Chaining →  │
│  Recommendation      │
└──────────┬───────────┘
           │  Outputs: knowledge_inference.json
           ▼
┌──────────────────────┐
│  main.py             │  ← Integration (Run Last)
│  Reads all outputs   │
│  → Final Summary     │
└──────────────────────┘
           │  Outputs: final_system_summary.txt
```

---

## 📂 HOW TO RUN (In Order)

```powershell
# Step 1: ML Pipeline
.venv\Scripts\python.exe Src\ml_pipeline.py

# Step 2: Search Agent
.venv\Scripts\python.exe Src\search_agent.py

# Step 3: CSP Solver
.venv\Scripts\python.exe Src\csp_solver.py

# Step 4: Knowledge Base
.venv\Scripts\python.exe Src\knowledge_base.py

# Step 5: Final Integration
.venv\Scripts\python.exe Src\main.py
```

---

## 🤖 ML VIVA — Tasks & File Locations

> **Your ML instructor will ask about these topics.**
> All ML code is in **ONE file**: `Src/ml_pipeline.py`

| # | ML Task | Where in Code | What to Say |
|---|---------|---------------|-------------|
| 1 | **Dataset Loading** | `ml_pipeline.py` → `load_data()` (Step 2) | "We load the CSV using `pd.read_csv()` from Pandas" |
| 2 | **Dataset Inspection** | `ml_pipeline.py` → `print_dataset_summary()` (Step 3) | "We check shape, columns, missing values, and class distribution" |
| 3 | **Missing Values** | `ml_pipeline.py` → `build_preprocessor()` (Step 5) | "We use `SimpleImputer` — median for numeric, most_frequent for categorical" |
| 4 | **Categorical Encoding** | `ml_pipeline.py` → `build_preprocessor()` (Step 5) | "We use `OneHotEncoder` to convert text categories into 0/1 columns" |
| 5 | **Numerical Scaling** | `ml_pipeline.py` → `build_preprocessor()` (Step 5) | "We use `StandardScaler` to normalize numeric features (mean=0, std=1)" |
| 6 | **Train/Test Split** | `ml_pipeline.py` → `preprocess_data()` (Step 5) | "80% train, 20% test, stratified to keep class proportions balanced" |
| 7 | **EDA (Charts)** | `ml_pipeline.py` → `perform_eda()` (Step 4) | "We plot class distribution, boxplots, and correlation heatmap" |
| 8 | **Logistic Regression** | `ml_pipeline.py` → `build_models()` (Step 6) | "Baseline linear model, uses One-vs-Rest for multiclass, simple and interpretable" |
| 9 | **Decision Tree** | `ml_pipeline.py` → `build_models()` (Step 6) | "Intermediate model, creates if-else rules, easy to visualize but can overfit" |
| 10 | **Random Forest** | `ml_pipeline.py` → `build_models()` (Step 6) | "Advanced ensemble of 300 trees, reduces variance compared to single tree" |
| 11 | **Model Evaluation** | `ml_pipeline.py` → `evaluate_model()` (Step 7) | "We calculate accuracy, precision, recall, F1, specificity, ROC-AUC" |
| 12 | **Confusion Matrix** | `ml_pipeline.py` → `_plot_confusion_matrix()` (Step 7) | "Shows actual vs predicted classes in a grid, helps identify misclassifications" |
| 13 | **ROC Curve** | `ml_pipeline.py` → `_plot_roc_curve()` (Step 7) | "Plots TPR vs FPR for each class. AUC closer to 1.0 = better model" |
| 14 | **Best Model Selection** | `ml_pipeline.py` → `train_and_evaluate_models()` (Step 7) | "We pick the model with highest F1-score and save it using joblib" |
| 15 | **K-Means Clustering** | `ml_pipeline.py` → `run_kmeans()` (Step 8) | "Unsupervised learning: groups students into 3 engagement clusters" |
| 16 | **Bias-Variance** | `ml_pipeline.py` → comments in Step 6 | "LR = high bias, DT = high variance, RF = reduces variance by averaging trees" |

### ML Output Files

| Output File | What It Contains |
|-------------|-----------------|
| `Output/model_results.csv` | Comparison table of all 3 models with metrics |
| `Output/prediction_output.json` | Sample prediction with class and probabilities |
| `Output/best_model.pkl` | Saved best model (loaded by frontend) |
| `Output/preprocessor.pkl` | Saved preprocessor (loaded by frontend) |
| `Output/Figures/*.png` | All EDA and evaluation charts |

---

## 🧠 AI VIVA — Tasks & File Locations

> **Your AI instructor will ask about these topics.**
> AI code is in **THREE files**: `search_agent.py`, `csp_solver.py`, `knowledge_base.py`

### AI Part 1: Search Agent (`Src/search_agent.py`)

| # | AI Task | Where in Code | What to Say |
|---|---------|---------------|-------------|
| 1 | **Intelligent Agent** | `search_agent.py` → entire file | "Our agent reads ML prediction and recommends actions to reach Academic Safe Zone" |
| 2 | **PEAS Framework** | `search_agent.py` → Step 2 (comments) | "P=Improve performance, E=Academic system, A=Actions like tutoring, S=ML prediction & features" |
| 3 | **Risk Mapping** | `search_agent.py` → `RISK_MAP` + `map_predicted_to_risk()` (Step 3) | "H→Low Risk, M→Medium Risk, L→High Risk (reverse mapping)" |
| 4 | **State-Space Problem** | `search_agent.py` → `build_graph()` (Step 5) | "Initial state=risk level, Goal=Academic Safe Zone, Actions=edges in graph" |
| 5 | **BFS Algorithm** | `search_agent.py` → `bfs()` (Step 6) | "Uses FIFO queue, explores level-by-level, finds shortest path, equal-cost actions" |
| 6 | **Why BFS not DFS?** | `search_agent.py` → comments in Step 6 | "BFS guarantees shortest path with equal-cost actions. DFS might find a longer path." |

### AI Part 2: CSP Solver (`Src/csp_solver.py`)

| # | AI Task | Where in Code | What to Say |
|---|---------|---------------|-------------|
| 7 | **CSP Definition** | `csp_solver.py` → Steps 2-4 | "Variables=schedule items, Domains=possible values, Constraints=minimums & limits" |
| 8 | **CSP Variables** | `csp_solver.py` → `VARIABLES` (Step 2) | "study_hours, resource_sessions, tutoring_days, participation, monitoring, rest" |
| 9 | **CSP Domains** | `csp_solver.py` → `DOMAINS` (Step 3) | "Each variable has a small set of possible values, e.g., study_hours=[1,2,3,4,5]" |
| 10 | **CSP Constraints** | `csp_solver.py` → `RISK_CONSTRAINTS` + `satisfies_constraints()` (Steps 4,6) | "Risk minimums + total workload ≤ 8 + rest ≥ 7 hours" |
| 11 | **Backtracking** | `csp_solver.py` → `backtracking_search()` (Step 7) | "Try values one by one, check consistency, if violation → backtrack and try next value" |
| 12 | **Feasibility Check** | `csp_solver.py` → `path_is_feasible()` (Step 5) | "Checks if the search agent's path matches the expected recommended path" |

### AI Part 3: Knowledge Base (`Src/knowledge_base.py`)

| # | AI Task | Where in Code | What to Say |
|---|---------|---------------|-------------|
| 13 | **Knowledge Representation** | `knowledge_base.py` → `initial_facts_from_context()` (Step 3) | "Facts are stored as key-value pairs in a Python dictionary" |
| 14 | **IF-THEN Rules** | `knowledge_base.py` → `build_rules()` (Step 4) | "20 rules like: IF attendance > 7 days THEN attendance_problem. Each has name, condition, action" |
| 15 | **Forward Chaining** | `knowledge_base.py` → `forward_chain()` (Step 5) | "Start with facts, fire matching rules, add new facts, repeat until nothing new fires" |
| 16 | **Inference Trace** | `knowledge_base.py` → `forward_chain()` → `trace` (Step 5) | "Records which rules fired and why — makes the system explainable" |
| 17 | **Contrasting Cases** | `knowledge_base.py` → `main()` (Step 8) | "We test on 2 cases: actual prediction AND a simulated good student to show different results" |
| 18 | **Final Recommendation** | `knowledge_base.py` → `derive_final_recommendation()` (Step 6) | "Based on inferred facts, generate a human-readable recommendation" |

### AI Output Files

| Output File | What It Contains |
|-------------|-----------------|
| `Output/search_path.json` | BFS path from risk level to Academic Safe Zone |
| `Output/csp_solution.json` | Feasible schedule plan from backtracking |
| `Output/knowledge_inference.json` | Rules fired, inference trace, recommendation |

---

## 🔗 INTEGRATION (`Src/main.py`)

| Task | Where | What to Say |
|------|-------|-------------|
| **End-to-End Workflow** | `main.py` → `format_workflow()` | "Shows the complete pipeline from dataset to final recommendation" |
| **Combined Recommendation** | `main.py` → `build_final_combined_recommendation()` | "Merges search path + CSP plan + knowledge recommendation into one" |
| **Final Summary** | `main.py` → `main()` | "Reads all JSON outputs and prints/saves a consolidated summary" |

---

## ❓ QUICK VIVA CHEAT SHEET

### If ML instructor asks...

| Question | Answer | Show This File |
|----------|--------|----------------|
| "Where is preprocessing?" | `ml_pipeline.py`, `build_preprocessor()` function | Open Step 5 |
| "What models did you use?" | Logistic Regression, Decision Tree, Random Forest | Open Step 6 in `ml_pipeline.py` |
| "Which model is best?" | Check `Output/model_results.csv` — highest F1 score | Open the CSV file |
| "What is overfitting?" | Model fits noise, good on train but bad on test | Read comments in Step 6 |
| "What is K-Means?" | Unsupervised clustering, groups students by engagement | Open Step 8 in `ml_pipeline.py` |
| "Show me the confusion matrix" | `Output/Figures/confusion_matrix_*.png` | Open the image files |
| "What is ROC-AUC?" | Area under ROC curve, 1.0=perfect, 0.5=random | Open Step 7 in `ml_pipeline.py` |

### If AI instructor asks...

| Question | Answer | Show This File |
|----------|--------|----------------|
| "What search algorithm?" | BFS — Breadth-First Search | Open Step 6 in `search_agent.py` |
| "What is PEAS?" | Performance, Environment, Actuators, Sensors | Open Step 2 in `search_agent.py` |
| "What is CSP?" | Variables + Domains + Constraints solved by backtracking | Open `csp_solver.py` Steps 2-7 |
| "What is backtracking?" | Try values, check constraints, undo if violation | Open Step 7 in `csp_solver.py` |
| "What is forward chaining?" | Fire rules on facts until no new facts added | Open Step 5 in `knowledge_base.py` |
| "How many rules?" | 20 IF-THEN rules in the knowledge base | Open Step 4 in `knowledge_base.py` |
| "Why BFS not DFS?" | BFS = shortest path with equal costs, DFS may find longer path | Read comments in `search_agent.py` Step 6 |
| "Show the inference trace" | Open `Output/knowledge_inference.json` | Check "rules_fired" and "inference_trace" |

---

## 📁 COMPLETE FILE MAP

```
Student Performance AI ML Project/
│
├── Data/
│   └── xAPI-Edu-Data.csv           ← Dataset (480 students, 17 columns)
│
├── Src/
│   ├── ml_pipeline.py              ← ALL ML CODE (preprocessing, models, evaluation, clustering)
│   ├── search_agent.py             ← AI: BFS Search + PEAS + State-Space
│   ├── csp_solver.py               ← AI: CSP + Backtracking
│   ├── knowledge_base.py           ← AI: Forward Chaining + IF-THEN Rules
│   └── main.py                     ← Integration: combines all outputs
│
├── Output/
│   ├── model_results.csv           ← ML: Model comparison metrics
│   ├── prediction_output.json      ← ML: Sample prediction (used by AI)
│   ├── best_model.pkl              ← ML: Saved best model
│   ├── preprocessor.pkl            ← ML: Saved preprocessor
│   ├── search_path.json            ← AI: BFS action path
│   ├── csp_solution.json           ← AI: Feasible academic plan
│   ├── knowledge_inference.json    ← AI: Inference trace + recommendation
│   ├── final_system_summary.txt    ← Integration: final summary
│   └── Figures/                    ← ML: All charts and plots
│
├── Frontend/
│   └── app.py                      ← Streamlit web interface
│
├── Reports/
│   ├── VIVA_GUIDE.md               ← Viva Q&A (25+ questions)
│   └── COURSE_ALIGNMENT.md         ← Topic → file mapping
│
└── requirements.txt                ← Python dependencies
```

---

> **💡 TIP**: Before your viva, open each file and scroll to the STEP number mentioned in the tables above. Each step is clearly marked with a big comment header so you can find it instantly.
