# Student Performance Prediction and Academic Advising System

## 1. Project Overview
This project is a combined Artificial Intelligence and Machine Learning semester project that predicts student performance and generates academic advising recommendations. The system starts from the raw student dataset and ends with an intelligent academic recommendation that supports intervention planning, risk reduction, and academic guidance.

The workflow integrates supervised learning, unsupervised learning, a search-based intelligent agent, a constraint satisfaction problem solver, and a rule-based knowledge system into a single end-to-end decision support pipeline.

## 2. Dataset
The dataset used in this project is:

`Data/xAPI-Edu-Data.csv`

Target column:

`Class`

Target classes:

- `H` = High performance
- `M` = Medium performance
- `L` = Low performance

Risk mapping used by the intelligent system:

- `H` = Low Risk
- `M` = Medium Risk
- `L` = High Risk

The dataset includes academic engagement and student-related features such as:

- `raisedhands`
- `VisITedResources`
- `AnnouncementsView`
- `Discussion`
- `StudentAbsenceDays`
- `ParentAnsweringSurvey`
- `ParentschoolSatisfaction`

## 3. Project Objectives
The main objectives of the project are to:

- Predict student performance class.
- Compare multiple machine learning models.
- Evaluate models using classification metrics.
- Apply unsupervised learning through clustering.
- Use ML prediction as input to AI decision-making.
- Generate academic advising recommendations.
- Validate recommendations using CSP.
- Use rule-based reasoning through a knowledge-based system.

## 4. Methodology
The complete workflow is:

Dataset -> Data preprocessing -> Exploratory Data Analysis -> Machine Learning model training -> Model evaluation -> Best model selection -> Risk mapping -> K-Means clustering -> BFS search agent -> CSP validation -> Forward chaining knowledge system -> Final recommendation

The system is designed so that the machine learning component produces the student performance prediction, and the AI components convert that prediction into a practical academic advising decision.

## 5. Machine Learning Component
The supervised learning stage includes three models:

- Logistic Regression as the baseline model
- Decision Tree as the intermediate model
- Random Forest as the advanced model

The best model is selected based on the evaluation results saved in `Output/model_results.csv`.

Preprocessing includes:

- Missing value check
- Categorical feature encoding
- Numerical feature scaling
- Train/test split

## 6. Evaluation Metrics
The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- Confusion matrix
- ROC curve
- AUC score

The evaluation results are saved in `Output/model_results.csv`, and the figures are saved under `Output/Figures/`.

## 7. Unsupervised Learning
K-Means clustering is applied using engagement-related numerical features:

- `raisedhands`
- `VisITedResources`
- `AnnouncementsView`
- `Discussion`

Clustering helps identify student engagement patterns and supports the academic advising interpretation layer.

## 8. AI Search Agent
The ML prediction is converted into a risk level, and the risk level becomes the initial state for the search agent. Breadth-First Search (BFS) is used to find an action path toward the Academic Safe Zone.

PEAS framework:

Performance:

Improve academic performance and reduce student risk.

Environment:

Academic advising system.

Actuators:

Recommend attendance improvement, resource usage, class participation, tutoring, weekly monitoring, and maintaining current plan.

Sensors:

ML prediction, absence days, raised hands, visited resources, announcements viewed, discussion activity, parent survey, and parent satisfaction.

## 9. Constraint Satisfaction Problem
The CSP validates or refines the search-agent recommendation. Backtracking search is used to generate a feasible academic support plan.

Variables:

- `study_hours`
- `resource_sessions`
- `tutoring_days`
- `participation_sessions`
- `monitoring_sessions`
- `rest_hours`

The constraints include workload limits, rest requirements, and risk-based academic support requirements. The CSP result is saved in `Output/csp_solution.json`.

## 10. Knowledge-Based System
The system uses IF-THEN rules and forward chaining to infer academic guidance from the prediction, the search path, and the CSP output. The system produces an inference trace, is tested on contrasting student cases, and the final reasoning supports or refines the recommendation.

## 11. Project Structure
```text
Student Performance AI ML Project/
│
├── Data/
│   └── xAPI-Edu-Data.csv
│
├── Src/
│   ├── ml_pipeline.py
│   ├── search_agent.py
│   ├── csp_solver.py
│   ├── knowledge_base.py
│   └── main.py
│
├── Output/
│   ├── model_results.csv
│   ├── prediction_output.json
│   ├── search_path.json
│   ├── csp_solution.json
│   ├── knowledge_inference.json
│   ├── final_system_summary.txt
│   ├── best_model.pkl
│   └── Figures/
│
├── Reports/
│
├── Presentation/
│
├── requirements.txt
└── README.md
```

## 12. Important Files
Brief description of the main files in the project:

- `Src/ml_pipeline.py` - Loads the dataset, preprocesses it, trains the models, evaluates them, performs K-Means clustering, and saves the machine learning outputs.
- `Src/search_agent.py` - Converts the predicted class into a risk level and generates a BFS-based academic advising path.
- `Src/csp_solver.py` - Validates the search-agent recommendation with a constraint satisfaction model and produces a feasible academic plan.
- `Src/knowledge_base.py` - Applies forward chaining over domain rules to produce reasoning-based recommendations and inference traces.
- `Src/main.py` - Integrates the outputs from all stages and prints the final system summary.
- `Output/model_results.csv` - Model comparison table with evaluation metrics.
- `Output/prediction_output.json` - ML prediction output and predicted class.
- `Output/search_path.json` - BFS search path and risk mapping.
- `Output/csp_solution.json` - CSP variables, feasibility result, and refined plan.
- `Output/knowledge_inference.json` - Knowledge-based inference trace and conclusions.
- `Output/final_system_summary.txt` - Final integrated summary of the complete system.

## 13. How to Run the Project
The project is intended for Windows. Run the following commands from the project root.

Create virtual environment:

```powershell
python -m venv .venv
```

Activate virtual environment:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run ML pipeline:

```powershell
.venv\Scripts\python.exe Src\ml_pipeline.py
```

Run search agent:

```powershell
.venv\Scripts\python.exe Src\search_agent.py
```

Run CSP solver:

```powershell
.venv\Scripts\python.exe Src\csp_solver.py
```

Run knowledge-based system:

```powershell
.venv\Scripts\python.exe Src\knowledge_base.py
```

Run final integrated system:

```powershell
.venv\Scripts\python.exe Src\main.py
```

## 14. Generated Outputs
The project generates the following outputs:

- EDA figures in `Output/Figures/`
- Model comparison table in `Output/model_results.csv`
- Best trained model in `Output/best_model.pkl`
- Prediction output in `Output/prediction_output.json`
- BFS search path in `Output/search_path.json`
- CSP solution in `Output/csp_solution.json`
- Knowledge inference trace in `Output/knowledge_inference.json`
- Final system summary in `Output/final_system_summary.txt`

## 15. Final Recommendation Output
The final system produces an academic advising recommendation based on:

- ML model prediction
- Risk level
- BFS search path
- CSP validated plan
- Knowledge-based inference

This allows the system to move from raw student data to an actionable and explainable academic advising decision.

## 16. Group Members
- Member 1:
- Member 2:
- Member 3:

## 17. Future Improvements
Possible future improvements include:

- Streamlit frontend
- Hyperparameter tuning
- Larger dataset
- Real-time student dashboard
- Additional advanced models
- Personalized intervention tracking

## 18. Note
This project is currently implemented as a Python-based intelligent system. A frontend is not required by the given project instructions, but it can be added later as a future improvement.
# Student-performance-detector-
