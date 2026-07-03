# VIVA Guide — Student Performance AI ML Project

1. Project One-Line Explanation
This project predicts student performance using machine learning and then uses AI decision-making methods to recommend academic support actions.

2. Why We Selected Student Performance Dataset
The dataset contains engagement, attendance, and parental/academic features useful to predict student performance (High/Medium/Low). It is suitable for classification tasks and for building simple AI agents and rule-based reasoning.

3. Dataset Columns Explanation
- `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`: engagement features (numeric).
- `StudentAbsenceDays`: attendance indicator.
- Parental features: `ParentAnsweringSurvey`, `ParentschoolSatisfaction`.
- `Class`: target column with values H/M/L.

4. Target Column Explanation
- `Class` is categorical with three labels: H (High), M (Medium), L (Low). We treat it as multiclass classification and map to risk levels.

5. Why This Is Classification
We predict discrete labels (H/M/L). Supervised learning techniques for classification are appropriate.

6. Preprocessing Explanation
- Missing values: checked and imputed (median for numeric, most frequent for categorical).
- Categorical encoding: OneHotEncoder for categorical features.
- Numerical scaling: StandardScaler on numeric features to help some models and distance-based methods.
- Train/test split: used to evaluate generalization on unseen data (stratified by class).

7. Why We Used Logistic Regression
- Baseline model.
- Simple linear classifier, interpretable coefficients.
- For multiclass, can use One-vs-Rest or multinomial approaches.

8. Why We Used Decision Tree
- Produces human-readable rules.
- Easy to interpret in viva.
- Can overfit if not controlled (discuss pruning/hyperparameters).

9. Why We Used Random Forest
- Ensemble of decision trees; reduces variance compared to single tree.
- Often delivers better performance while remaining explainable at a high level.

10. Evaluation Metrics Explanation
- Accuracy: overall fraction correctly predicted.
- Precision: of predicted positives, fraction truly positive (macro-average for multiclass).
- Recall: of actual positives, fraction correctly predicted.
- Specificity: true negative rate (averaged across classes for multiclass).
- F1-score: harmonic mean of precision and recall.
- ROC-AUC: area under ROC curve; for multiclass we use macro/OVR averaging.
- Confusion matrix: counts of true vs predicted classes.

11. Bias-Variance Explanation
- Underfitting: model too simple, high bias, performs poorly on train and test.
- Overfitting: model too complex, low bias but high variance, good train but poor test.
- Random Forest reduces variance by averaging many trees.
- Regularization (e.g., in logistic regression) controls complexity.

12. K-Means Explanation
- Unsupervised clustering on engagement features: `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`.
- Helps identify groups of students by participation/engagement.

13. AI Agent Explanation (PEAS)
- Performance: reduce academic risk and improve student performance.
- Environment: academic advising environment and available student signals.
- Actuators: recommended actions (attendance improvement, resource usage, tutoring, monitoring).
- Sensors: ML prediction, attendance, engagement features, parental responses.

14. BFS Explanation
- Breadth-First Search explores level-by-level using a queue.
- It finds the shortest sequence of actions (in number of steps) when actions have equal cost.
- We use BFS because actions are treated equal-cost and it is easy to explain.

15. CSP Explanation
- Variables: study_hours, resource_sessions, tutoring_days, participation_sessions, monitoring_sessions, rest_hours.
- Domains: small discrete sets (e.g., study_hours 1-5).
- Constraints: minimums per risk level, global workload limit (sum <= 8), rest_hours >=7.
- Backtracking: try values, backtrack on violation until a feasible plan is found.

16. Knowledge-Based System Explanation
- Facts: known values from ML, search-agent and CSP outputs.
- Rules: IF-THEN rules (12+) that derive new facts like `intervention_required`.
- Forward chaining: repeatedly apply rules to infer new facts until no more apply.
- Inference trace: a list of which rules fired and their justifications.

17. How All Components Are Connected
Dataset -> Preprocessing -> ML models -> Prediction (predicted_class) -> Risk mapping -> Search agent (BFS) -> Search path -> CSP validates/refines plan -> Knowledge base uses facts + rules to infer final recommendation -> main summarises outputs.

18. Common Viva Questions and Answers (25+)
1. Q: What is the target variable?
   A: `Class` with labels H/M/L.
2. Q: Why classification not regression?
   A: Target is categorical classes, not continuous values.
3. Q: Why OneHotEncoder?
   A: To convert categorical variables into numeric features for models.
4. Q: Why scale numerical data?
   A: Many algorithms (and distance-based methods) perform better when features are scaled.
5. Q: Why train/test split?
   A: To measure generalization on unseen data and avoid overfitting.
6. Q: Why Logistic Regression?
   A: Simple baseline; interpretable and fast.
7. Q: Why Decision Tree?
   A: Interpretable rules and easy to visualize.
8. Q: Why Random Forest?
   A: Ensemble to reduce variance and typically improve accuracy.
9. Q: What is overfitting?
   A: Model fits noise; high train accuracy but low test accuracy.
10. Q: What is underfitting?
    A: Model too simple to capture patterns; poor performance on both train and test.
11. Q: What is bias-variance tradeoff?
    A: Balancing model complexity to reduce both systematic error (bias) and sensitivity to data (variance).
12. Q: What is ROC-AUC?
    A: Area under ROC, measures discrimination ability; for multiclass we use macro averaging.
13. Q: What is specificity?
    A: True negative rate; for multiclass averaged across classes.
14. Q: What is K-Means used for here?
    A: Unsupervised grouping of students by engagement.
15. Q: What is PEAS?
    A: Performance, Environment, Actuators, Sensors — agent specification.
16. Q: What is BFS and why used?
    A: Breadth-First Search; used because actions are equal cost and it finds shortest step sequences.
17. Q: Why not DFS?
    A: DFS explores deep paths first and may return a long or non-minimal path; BFS is easier to justify for shortest-action plans.
18. Q: What is CSP?
    A: Constraint Satisfaction Problem; choose values satisfying all constraints.
19. Q: What is backtracking?
    A: Systematically try assignments and revert when constraints fail.
20. Q: What is forward chaining?
    A: Rule-driven inference that starts from facts and derives consequences.
21. Q: How does CSP refine the search-agent path?
    A: CSP tests whether a recommended path can be implemented with feasible resource/time allocations and adjusts plan if needed.
22. Q: What are limitations of the project?
    A: Simplified assumptions, static rules, limited dataset, and no online adaptation.
23. Q: What improvements could you add?
    A: More features, cross-validation, hyperparameter tuning, and user studies.
24. Q: What happens if prediction is High Risk?
    A: Search-agent proposes the High Risk path; CSP enforces stronger minimums; KB recommends urgent interventions.
25. Q: How to explain a model's decision in viva?
    A: For Logistic Regression show coefficients; for Decision Tree show split rules; for Random Forest show feature importances.


---

This file is intentionally concise and viva-oriented. Review and memorise the short answers.
