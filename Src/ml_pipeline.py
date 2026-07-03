"""
ML PIPELINE — Machine Learning Component  (Src/ml_pipeline.py)
===============================================================
Runs the complete ML workflow for the Student Performance project:
  Step 2 — Data preprocessing  (missing values, encoding, scaling, split)
  Step 3 — Train 3 models      (Logistic Reg, Decision Tree, Random Forest)
  Step 4 — Bias-variance & regularization discussion (see comments)
  Step 5 — Evaluate models      (accuracy, precision, recall, specificity,
                                  F1, ROC-AUC, confusion matrix)
  Step 6 — Unsupervised learning (K-Means clustering)

Usage:  python Src/ml_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

# ─────────────────────────────────────────────────────────────────
#  PROJECT PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "xAPI-Edu-Data.csv"
OUTPUT_DIR = PROJECT_ROOT / "Output"
FIGURES_DIR = OUTPUT_DIR / "Figures"
MODEL_RESULTS_PATH = OUTPUT_DIR / "model_results.csv"
PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "prediction_output.json"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pkl"

TARGET_COLUMN = "Class"   # H (High), M (Medium), L (Low)
RANDOM_STATE = 42         # Fixed seed → reproducible results every run


def ensure_output_dirs() -> None:
    """Create Output/ and Output/Figures/ if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_figure(path: Path) -> None:
    """Save the current matplotlib figure to *path* and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# =================================================================
#  STEP 2 — DATA PREPROCESSING
# =================================================================
# VIVA: Preprocessing converts raw data into a form ML models can use.
#   • Handle missing values  → SimpleImputer
#   • Encode categorical     → OneHotEncoder (creates binary 0/1 columns)
#   • Scale numerical        → StandardScaler (mean=0, std=1 so features
#                               are on the same scale and distance-based
#                               algorithms work correctly)
#   • Train/test split       → stratify keeps class proportions equal in
#                               both sets so evaluation is fair
# =================================================================

def load_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset CSV into a Pandas DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def print_dataset_summary(df: pd.DataFrame, target: str = TARGET_COLUMN) -> None:
    """Print shape, columns, missing values, and class distribution."""
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isna().sum())
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    print("Target distribution:\n", df[target].value_counts())


def _one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder (handles sklearn version differences)."""
    # VIVA: OneHotEncoder turns each category into a separate binary column.
    #       handle_unknown='ignore' avoids errors if test data has unseen categories.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _split_columns(df: pd.DataFrame, target: str = TARGET_COLUMN) -> Tuple[List[str], List[str]]:
    """Separate feature columns into categorical and numerical lists."""
    features = [c for c in df.columns if c != target]
    numerical = [c for c in features if is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numerical]
    return categorical, numerical


def build_preprocessor(df: pd.DataFrame, target: str = TARGET_COLUMN) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies different pipelines to
    numerical vs categorical columns in one step.
    """
    categorical, numerical = _split_columns(df, target)

    # Numerical: fill missing with median → standardise
    # VIVA: Median is robust to outliers (unlike mean).
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),               # z = (x − μ) / σ
    ])

    # Categorical: fill missing with mode → one-hot encode
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0,      # Always return a dense array
    )


def preprocess_data(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, LabelEncoder]:
    """
    Encode target, split data 80/20, and transform features.

    Returns: X_train, X_test, y_train, y_test, preprocessor, label_encoder
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    X = df.drop(columns=[target])           # Feature matrix
    y_raw = df[target].astype(str)

    # VIVA: LabelEncoder maps class labels to integers (H→0, L→1, M→2).
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw), name=target)

    # VIVA: stratify=y ensures each split has the same H/M/L proportions
    #       as the full dataset, which is critical for imbalanced classes.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    preprocessor = build_preprocessor(df, target)
    X_train_processed = preprocessor.fit_transform(X_train)  # Learn stats from train
    X_test_processed = preprocessor.transform(X_test)        # Apply same stats (no fit!)
    # VIVA: We NEVER fit on test data — that would leak information from the
    #       test set into the model and produce over-optimistic results.

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, label_encoder


# =================================================================
#  STEP 3 — DEVELOP THREE ML MODELS
# =================================================================
#  1. Logistic Regression  (BASELINE)
#       – Linear model, simple, fast, interpretable coefficients.
#       – Uses softmax / one-vs-rest for multiclass.
#
#  2. Decision Tree         (INTERMEDIATE)
#       – Splits data with if/else rules; easy to visualise.
#       – Prone to overfitting without depth limits.
#
#  3. Random Forest          (ADVANCED — ENSEMBLE)
#       – Builds many decision trees on random subsets of data.
#       – Each tree votes; majority wins → reduces variance.
#       – "Ensemble" = combining multiple weak learners.
# =================================================================
#
#  STEP 4 — BIAS-VARIANCE TRADEOFF & REGULARISATION
# -----------------------------------------------------------------
#  VIVA KEY POINTS:
#    Bias  = error from oversimplifying → underfitting
#    Variance = error from over-sensitivity to training data → overfitting
#
#    Model              Bias        Variance    Note
#    ─────────────────  ──────────  ──────────  ──────────────────────
#    Logistic Reg       Higher      Low         May underfit non-linear data
#    Decision Tree      Low         High        Overfits training data easily
#    Random Forest      Low         Lower       Averaging trees reduces variance
#
#  Regularisation (max_iter in LR, max_depth / min_samples in trees)
#  controls model complexity to find the sweet spot between bias & variance.
# =================================================================

def build_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Instantiate the three classifiers we will compare."""
    return {
        # Baseline: simple linear boundary
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state,
        ),
        # Intermediate: learns decision rules, can overfit
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        # Advanced ensemble: 300 trees, majority voting
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state,
        ),
    }


# =================================================================
#  STEP 5 — MODEL EVALUATION
# =================================================================
#  VIVA DEFINITIONS (multiclass uses MACRO average = mean per class):
#    Accuracy    — fraction of all predictions that are correct
#    Precision   — of predicted positives, how many are truly positive?
#    Recall      — of actual positives, how many did we find? (Sensitivity)
#    Specificity — TN / (TN + FP); how well the model rejects negatives
#    F1-Score    — harmonic mean of precision & recall; balances both
#    ROC curve   — plots TPR vs FPR at different thresholds
#    AUC         — area under ROC curve; 1.0 = perfect, 0.5 = random
#    Confusion matrix — rows = actual, columns = predicted
# =================================================================

def _multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    """
    Macro-averaged specificity: TN / (TN + FP) per class, then averaged.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    specs: List[float] = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp          # Other classes predicted as i
        fn = cm[i, :].sum() - tp          # Class i predicted as other
        tn = total - tp - fp - fn         # Everything else
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return float(np.mean(specs))


def _plot_confusion_matrix(matrix: np.ndarray, labels: List[str], title: str, filename: str) -> None:
    """Save a heatmap of the confusion matrix."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save_figure(FIGURES_DIR / filename)


def _plot_roc_curve(
    y_test: np.ndarray, y_proba: np.ndarray,
    labels: List[str], model_name: str, filename: str,
) -> float:
    """
    Plot per-class ROC curves and return macro AUC.

    VIVA: ROC plots TPR vs FPR. A curve hugging the top-left corner
    means the model separates classes well.  AUC summarises this as a
    single number (higher = better).
    """
    y_bin = label_binarize(y_test, classes=np.arange(len(labels)))

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = float("nan")

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")

    plt.plot([0, 1], [0, 1], "--k", lw=1)   # Random-guess baseline
    plt.title(f"ROC Curves — {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    _save_figure(FIGURES_DIR / filename)
    return roc_auc


def evaluate_model(
    model_pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
    class_names: List[str], model_name: str,
) -> Dict[str, Any]:
    """Evaluate one model on the test set; return a metrics dict."""
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)
    labels_arr = np.arange(len(class_names))

    cm = confusion_matrix(y_test, y_pred, labels=labels_arr)

    # Generate visualisations (saved to Figures/)
    safe = model_name.lower().replace(" ", "_")
    roc_auc = _plot_roc_curve(y_test, y_proba, class_names, model_name,
                              f"roc_curve_{safe}.png")
    _plot_confusion_matrix(cm, class_names,
                           f"Confusion Matrix — {model_name}",
                           f"confusion_matrix_{safe}.png")

    return {
        "model":            model_name,
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_score":         float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "specificity":      float(_multiclass_specificity(y_test, y_pred, labels_arr)),
        "roc_auc":          float(roc_auc),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


def train_and_evaluate_models(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    class_names: List[str], random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, str, Pipeline]:
    """
    Train all 3 models, evaluate each, and pick the best by F1-score.

    Returns: results DataFrame, best model name, best model Pipeline.
    """
    results: List[Dict[str, Any]] = []
    store: Dict[str, Pipeline] = {}

    for name, estimator in build_models(random_state).items():
        pipe = Pipeline([("model", estimator)])
        pipe.fit(X_train, y_train)               # Train on training set only
        store[name] = pipe
        results.append(evaluate_model(pipe, X_test, y_test, class_names, name))

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODEL_RESULTS_PATH, index=False)

    # Best model = highest F1 (then accuracy as tiebreaker)
    best_row = results_df.sort_values(["f1_score", "accuracy"], ascending=False).iloc[0]
    best_name = str(best_row["model"])
    best_model = store[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)     # Persist for later use

    return results_df, best_name, best_model


# =================================================================
#  STEP 6 — UNSUPERVISED LEARNING (K-Means Clustering)
# =================================================================
#  VIVA: K-Means is UNSUPERVISED — it finds structure without labels.
#    1. Choose k centroids randomly
#    2. Assign each point to its nearest centroid (Euclidean distance)
#    3. Recompute centroids as mean of assigned points
#    4. Repeat 2-3 until centroids stabilise (convergence)
#
#  We cluster students by engagement features to discover patterns
#  (e.g., highly engaged vs disengaged groups).
#  Scaling is important because K-Means uses distance — features on
#  larger scales would dominate otherwise.
# =================================================================

def run_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """Run K-Means (k=3) on student engagement features and plot results."""
    features = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    available = [f for f in features if f in df.columns]
    if len(available) < 2:
        return pd.DataFrame()

    km_data = df[available].copy().fillna(df[available].median(numeric_only=True))
    scaled = StandardScaler().fit_transform(km_data)   # Scale before clustering!

    k = min(3, len(df)) if len(df) > 1 else 1
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_df = km_data.copy()
    cluster_df["cluster"] = kmeans.fit_predict(scaled)

    # Scatter plot: first two engagement features coloured by cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cluster_df, x=available[0], y=available[1],
                    hue="cluster", palette="deep")
    plt.title("K-Means Clustering on Engagement Features")
    _save_figure(FIGURES_DIR / "kmeans_clustering.png")

    return cluster_df


# =================================================================
#  EDA — Exploratory Data Analysis (supporting charts)
# =================================================================

def perform_eda(df: pd.DataFrame, target: str = TARGET_COLUMN) -> None:
    """Generate and save EDA charts to Output/Figures/."""
    ensure_output_dirs()

    # 1. Class distribution bar chart
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target, order=df[target].value_counts().index)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    _save_figure(FIGURES_DIR / "class_distribution.png")

    # 2-4. Feature vs Class plots
    for feat, fname, kind in [
        ("raisedhands",        "raisedhands_vs_class.png",        "box"),
        ("VisITedResources",   "visitedresources_vs_class.png",   "box"),
        ("StudentAbsenceDays", "studentabsencedays_vs_class.png", "count"),
    ]:
        if feat not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        if kind == "box":
            sns.boxplot(data=df, x=target, y=feat, order=df[target].value_counts().index)
        else:
            sns.countplot(data=df, x=feat, hue=target)
        plt.title(f"{feat} vs Class")
        _save_figure(FIGURES_DIR / fname)

    # 5. Correlation heatmap (numeric columns only)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        _save_figure(FIGURES_DIR / "correlation_heatmap.png")


# =================================================================
#  SAVE SAMPLE PREDICTION (for AI components)
# =================================================================

def save_sample_prediction(
    best_model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
    class_names: List[str], best_model_name: str,
) -> None:
    """Save one sample prediction with probabilities to JSON."""
    sample = X_test[:1]
    pred_label = class_names[int(best_model.predict(sample)[0])]
    actual_label = class_names[int(y_test[0])]
    probs = best_model.predict_proba(sample)[0]

    payload = {
        "model": best_model_name,
        "predicted_class": pred_label,
        "actual_class": actual_label,
        "probabilities": {class_names[i]: float(p) for i, p in enumerate(probs)},
    }
    with PREDICTION_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =================================================================
#  RUN COMPLETE ML PIPELINE
# =================================================================
# Order: Load → Inspect → EDA → Preprocess → Train → Evaluate → Cluster → Save

def run_pipeline() -> pd.DataFrame:
    """Execute the full ML pipeline from data loading to output saving."""
    ensure_output_dirs()

    # --- Data loading & inspection ---
    df = load_data()
    print_dataset_summary(df)

    # --- Exploratory Data Analysis ---
    perform_eda(df)

    # --- Step 2: Preprocessing ---
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess_data(df)
    class_names = label_encoder.classes_.tolist()  # ['H', 'L', 'M']

    # --- Steps 3-5: Train, evaluate, select best model ---
    results_df, best_name, best_model = train_and_evaluate_models(
        X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), class_names,
    )

    # Save preprocessor so the frontend can transform new inputs identically
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.pkl")

    # --- Step 6: Unsupervised learning ---
    run_kmeans(df)

    # --- Save sample prediction for AI components ---
    save_sample_prediction(best_model, X_test, y_test.to_numpy(), class_names, best_name)

    print(f"\nBest model: {best_name}")
    print(f"Model results saved to: {MODEL_RESULTS_PATH}")
    print(f"Prediction output saved to: {PREDICTION_OUTPUT_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    return results_df


if __name__ == "__main__":
    run_pipeline()
