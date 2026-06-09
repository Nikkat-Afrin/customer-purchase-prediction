"""
Customer Purchase Prediction — reproducible model comparison.

Enhancement layer over the notebook (which used logistic regression only).
Compares four classifiers with full metrics and writes:
    reports/model_comparison.md
    reports/figures/roc_curves.png
    reports/figures/confusion_matrix_best.png
    reports/figures/feature_importance_rf.png

Run from the repo root:   python src/model_comparison.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_and_prepare():
    df = pd.read_csv(ROOT / "data" / "Project4_Data.csv")
    df = df.drop(columns=[c for c in ["ID"] if c in df.columns])
    # Target: Y/N -> 1/0
    y = (df["TARGET"].astype(str).str.upper().str[0] == "Y").astype(int)
    X = df.drop(columns=["TARGET"])
    # One-hot any categorical columns (e.g. city, type_A/type_B if non-numeric)
    obj_cols = list(X.select_dtypes(include="object").columns)
    X = pd.get_dummies(X, columns=obj_cols, drop_first=True, dtype=int)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def evaluate(name, model, X_tr, X_te, y_tr, y_te, roc_store):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    roc_store[name] = roc_curve(y_te, proba)
    return {"Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "F1": f1_score(y_te, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te, proba)}, model


def main():
    X, y = load_and_prepare()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=RNG)
    scaler = StandardScaler().fit(X_tr)
    X_tr = pd.DataFrame(scaler.transform(X_tr), columns=X.columns, index=X_tr.index)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X.columns, index=X_te.index)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RNG),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1),
        "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
    }
    roc_store, rows, fitted = {}, [], {}
    for name, mdl in models.items():
        row, m = evaluate(name, mdl, X_tr, X_te, y_tr, y_te, roc_store)
        rows.append(row); fitted[name] = m

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print("\n=== Test-set model comparison ===")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    (ROOT / "reports").mkdir(exist_ok=True)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fmt = lambda v: v if isinstance(v, str) else f"{v:.3f}"
    lines = ["# Test-set model comparison", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for _, r in results.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = results.iloc[0]["Model"]

    # ROC overlay
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, _) in roc_store.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, fitted[name].predict_proba(X_te)[:,1]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Customer Purchase Prediction"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=120); plt.close()

    # Confusion matrix for best model
    pred = fitted[best].predict(X_te)
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred),
                           display_labels=["No buy", "Buy"]).plot(cmap="Greens", colorbar=False)
    plt.title(f"Confusion Matrix — {best} (test)")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix_best.png", dpi=120); plt.close()

    # Random Forest feature importance
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)
    plt.figure(figsize=(8, 6)); imp.plot.barh(color="#2e7d32")
    plt.title("Random Forest — Feature Importance"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance_rf.png", dpi=120); plt.close()

    print(f"\nBest by ROC-AUC: {best}\nFigures + table written to {ROOT/'reports'}")


if __name__ == "__main__":
    main()
