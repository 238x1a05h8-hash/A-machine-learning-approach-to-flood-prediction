"""
Rising Waters – ML-Based Flood Prediction System
Model Training Script
Generates synthetic dataset, trains multiple models, saves best model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed — skipping XGBoost model.")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Generate / Load Dataset
# ─────────────────────────────────────────────

def generate_dataset(n=2000, seed=42):
    """Generate synthetic flood prediction dataset."""
    rng = np.random.RandomState(seed)

    annual_rainfall    = rng.normal(1200, 300, n)          # mm
    seasonal_rainfall  = rng.normal(800,  200, n)          # mm (Jun–Sep)
    temperature        = rng.normal(28,   5,   n)          # °C
    cloud_cover        = rng.uniform(0,   100, n)          # %
    humidity           = rng.uniform(40,  100, n)          # %

    # Flood probability driven by features
    flood_score = (
        0.35 * (annual_rainfall  - 1200) / 300 +
        0.30 * (seasonal_rainfall - 800) / 200 +
        0.15 * (cloud_cover - 50)        / 50  +
        0.12 * (humidity - 70)           / 30  -
        0.08 * (temperature - 28)        / 5
    )
    flood_prob = 1 / (1 + np.exp(-flood_score * 2))
    flood      = (rng.rand(n) < flood_prob).astype(int)

    df = pd.DataFrame({
        "Annual_Rainfall":   annual_rainfall,
        "Seasonal_Rainfall": seasonal_rainfall,
        "Temperature":       temperature,
        "Cloud_Cover":       cloud_cover,
        "Humidity":          humidity,
        "Flood":             flood,
    })
    df.to_csv("flood_data.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────────

def run_eda(df):
    """Exploratory Data Analysis."""
    print("\n─── EDA ───────────────────────────────────────")
    print("Shape :", df.shape)
    print("\nInfo:")
    df.info()
    print("\nNull values:\n", df.isnull().sum())
    print("\nTarget distribution:\n", df["Flood"].value_counts())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    features = ["Annual_Rainfall", "Seasonal_Rainfall", "Temperature", "Cloud_Cover", "Humidity"]
    for ax, feat in zip(axes.flat, features):
        df[feat].hist(ax=ax, bins=30, color="#1a6bff", edgecolor="white", alpha=0.8)
        ax.set_title(feat)
        ax.set_xlabel("")
    axes.flat[-1].set_visible(False)
    plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("static/eda_distributions.png", dpi=100)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("static/eda_heatmap.png", dpi=100)
    plt.close()
    print("EDA plots saved.")


# ─────────────────────────────────────────────
# STEP 3: Preprocessing
# ─────────────────────────────────────────────

def preprocess(df):
    """Handle missing values, outliers, scaling."""
    df = df.dropna()

    # IQR outlier removal
    features = ["Annual_Rainfall", "Seasonal_Rainfall", "Temperature", "Cloud_Cover", "Humidity"]
    for col in features:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    X = df[features]
    y = df["Flood"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    joblib.dump(scaler, "transform.save")
    print(f"\nPreprocessing done. Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test


# ─────────────────────────────────────────────
# STEP 4: Model Training & Comparison
# ─────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    """Train and compare models; save best."""
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN":           KNeighborsClassifier(n_neighbors=7),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                           random_state=42, verbosity=0)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds   = model.predict(X_test)
        acc     = accuracy_score(y_test, preds)
        report  = classification_report(y_test, preds, output_dict=True)
        f1      = report["weighted avg"]["f1-score"]
        results[name] = {"model": model, "accuracy": acc, "f1": f1}
        print(f"\n{'─'*40}")
        print(f"Model : {name}")
        print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))

    # Select best by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    best      = results[best_name]
    joblib.dump(best["model"], "floods.save")

    print(f"\n✅ Best model: {best_name} (Acc={best['accuracy']:.4f}, F1={best['f1']:.4f})")
    print("Model saved as floods.save")

    # Comparison bar chart
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    f1s   = [results[n]["f1"]       for n in names]
    x     = np.arange(len(names))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#1a6bff", alpha=0.85)
    ax.bar(x + w/2, f1s,  w, label="F1 Score",  color="#00c8ff", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_facecolor("#f8faff")
    fig.patch.set_facecolor("#f8faff")
    plt.tight_layout()
    plt.savefig("static/model_comparison.png", dpi=100)
    plt.close()
    print("Comparison chart saved.")
    return best_name, best["accuracy"], best["f1"]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset()
    run_eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    best_name, best_acc, best_f1 = train_models(X_train, X_test, y_train, y_test)
    print(f"\n🎯 Pipeline complete. Best: {best_name} | Acc: {best_acc:.2%} | F1: {best_f1:.2%}")
