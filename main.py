from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load the educational dataset and return features, target and names."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return dataset, X, y


def explore_data(dataset, X, y):
    """Print basic information and save a small descriptive statistics file."""
    print("\n=== DATASET OVERVIEW ===")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target classes: {list(dataset.target_names)}")
    print(f"Missing feature values: {int(X.isnull().sum().sum())}")

    print("\n=== FIRST 5 ROWS ===")
    print(X.head())

    print("\n=== CLASS COUNTS ===")
    # sklearn encodes the target as 0 and 1. Names are used in reports/plots.
    print(y.map({i: name for i, name in enumerate(dataset.target_names)}).value_counts())

    summary = X.describe().T
    summary.to_csv(OUTPUT_DIR / "feature_summary.csv")
    print(f"\nSaved feature summary to: {OUTPUT_DIR / 'feature_summary.csv'}")


def create_visualizations(X, y, target_names):
    """Create and save two exploratory charts."""
    # Chart 1: how many samples are in each target class?
    class_counts = y.map({i: name for i, name in enumerate(target_names)}).value_counts()
    ax = class_counts.plot(kind="bar", rot=0)
    ax.set_title("Number of samples per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close()

    # Chart 2: relationship between mean radius and mean texture, colored by class.
    plot_data = X[["mean radius", "mean texture"]].copy()
    plot_data["class"] = y.map({i: name for i, name in enumerate(target_names)})
    for class_name in target_names:
        subset = plot_data[plot_data["class"] == class_name]
        plt.scatter(
            subset["mean radius"],
            subset["mean texture"],
            label=class_name,
            alpha=0.7,
        )
    plt.title("Mean radius vs mean texture")
    plt.xlabel("Mean radius")
    plt.ylabel("Mean texture")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "radius_vs_texture.png", dpi=200)
    plt.close()
    print("Saved charts in the outputs/ folder.")


def train_and_evaluate(dataset, X, y):
    """Train two models, compare accuracy, and save the best model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append({"model": name, "accuracy": accuracy})
        trained_models[name] = model
        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy:.2%}")
        print(classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
            zero_division=0,
        ))

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    best_predictions = best_model.predict(X_test)

    # Save confusion matrix for the selected model.
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        display_labels=dataset.target_names,
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Save model and feature names for future use.
    joblib.dump(
        {"model": best_model, "feature_names": list(X.columns),
         "target_names": list(dataset.target_names), "model_name": best_name},
        OUTPUT_DIR / "best_model.joblib",
    )

    print(f"\nBest model by test accuracy: {best_name}")
    print("Saved model to outputs/best_model.joblib")
    print("Note: accuracy alone is not enough to assess a medical model.")
    return results_df


def main():
    dataset, X, y = load_data()
    explore_data(dataset, X, y)
    create_visualizations(X, y, dataset.target_names)
    train_and_evaluate(dataset, X, y)
    print("\nProject finished. Check the outputs/ folder.")


if __name__ == "__main__":
    main()


