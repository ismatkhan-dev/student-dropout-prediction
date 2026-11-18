
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project folder and dataset path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data.csv"

# Load dataset
data = pd.read_csv(DATA_PATH, sep=";")

# Separate features and target
X = data.drop("Target", axis=1)
y = data["Target"]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Train and evaluate models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted"
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted"
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted"
        )
    })

# Create comparison table
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.round(3))

# Select best model using F1 Score
best_name = results_df.loc[
    results_df["F1 Score"].idxmax(), "Model"
]

best_model = models[best_name]

# Save best model and feature columns
joblib.dump(
    {
        "model": best_model,
        "feature_columns": X.columns.tolist()
    },
    BASE_DIR / "student_dropout_model.pkl"
)

# Save comparison results
results_df.to_csv(
    BASE_DIR / "model_comparison_results.csv",
    index=False
)

print("\nBest Model:", best_name)
print("Model saved successfully!")
print("Comparison results saved successfully!")
