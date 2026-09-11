import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set your tracking server URI (or DagsHub URI)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name
mlflow.set_experiment("wine-autolog-experiment")

# Enable automatic logging for scikit-learn
mlflow.autolog()

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

max_depth = 9
n_estimators = 15

# Start Run
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    # mlflow.autolog() automatically logs params, training metrics, and the model during .fit()!
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy_s = accuracy_score(y_test, y_pred)

    # Optional custom plot artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion-matrix.png")
    plt.close()

    mlflow.log_artifact("confusion-matrix.png")

    print(f"Test Accuracy: {accuracy_s}")