import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='kaif9910', repo_name='MLOPS-Experiments-with-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/kaif9910/MLOPS-Experiments-with-MLflow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 9
n_estimators = 15

# MENTION YOUR EXPERIMENTS BELOW

mlflow.set_experiment('MLOPS-EXP11')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy_s = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy_s", accuracy_s)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # creating confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.title('confusion matrix')

    # save plot

    plt.savefig("confusion-matrix.png")

    # log artifact using mlflow

    mlflow.log_artifact("confusion-matrix.png")
    mlflow.log_artifact(__file__)   # only works when run as a .py script

    # tags

    mlflow.set_tags({"Author": "vikash", "project": "wine classification"})

    # log the model

    mlflow.sklearn.log_model(rf, "random_forest_model")

    print(accuracy_s)