import mlflow as ml_flow
import mlflow.sklearn as m
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


def log_model_metrics(model, X_test, y_test, experiment_name="Defect severity classification"):
    # Setting up MLFlow Pipeline
    ml_flow.set_experiment(experiment_name)
    ml_flow.set_tracking_uri("file:./mlruns")
    with ml_flow.start_run():
        y_pred = model.predict(X_test)

        # Calculating scores
        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
        recall = recall_score(y_true=y_test, y_pred=y_pred, average="macro")
        F1_score = f1_score(y_true=y_test, y_pred=y_pred, average="macro")

        # Logging with the ML Flow pipeline
        ml_flow.log_metric("Accuracy: ", accuracy)
        ml_flow.log_metric("Recall: ", recall)
        ml_flow.log_metric("F1 Score: ", F1_score)
        m.log_model(model, "logistic_regression_model")

        print("ML Flow running successfully!")

