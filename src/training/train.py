import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.ml_flow.ml_flow_utils import log_model_metrics


def train():
    # load processed data from csv files
    train_data = pd.read_csv("data/processed/train.csv")
    test_data = pd.read_csv("data/processed/test.csv")

    # Separate features and labels
    X_train = train_data.drop("target", axis=1)
    X_test = test_data.drop("target", axis=1)
    y_train = train_data["target"]
    y_test = test_data["target"]
    print("Loaded the data with dimensions: ", X_train.shape, X_test.shape)
    
    # Train the model
    model = LogisticRegression(multi_class="multinomial", max_iter=500)
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Save the model
    file_path = "data/model/model.pkl"
    with open(file_path, "wb") as m:
        pickle.dump(model, m)
    print(f"Model saved to file path: {file_path}")

    # Log with MLFlow
    log_model_metrics(model, X_test, y_test)


# Calling the function
train()