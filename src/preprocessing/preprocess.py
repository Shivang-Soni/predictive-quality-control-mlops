import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess(file_path="data/raw/defects_data.csv"):
    # Load file.
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    #Severity has three possible values: to map it to 0,1,2
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["severity"])
    # Extract columns.
    columns = df.select_dtypes(include='object').columns
    # Store one hot coded variables.
    one_hot_encoded = pd.get_dummies(df, columns=columns)

    # Separate labels(target).
    X = one_hot_encoded.drop(labels="target", axis=1)
    Y = one_hot_encoded["target"]

    # Scale the values as per mean and standard deviation.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Store as a DataFrame
    preprocessed_data = pd.DataFrame(X_scaled, columns=X.columns)
    preprocessed_data["target"] = Y

    # Split the dataset in train and test datasets and save the files
    train_df, test_df = train_test_split(preprocessed_data, test_size=0.2, random_state=42)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    # Save the scaler and label encoder object as pickle(binary) files
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("data/processed/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Confirm whether proprocessing has been successful
    print("Datenverarbeitung ist abgeschlossen.")


preprocess()