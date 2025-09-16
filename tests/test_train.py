import os
import pandas as pd

def test_model_exists():
    assert os.path.exists("data/model/model.pkl"), "No model is in the directory!"

def test_training_data_exists():
    assert os.path.exists("data/processed/train.csv"), "Training data file is not available!"
    df = pd.read_csv("data/processed/train.csv")
    assert not df.empty, "Training data is missing!"
    assert "target" in df.columns, "Target column not found!"