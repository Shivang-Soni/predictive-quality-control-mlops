import os

def test_processed_files_exist():
    files = [
        "data/processed/label_encoder.pkl",
        "data/processed/scaler.pkl",
        "data/processed/test.csv",
        "data/processed/train.csv"
    ]
    for f in files:
        assert os.path.exists(f), f"{f} not found!"

def test_target_column_exists():
    files = [
        "data/processed/train.csv",
        "data/processed/test.csv",
    ]
    for f in files:
        assert os.path.exists(f), f"target not found in {f}!"