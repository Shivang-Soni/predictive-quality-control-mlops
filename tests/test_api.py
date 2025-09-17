import requests

# Local API URL
url = "http://127.0.0.1:8000/predict/"

# Sample Data
data = {
    "defect_id": 1,
    "product_id": 101,
    "defect_type": "scratch",
    "defect_date": "2025-09-16",
    "defect_location": "front",
    "severity": "minor",
    "inspection_method": "visual",
    "repair_cost": 12.5
}

# POST Request
response = requests.post(url, json=data)

# Prüfen, ob alles korrekt läuft
assert response.status_code == 200, f"Status Code: {response.status_code}"
assert "Predicted severity" in response.json(), "Key 'Predicted severity' fehlt"

print(" Integration Test Passed")
print("Response:", response.json())
