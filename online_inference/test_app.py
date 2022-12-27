from fastapi.testclient import TestClient
import pandas as pd
import pickle
import json
import pytest
from app import app, load_model


client = TestClient(app)
# pkl_filename = "../ml_project/models/saved_models/model.pkl"
# with open(pkl_filename, 'rb') as file:
#     lr_model = pickle.load(file)
pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as file:
    lr_model = pickle.load(file)

@pytest.fixture(scope='session', autouse=True)
def start():
    load_model()

def test_predict():
    request = {
        'age': 50,
        'sex': 0,
        'cp': 3,
        'trestbps': 100,
        'chol': 100,
        'fbs': 1,
        'restecg': 0,
        'thalach': 80,
        'exang': 0,
        'oldpeak': 5,
        'slope': 0,
        'ca': 0,
        'thal': 1
        }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.json() == {'message': 'disease'}
    assert response.status_code == 200

def test_predict_nodecease():
    request = {
        "age": 40,
        "sex": 0,
        "cp": 2,
        "trestbps": 100,
        "chol": 90,
        "fbs": 1,
        "restecg": 0,
        "thalach": 100,
        "exang": 0,
        "oldpeak": 3,
        "slope": 1,
        "ca": 1,
        "thal": 1
        }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.json() == {'message': 'no disease'}
    assert response.status_code == 200

def test_predict_wrong_age():
    request = {
        "age": 120,
        "sex": 0,
        "cp": 2,
        "trestbps": 100,
        "chol": 90,
        "fbs": 1,
        "restecg": 0,
        "thalach": 100,
        "exang": 0,
        "oldpeak": 3,
        "slope": 1,
        "ca": 1,
        "thal": 1
        }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.json() == {"detail": "age is not correct"}
    assert response.status_code == 404
