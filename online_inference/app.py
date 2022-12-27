import pandas as pd
import os
import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel
from schemas import Illness


app = FastAPI()

lr_model = None

@app.on_event("startup")
def load_model():
    # with open("../ml_project/models/saved_models/model.pkl", "rb") as f:
    #     global lr_model
    #     lr_model = pickle.load(f)
    with open("model.pkl", "rb") as f:
        global lr_model
        lr_model = pickle.load(f)

@app.get('/')
def root():
    return {'message': 'Hello friends!'}

@app.post('/predict')
async def make_smth(data: Illness):
    x_data = pd.DataFrame.from_records([data.dict()])
    predict = lr_model.predict(x_data)
    decease = 'no disease' if not predict[0] else 'disease'
    return {'message': decease}

@app.get('/health')
async def model_works():
    if lr_model:
        return {'200': 'model is ok'}
    return {'error': 'model is NOT ok'}
