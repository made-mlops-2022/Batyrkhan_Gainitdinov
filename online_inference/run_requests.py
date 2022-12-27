import json
import logging

import pandas as pd

import requests

logger = logging.getLogger('request')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

data = pd.read_csv('../ml_project/data/interim/test_synthetic.csv').drop('condition', axis=1)
data_requests = data.to_dict(orient='records')

for request in data_requests:
    response = requests.post(
        'http://127.0.0.1:5001/predict',
        json.dumps(request)
    )
    logger.info(f'Message: {response.json()}')
