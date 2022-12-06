from faker import Faker
import pandas as pd
import numpy as np
import random
import yaml
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pickle

import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def _data_preprocess(input_dir: str, output_dir: str):
    X = pd.read_csv(f'{input_dir}/data.csv').iloc[:,1:]
    y = pd.read_csv(f'{input_dir}/target.csv').iloc[:,-1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=False, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    pd.DataFrame(X_train_sc).to_csv(f'{output_dir}/train_features.csv')
    pd.DataFrame(X_val_sc).to_csv(f'{output_dir}/val_features.csv')
    pd.DataFrame(y_train).to_csv(f'{output_dir}/train_target.csv')
    pd.DataFrame(y_val).to_csv(f'{output_dir}/val_target.csv')


def _model_training(input_dir: str, model_dir: str):
    X_train = pd.read_csv(f'{input_dir}/train_features.csv').iloc[:,1:]
    y_train = pd.read_csv(f'{input_dir}/train_target.csv').iloc[:,-1]

    model = LogisticRegression(max_iter=500, penalty="l2")
    model.fit(X_train, y_train)
    with open(model_dir, "wb") as f:
        pickle.dump(model, f)


def _model_prediction(input_dir: str, model_dir: str, metrics_dir: str):
    X_val = pd.read_csv(f'{input_dir}/val_features.csv').iloc[:,1:]
    y_val = pd.read_csv(f'{input_dir}/val_target.csv').iloc[:,-1]

    with open(model_dir, "rb") as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_val)
    # f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{metrics_dir}/classification_report.csv')
    

default_args = {
    "owner": "Batyrkhan",
    # "depends_on_past": False,
    "start_date": datetime(2022, 12, 1),
    # "email_on_failure": False,
    # "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="13_weekly_ml",
    default_args=default_args,
    schedule_interval="0 14 * * 2",
) as dag:

    new_folder_preprocess = BashOperator(
        task_id="folder_creator_preprocess",
        bash_command='mkdir -p /opt/airflow/data/preprocess/{{ ds }}',
    )
    
    preprocess = PythonOperator(
        task_id="data_preprocess",
        python_callable=_data_preprocess,
        op_kwargs={
            "input_dir": "/opt/airflow/data/raw/{{ ds }}",
            "output_dir": "/opt/airflow/data/preprocess/{{ ds }}",
            }
        )

    new_folder_models = BashOperator(
        task_id="folder_creator_models",
        bash_command='mkdir -p /opt/airflow/data/models/{{ ds }}',
    )

    train = PythonOperator(
        task_id="model_train",
        python_callable=_model_training,
        op_kwargs={
            'input_dir': "/opt/airflow/data/preprocess/{{ ds }}", 
            'model_dir': '/opt/airflow/data/models/{{ ds }}/model.pkl',
        }
    )

    new_folder_reports = BashOperator(
        task_id="folder_creator_report",
        bash_command='mkdir -p /opt/airflow/data/reports/{{ ds }}',
    )

    predict = PythonOperator(
        task_id="model_predict",
        python_callable=_model_prediction,
        op_kwargs={
            'input_dir': "/opt/airflow/data/preprocess/{{ ds }}", 
            'model_dir': '/opt/airflow/data/models/{{ ds }}/model.pkl', 
            'metrics_dir': '/opt/airflow/data/reports/{{ ds }}'
        }
    )

    new_folder_preprocess >> preprocess >> new_folder_models >> train >> new_folder_reports >> predict
