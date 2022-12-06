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

from docker.types import Mount
import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable


default_args = {
    "owner": "Batyrkhan",
    # "depends_on_past": False,
    "start_date": datetime(2022, 12, 1),
    # "email_on_failure": False,
    # "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


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
    X_train = pd.read_csv(f'{input_dir}/train_features.csv').iloc[:, 1:]
    y_train = pd.read_csv(f'{input_dir}/train_target.csv').iloc[:, -1]

    model = LogisticRegression(max_iter=500, penalty="l2")
    model.fit(X_train, y_train)
    with open(model_dir + '/' + 'model.pkl', "wb") as f:
        pickle.dump(model, f)

def _model_validation(input_dir: str, model_dir: str, predictions_dir: str):
    X_val = pd.read_csv(f'{input_dir}/val_features.csv').iloc[:, 1:]
    with open(model_dir + '/' + 'model.pkl', "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_val)
    report_df = pd.DataFrame(y_pred)
    report_df.to_csv(f'{predictions_dir}/predictions.csv')


with DAG(
    dag_id="13_daily_ml",
    default_args=default_args,
    schedule_interval="0 12 * * *",
) as dag:

    my_var = Variable.get("my_var")

    new_folder_model = BashOperator(
        task_id="folder_creator_model_daily",
        bash_command='mkdir -p /opt/airflow/data/model_daily/{{ ds }}',
    )

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

    # preprocess = DockerOperator(
    #     image="airflow-preprocess",
    #     command="--input-dir /opt/airflow/data/raw/{{ ds }} --output-dir /opt/airflow/data/preprocess/{{ ds }}",
    #     task_id="docker-airflow-preprocess",
    #     do_xcom_push=False,
    #     mount_tmp_dir=False,
    #     mounts=[Mount(source="/home/bgainitdinov/MADE_lectures/MLops/airflow-examples/data/", target="/data", type='bind')]
    # )

    train = PythonOperator(
        task_id="model_train",
        python_callable=_model_training,
        op_kwargs={
            'input_dir': "/opt/airflow/data/preprocess/{{ ds }}", 
            'model_dir': f'{my_var}',
        }
    )

    # predict = DockerOperator(
    #     image="airflow-predict",
    #     # my_var=Variable.get("my_variables", deserialize_json=True),
    #     # my_var=Variable.get("my_var"),
    #     # my_var='/opt/airflow/data/model_daily/{{ ds }}',
    #     command=f"--input-dir /opt/airflow/data/preprocess/{{ ds }} --model-dir {my_var}",
    #     task_id="docker-airflow-predict",
    #     do_xcom_push=False,
    #     mount_tmp_dir=False,
    #     mounts=[Mount(source="/home/bgainitdinov/MADE_lectures/MLops/airflow-examples/data/", target="/data", type='bind')]
    # )

    new_folder_predictions = BashOperator(
        task_id="folder_creator_predictions",
        bash_command='mkdir -p /opt/airflow/data/predictions/{{ ds }}',
    )

    validation = PythonOperator(
        task_id="model_validation",
        python_callable=_model_validation,
        op_kwargs={
            'input_dir': "/opt/airflow/data/preprocess/{{ ds }}", 
            'model_dir': f'{my_var}',
            'predictions_dir': "/opt/airflow/data/predictions/{{ ds }}"
        }
    )

    # predictions = DockerOperator(
    #     image="airflow-predictions",
    #     # my_var=Variable.get("my_variables", deserialize_json=True),
    #     # my_var=Variable.get("my_var"),
    #     # my_var='/opt/airflow/data/model_daily/{{ ds }}',
    #     command=f"--input-dir /opt/airflow/data/preprocess/{{ ds }} --model-dir {my_var} --predictions-dir /opt/airflow/data/predictions/{{ ds }}",
    #     task_id="docker-airflow-predictions",
    #     do_xcom_push=False,
    #     mount_tmp_dir=False,
    #     mounts=[Mount(source="/home/bgainitdinov/MADE_lectures/MLops/airflow-examples/data/", target="/data", type='bind')]
    # )

    new_folder_model >> new_folder_preprocess >> preprocess >> train >> new_folder_predictions >> validation
