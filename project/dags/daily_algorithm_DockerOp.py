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

with DAG(
    dag_id="13_daily_ml_dockerOp",
    default_args=default_args,
    schedule_interval="0 12 * * *",
) as dag:

    my_var = Variable.get("my_var")

    new_folder_model = BashOperator(
        task_id="folder_creator_model_daily",
        bash_command='mkdir -p /opt/airflow/data/model_daily/{{ ds }}',
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /opt/airflow/data/raw/{{ ds }} --output-dir /opt/airflow/data/preprocess/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/bgainitdinov/MADE_lectures/MLops/airflow-examples/data/", target="/data", type='bind')]
    )

    predict = DockerOperator(
        image="airflow-predict",
        # my_var=Variable.get("my_variables", deserialize_json=True),
        # my_var=Variable.get("my_var"),
        # my_var='/opt/airflow/data/model_daily/{{ ds }}',
        command=f"--input-dir /opt/airflow/data/preprocess/{{ ds }} --model-dir {my_var}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/bgainitdinov/MADE_lectures/MLops/airflow-examples/data/", target="/data", type='bind')]
    )

    new_folder_predictions = BashOperator(
        task_id="folder_creator_predictions",
        bash_command='mkdir -p /opt/airflow/data/predictions/{{ ds }}',
    )

    predictions = DockerOperator(
        image="airflow-predictions",
        # my_var=Variable.get("my_variables", deserialize_json=True),
        # my_var=Variable.get("my_var"),
        # my_var='/opt/airflow/data/model_daily/{{ ds }}',
        command=f"--input-dir /opt/airflow/data/preprocess/{{ ds }} --model-dir {my_var} --predictions-dir /opt/airflow/data/predictions/{{ ds }}",
        task_id="docker-airflow-predictions",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/bgainitdinov/MADE_lectures/MLops/airflow-examples/data/", target="/data", type='bind')]
    )

    new_folder_model >> preprocess >> predict >> new_folder_predictions >> predictions
