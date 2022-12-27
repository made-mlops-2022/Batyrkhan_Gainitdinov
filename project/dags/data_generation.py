from faker import Faker
import pandas as pd
import numpy as np
import random
import yaml
import os
from datetime import datetime, timedelta

import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def _data_gen(output_dir: str):
    def load_config(config_name):
        # with open(os.path.join(CONFIG_PATH, config_name)) as file:
        with open(config_name) as file:
            config = yaml.safe_load(file)
        return config
    
    CONFIG_PATH = ""
    config = load_config("my_config.yaml")
    f = Faker()
    df = pd.read_csv(config['data'])
    # df = pd.read_csv(os.path.join('../', config['raw_data'], config['data']))
    synth_df = np.zeros_like(df)
    for i in range(df.shape[0]):
        synth_df[i, 0] = f.random_int(29, 77)
        synth_df[i, 1] = f.random_int(0, 1)
        synth_df[i, 2] = f.random_int(0, 3)
        synth_df[i, 3] = f.random_int(94, 200)
        synth_df[i, 4] = random.randint(126, 564)
        synth_df[i, 5] = f.random_int(0, 1)
        synth_df[i, 6] = random.choice([0, 2])
        synth_df[i, 7] = f.random_int(71, 202)
        synth_df[i, 8] = f.random_int(0, 1)
        synth_df[i, 9] = random.uniform(0, 6.2)
        synth_df[i, 10] = f.random_int(0, 1)
        synth_df[i, 11] = f.random_int(0, 3)
        synth_df[i, 12] = f.random_int(0, 2)
        synth_df[i, 13] = f.random_int(0, 1)
    synth_df[[10, 20, 30, 40], 6] = 1
    synth_df[100:190, 9] = 0
    synth_df[100:120, 10] = 2
    synth_df[100:190, 11] = 0
    synth_df = pd.DataFrame(synth_df, columns=df.columns)
    synth_df = synth_df.sample(frac=1).reset_index(drop=True)
    features = synth_df.iloc[:, :-1]
    target = synth_df.iloc[:, -1]
    features.to_csv(f'/opt/airflow/data/raw/{output_dir}/data.csv')
    target.to_csv(f'/opt/airflow/data/raw/{output_dir}/target.csv')

default_args = {
    "owner": "Batyrkhan",
    # "depends_on_past": False,
    "start_date": datetime(2022, 12, 1),
    # "email_on_failure": False,
    # "email_on_retry": False,
    # "retries": 1,
    # "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="12_synthetic_data_generation",
    default_args=default_args,
    schedule_interval="0 1 * * 0-6",
) as dag:

    # show_folders = BashOperator(
    #     task_id="folder_show",
    #     bash_command='ls',
    # )

    # delete_folders = BashOperator(
    #     task_id="folder_delete",
    #     bash_command='rm -rf /opt/airflow/data/raw/{{ ds }}',
    # )

    new_folder = BashOperator(
        task_id="folder_creator",
        bash_command='mkdir -p /opt/airflow/data/raw/{{ ds }}',
    )

    generator = PythonOperator(
        task_id="data_gener",
        python_callable=_data_gen,
        op_kwargs={
            "output_dir": "{{ ds }}",
        }
    )

    notify = BashOperator(
        task_id="notify",
        bash_command='echo "data is generated"',
    )

    new_folder >> generator >> notify
