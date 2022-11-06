from faker import Faker
import pandas as pd
import numpy as np
import random
import logging
import yaml
import os


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


CONFIG_PATH = "../../"
config = load_config("my_config#1.yaml")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join('../', config['logs_path'], config['data_log']))
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
f = Faker()
df = pd.read_csv(os.path.join('../../', config['raw_data'], config['data']))
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
train_synth_df = synth_df.iloc[:config['synth_train_size'], :]
test_synth_df = synth_df.iloc[config['synth_train_size']:, :]
logger.info('synthetic data has generated')
train_synth_df.to_csv(os.path.join('../../', config['synthetic_data'], config['synth_train_data']))
test_synth_df.to_csv(os.path.join('../../', config['synthetic_data'], config['synth_test_data']))
