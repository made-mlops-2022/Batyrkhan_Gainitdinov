import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
import logging
import yaml
import os
import sys
sys.path.append('../../')
from dataclass_config import Config


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


CONFIG_PATH = "../../"
config = load_config("my_config#1.yaml")
config2 = load_config("my_config#2.yaml")
config_d = Config(*config.values())


def train(model, train_df, logger):
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    try:
        model.fit(X_train, y_train)
    except Exception as err:
        logger.error('model has not trained')
        logger.exception(err) 
    with open(os.path.join('../..', config2['model_path'], config2['model_name']), "wb") as f:
        logger.info(f'model {model} saved as pickle')
        pickle.dump(model, f)


if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join('../../', config_d.train_test_data, config_d.train_data))
    if config['model'] == 'KNN':
        config = config2
        model = KNeighborsClassifier(
            n_neighbors=config['n_neighbors'],
            weights=config['weights'],
            algorithm=config['algorithm'],
            leaf_size=config['leaf_size'],
            p=config['p'],
            metric=config['metric'],
            n_jobs=config['n_jobs'])
    else:
        model = LogisticRegression(max_iter=config_d.max_iter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join('../', config_d.logs_path, config_d.train_log))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'model {model} started to train')
    train(model, train_df, logger)
