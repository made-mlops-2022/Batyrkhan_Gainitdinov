import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import logging
sys.path.append('../features/')
sys.path.append('../models/')
from feature_selector import FeatureSelector
from train_model import train
from predict_model import predict
import yaml
import os


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


CONFIG_PATH = "../../"
config = load_config("my_config#1.yaml")
config2 = load_config("my_config#2.yaml")


def test_if_features_selected_correctly():
    numerical_features = config['numerical_features']
    selector = FeatureSelector(numerical_features)
    df = pd.read_csv(os.path.join('../../', config['raw_data'], config['data']))
    df = selector.transform(df)
    assert df.shape[1] == 5


def test_weights_of_classifier_are_not_empty():
    model = LogisticRegression(max_iter=config['max_iter'])
    train_df = pd.read_csv(os.path.join('../../', config['train_test_data'], config['train_data']))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join('../', config['logs_path'], config['train_log']))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'model {model} started to train')
    train(model, train_df, logger)
    with open(os.path.join('../..', config2['model_path'], config2['model_name']), "rb") as f:
        model = pickle.load(f)
    assert len(model.coef_[0]) == 29
    assert sum(model.coef_[0]) != 0


def test_accuracy_more_than_random_guess():
    test_df = pd.read_csv(os.path.join('../../', config['train_test_data'], config['test_data']))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join('../', config['logs_path'], config['predict_log']))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    y_test, y_pred = predict(test_df, logger)
    assert accuracy_score(y_test, y_pred) > 0.5


def test_train_on_synthetic_data():
    model = LogisticRegression(max_iter=config['max_iter'])
    train_df = pd.read_csv(os.path.join('../../', config['synthetic_data'], config['synth_train_data']))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join('../', config['logs_path'], config['train_log']))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'model {model} started to train')
    train(model, train_df, logger)
    with open(os.path.join('../..', config2['model_path'], config2['model_name']), "rb") as f:
        model = pickle.load(f)
    assert len(model.coef_[0]) == 14
    assert sum(model.coef_[0]) != 0


def test_predict_on_synthetic_data():
    test_df = pd.read_csv(os.path.join('../../', config['synthetic_data'], config['synth_test_data']))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join('../', config['logs_path'], config['predict_log']))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    y_test, y_pred = predict(test_df, logger)
    assert accuracy_score(y_test, y_pred) > 0.5
