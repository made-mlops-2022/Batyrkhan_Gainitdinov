import pandas as pd
import pickle
import logging
import yaml
import os


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


CONFIG_PATH = "../../"
config = load_config("my_config#1.yaml")
config2 = load_config("my_config#2.yaml")


def predict(test_df, logger):
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    with open(os.path.join('../..', config2['model_path'], config2['model_name']), "rb") as f:
        model = pickle.load(f)
        logger.info('model successfuly loaded for prediction')
    try:
        y_pred = model.predict(X_test)
        assert y_pred.shape == y_test.shape
    except AssertionError as err:
        logger.error('shapes are not the same')
        logger.exception(err)
    else:
        logger.info('model predicted test values')
    return y_test, y_pred


if __name__ == '__main__':
    test_df = pd.read_csv(os.path.join('../../', config['train_test_data'], config['test_data']))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join('../', config['logs_path'], config['predict_log']))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    y_test, y_pred = predict(test_df, logger)
    pd.Series(y_pred).to_csv(os.path.join('../../', config['y_pred_path'], 'prediction.csv'))
