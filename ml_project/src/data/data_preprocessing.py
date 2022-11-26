import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
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
df = pd.read_csv(os.path.join('../../', config['raw_data'], config['data']))
categorical_features = config['categ_features']
numerical_features = config['numerical_features']
scaler = StandardScaler()
for col in numerical_features:
    df[col] = scaler.fit_transform(df[[col]])
transf = make_column_transformer(
    (OneHotEncoder(), categorical_features), remainder='passthrough'
    )
transformed = transf.fit_transform(df)
transformed_df = pd.DataFrame(transformed, columns=transf.get_feature_names_out())
transformed_df = transformed_df.sample(frac=1).reset_index(drop=True)
logger.info('data has preprocessed')
train_df = transformed_df.iloc[:200, :]
test_df = transformed_df.iloc[200:, :]
train_df.to_csv(os.path.join('../../', config['train_test_data'], config['train_data']))
test_df.to_csv(os.path.join('../../', config['train_test_data'], config['test_data']))
logger.info('train and test dataframes are saved to csv format')
