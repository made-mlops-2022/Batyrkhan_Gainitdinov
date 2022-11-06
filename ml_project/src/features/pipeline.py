import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from feature_selector import FeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yaml
import os


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


CONFIG_PATH = "../../"
config = load_config("my_config#1.yaml")


def pipeline():
    categorical_features = config['categ_features']
    numerical_features = config['numerical_features']
    categorical_pipeline = Pipeline(steps=[
        ('cat_selector', FeatureSelector(categorical_features)),
        ('one_hot_encoder', OneHotEncoder(sparse=False))
        ])
    numerical_pipeline = Pipeline(steps=[
        ('num_selector', FeatureSelector(numerical_features)),
        ('std_scaler', StandardScaler())
        ])
    full_pipeline = FeatureUnion(transformer_list=[
        ('categorical_pipeline', categorical_pipeline),
        ('numerical_pipeline', numerical_pipeline)
        ])
    data = pd.read_csv(os.path.join('../../', config['raw_data'], config['data']))
    X = data.drop(config['target_name'], axis=1)
    y = data[config['target_name']].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=42)
    full_pipeline_m = Pipeline(steps=[
        ('full_pipeline', full_pipeline),
        ('model', LogisticRegression(max_iter=config['max_iter']))
        ])
    full_pipeline_m.fit(X_train, y_train)
    y_pred = full_pipeline_m.predict(X_test)
    return y_test, y_pred


if __name__ == '__main__':
    y_test, y_pred = pipeline()
