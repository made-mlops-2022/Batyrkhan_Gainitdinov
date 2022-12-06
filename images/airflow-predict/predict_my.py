import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

import click


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
def predict(input_dir: str, model_dir: str):
    X_train = pd.read_csv(f'{input_dir}/train_features.csv').iloc[:, 1:]
    y_train = pd.read_csv(f'{input_dir}/train_target.csv').iloc[:, -1]

    model = LogisticRegression(max_iter=500, penalty="l2")
    model.fit(X_train, y_train)
    with open(model_dir + '/' + 'model.pkl', "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    predict()
