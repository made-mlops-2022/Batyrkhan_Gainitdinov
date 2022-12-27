import pandas as pd
import pickle

import click


@click.command("predictions")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--predictions-dir")
def predictions(input_dir: str, model_dir: str, predictions_dir: str):
    X_val = pd.read_csv(f'{input_dir}/val_features.csv').iloc[:, 1:]
    with open(model_dir + '/' + 'model.pkl', "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_val)
    report_df = pd.DataFrame(y_pred)
    report_df.to_csv(f'{predictions_dir}/predictions.csv')


if __name__ == '__main__':
    predictions()