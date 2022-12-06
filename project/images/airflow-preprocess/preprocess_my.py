
import pandas as pd
import click
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    X = pd.read_csv(f'{input_dir}/data.csv').iloc[:,1:]
    y = pd.read_csv(f'{input_dir}/target.csv').iloc[:,-1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=False, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    pd.DataFrame(X_train_sc).to_csv(f'{output_dir}/train_features.csv')
    pd.DataFrame(X_val_sc).to_csv(f'{output_dir}/val_features.csv')
    pd.DataFrame(y_train).to_csv(f'{output_dir}/train_target.csv')
    pd.DataFrame(y_val).to_csv(f'{output_dir}/val_target.csv')


if __name__ == '__main__':
    preprocess()
