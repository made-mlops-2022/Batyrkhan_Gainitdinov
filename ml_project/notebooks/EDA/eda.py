import pandas as pd
from pandas_profiling import ProfileReport
import sweetviz as sv
import yaml
import os


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


CONFIG_PATH = "../../"
config = load_config("my_config#1.yaml")
df = pd.read_csv(os.path.join('../../', config['raw_data'], config['data']))
profile = ProfileReport(df, explorative=True)
profile.to_file(os.path.join('../../', config['path_to_reports'], config['pd_report']))
sweet_report = sv.analyze(df)
sweet_report.show_html(os.path.join('../../', config['path_to_reports'], config['sweetviz_report']))