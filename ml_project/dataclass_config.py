from dataclasses import dataclass


@dataclass
class Config:
    data_directory: str
    raw_data: str
    train_test_data: str
    synthetic_data: str
    train_data: str
    test_data: str
    synth_train_data: str
    synth_test_data: str
    data: str
    path_to_reports: str
    pd_report: str
    sweetviz_report: str
    logs_path: str
    data_log: str
    train_log: str
    predict_log: str
    categ_features: list
    numerical_features: list
    synth_train_size: int
    drop_columns: list
    target_name: str
    test_size: float
    model_directory: str
    model_name: str
    model: str
    y_pred_path: str
    max_iter: int
