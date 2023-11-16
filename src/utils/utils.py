import sys

sys.path.append("../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import itertools
from config.variables import metrics_evaluate, day_split_test


def check_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_date_date_features(data_frame):
    """
    Create new features such as day, month, year, quarter
    Parameter:
        data_frame (pandas DataFrame): The DataFrame to which new columns will be added
    Returns:
        data_frame (pandas DataFrame): The input DataFrame with new date/time features added as columns
    """
    data_frame['day'] = data_frame.index.day
    data_frame['month'] = data_frame.index.month
    data_frame['year'] = data_frame.index.year
    data_frame['quarter'] = data_frame.index.quarter
    data_frame['dayofweek'] = data_frame.index.dayofweek
    data_frame['dayofyear'] = data_frame.index.dayofyear

    return data_frame


def create_date_lagged_3_features(data_frame):
    # Create features totalU_lag1, totalU_lag2, totalU_lag3
    data_frame['totalU_lag1'] = data_frame['totalU'].shift(1)
    data_frame['totalU_lag2'] = data_frame['totalU'].shift(2)
    data_frame['totalU_lag3'] = data_frame['totalU'].shift(3)
    data_frame = data_frame.dropna()
    return data_frame


def split_data_statistical_models(df):
    train_df = df[df['Date_time'] < day_split_test]
    test_df = df[df['Date_time'] >= day_split_test]

    print("--------------Total Debts-----------------")
    print(f"Train Size: {len(train_df)}, Test Size: {len(test_df)}")
    return train_df, test_df


def make_predictions_and_print_rmse(model, test_df):
    print(f"forecasting and RMSE of total debts")

    forecast, confidence_interval = model.predict(X=test_df, n_periods=len(test_df), return_conf_int=True)
    forecasts = pd.Series(forecast, index=test_df[:len(test_df)].index)
    lower = pd.Series(confidence_interval[:, 0], index=test_df[:len(test_df)].index)
    upper = pd.Series(confidence_interval[:, 1], index=test_df[:len(test_df)].index)

    rmse = np.sqrt(np.mean((forecast.values - test_df.values) ** 2))

    print("RMSE is: ", rmse)

    return forecasts, lower, upper


def plot_predictions(train_values, test_values, predicted_values, lower_confidence, upper_confidence):
    plt.figure(figsize=(10, 6))
    plt.plot(train_values.index, train_values, label='Train Values')
    plt.plot(test_values.index, test_values, label='Test Values')
    plt.plot(predicted_values.index, predicted_values, color='red', label='Predicted Values')
    plt.fill_between(lower_confidence.index, lower_confidence, upper_confidence, color='gray', alpha=0.3,
                     label='Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Total Debts')
    plt.title('Total Debts: Train, Predicted, and True Values with Confidence Interval')
    plt.legend()
    plt.show()
    return


def compute_mean_metric(metrics_list, n_cv_folds, method="RMSE"):
    if method not in metrics_evaluate:
        raise ValueError(f"Invalid method: {method}. Available methods: {metrics_evaluate}")
    mean_metric = sum(element[method] for element in metrics_list) / n_cv_folds
    return mean_metric


def get_y_train(df):
    data_train = df[df.index < day_split_test]
    y_all_train = data_train['totalU'].tolist()
    return y_all_train


def explore_list(lst):
    """
    explore nested list into a single-level list.

    Example:
        #>>> nested_list = [[1, 2], [5, 6]]
        #>>> explore_list(nested_list)
        [1, 2, 5, 6]
    """
    return list(itertools.chain(*lst))


def get_metric_files(result_path, method):
    rmse_benchmark = {
        "linear_regression": [],
        "random_forest_regressor": [],
        "gradient_boosting_regressor": [],
        "decision_tree_regressor": [],
        'RGU': [],
        'dataset': []
    }

    for dataset in os.listdir(result_path):
        dataset_path = os.path.join(result_path, dataset)
        for model_dir in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, model_dir, 'average_results.json')
            with open(file_path) as f:
                data = json.load(f)
                if model_dir == 'DL_models':
                    rmse_benchmark['dataset'].append(data['dataset'])
                    rmse_benchmark['RGU'].append(data[method])
                elif model_dir == 'ML_regression_models':
                    for element in data:
                        model_name = element["model"]
                        if model_name in rmse_benchmark:
                            rmse_benchmark[model_name].append(element[f'mean_{method}'])

    return rmse_benchmark
