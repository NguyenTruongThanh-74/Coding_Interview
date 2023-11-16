import sys
sys.path.append("../")
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_processing.data_processing_helper import preprocess_data
from utils.results_writer import ResultsWriter
from utils.utils import compute_mean_metric, get_y_train, explore_list
from config.variables import features_path, result_path, ml_models,N_SPLITS, MAX_TRAIN_SIZE, TEST_SIZE, NUM_OF_TEST,ml_name_map,feature_file_list

import warnings

warnings.filterwarnings('ignore')

# Code


class MachineLearningModel:
    def __init__(self):
        self.ml_models = ml_models
        self.file_list = feature_file_list

    def plot_chart(self,result,axis, position_index):
        ax =  axis[position_index]
        train_result = result["train"]
        test_result = result["test"]
        predictions_result= result["prediction"]
        algorithm_name = result["model"]
        chart_title = ml_name_map[algorithm_name]
        
        split_index = len(train_result)
        df = pd.DataFrame(train_result + test_result)
        
        train_data = df[:split_index]
        test_data = df[split_index:]
        
        ax.plot(train_data.index, train_data[0], color="blue")
        ax.plot(test_data.index, test_data[0], color="red")
        ax.plot(test_data.index, predictions_result, color="green")
        
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Total Debts", fontsize=12)
        ax.axvline(test_data.index[0], color='black', ls='--')
        ax.legend(["Train data", "Test data", "Predicted data"])
        
        ax.title.set_text(f"{chart_title}")
    def plot_synthesize_charts(self,results_path):
        with open(results_path) as f:
            results = json.load(f)

        figure, axis = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10, 8))
        plt.figure(figsize=(10, 6))
        axis = axis.ravel()

        for idx, result in enumerate(results):
            self.plot_chart(result,axis, idx)
        figure.tight_layout()

        plt.show()

    def model_train(self,df,ml_result_test,ml_result_prediction,results_writer):
        tss = TimeSeriesSplit(n_splits=N_SPLITS, max_train_size=MAX_TRAIN_SIZE, test_size=TEST_SIZE)
        for fold_num, (train_index, test_index) in enumerate(tss.split(df)):
            train = df.iloc[train_index]
            test = df.iloc[test_index]

            TARGET = 'totalU'
            FEATURES = [feature for feature in df.columns if feature != TARGET]

            X_train, y_train = train[FEATURES], train[TARGET]
            X_test, y_test = test[FEATURES], test[TARGET]

            for ml_model, reg in self.ml_models.items():
                reg.fit(X_train, y_train)
                y_prediction = reg.predict(X_test)
                # -------------------------------------------------------------------------------------
                mae = mean_absolute_error(y_test, y_prediction)
                mse = mean_squared_error(y_test, y_prediction)
                rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
                r2 = r2_score(y_test.tolist(), y_prediction)
                # ---------------------------- Writing Output -----------------------------------------
                metrics = {
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2_score": r2
                }

                predictions = {
                    "date": y_test.index.strftime('%Y-%m-%d').tolist(),
                    "num_test_samples": len(y_test),
                    "target": y_test.tolist(),
                    "predictions": y_prediction.tolist()
                }
                ml_result_test[ml_model].append(y_test.tolist())
                ml_result_prediction[ml_model].append(y_prediction.tolist())
                results_writer.write_fold_results(fold_num + 1, ml_model, metrics, predictions, indent=2)
        return ml_result_test,ml_result_prediction

    def caculate_evaluate_metrics(self,df,file_name,ml_result_test,ml_result_prediction,results_writer):
        final_output_list = []
        rmse_benchmark = {}
        for ml_model in self.ml_models.keys():
            metrics_list = []
            for fold_num in range(1, N_SPLITS + 1):
                fold_results = results_writer.results[ml_model][f"Fold_{fold_num}"]["metrics"]
                metrics_list.append(fold_results)

            mean_mae = compute_mean_metric(metrics_list, N_SPLITS, method="MAE")
            mean_mse = compute_mean_metric(metrics_list, N_SPLITS, method="MSE")
            mean_rmse = compute_mean_metric(metrics_list, N_SPLITS, method="RMSE")
            mean_r2 = compute_mean_metric(metrics_list, N_SPLITS, method="R2_score")

            # ---------------------------- Writing Output -----------------------------------------
            # it is convenient for data visualization
            all_train = get_y_train(df)
            final_output = {
                "dataset": file_name.split('.')[0],
                "K_folds": N_SPLITS,
                "model": ml_model,
                "num_test_samples": NUM_OF_TEST,
                "mean_MAE": mean_mae,
                "mean_MSE": mean_mse,
                "mean_RMSE": mean_rmse,
                "mean_R2_score": mean_r2,
                "train": all_train,
                "test": explore_list(ml_result_test[ml_model]),
                "prediction": explore_list(ml_result_prediction [ml_model])
            }

            final_output_list.append(final_output)
            rmse_benchmark.update({ml_model: mean_rmse})
        results_writer.write_average_results(final_output_list, indent=2)
    def run_ml_models_results(self):
        for file_name in feature_file_list:
            print(datetime.datetime.now(), " File {}".format(file_name))
            results_writer = ResultsWriter(result_path, file_name.split('.')[0])
            ml_result_test = {
                "linear_regression": [],
                "random_forest_regressor": [],
                "gradient_boosting_regressor":[],
                "decision_tree_regressor": []

            }

            ml_result_prediction = {
                "linear_regression": [],
                "random_forest_regressor": [],
                "gradient_boosting_regressor":[],
                "decision_tree_regressor":[]
            }
            # Read file csv
            dataset_path = os.path.join(features_path, file_name)
            df = pd.read_csv(dataset_path)
            df = preprocess_data(df)
            df = df.set_index('Date_time')
            self.model_train(df,ml_result_test,ml_result_prediction,results_writer)
            self.caculate_evaluate_metrics(df,file_name,ml_result_test,ml_result_prediction,results_writer)

