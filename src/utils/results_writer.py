import sys
sys.path.append("../")
import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.utils import check_dir_exists, explore_list

class ResultsWriter:
    def __init__(self, results_dir, dataset_name):
        self.results_dir = results_dir
        self.dataset_name = dataset_name
        self.results = {}

        check_dir_exists(self.results_dir)

    def write_dl_results(self, y_train, y_val, ground_truth, predictions, model_name, indent=None):
        data = {
            "dataset": self.dataset_name,
            "model": model_name,
            "num_test_samples": len(ground_truth),
            "mean_ground_truth": str(ground_truth.mean()),
            "mean_predictions": str(predictions.mean()),
            "MAE": mean_absolute_error(ground_truth, predictions),
            "MSE": mean_squared_error(ground_truth, predictions),
            "RMSE": np.sqrt(mean_squared_error(ground_truth, predictions)),
            "R2_score": r2_score(ground_truth, predictions),
            "test": np.ravel(ground_truth).tolist(),
            "prediction": np.ravel(predictions).tolist(),
            "train": explore_list(y_train),
            "val": explore_list(y_val),
        }

        dir_path = os.path.join(self.results_dir, self.dataset_name, "DL_models")
        check_dir_exists(dir_path)
        info_file = os.path.join(dir_path, "average_results.json")
        self.write_output(data, info_file, indent=indent)

    def write_fold_results(self, fold_num, model_name, metrics, predictions, indent=None):
        self.fold_dir = os.path.join(self.results_dir, self.dataset_name, "ML_regression_models")
        model_dir = os.path.join(self.fold_dir, f"Fold_{fold_num}", model_name)
        check_dir_exists(model_dir)

        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][f"Fold_{fold_num}"] = {"metrics": metrics}

        metrics_file = os.path.join(model_dir, "metrics.json")
        self.write_output(metrics, metrics_file, indent=indent)

        predictions_file = os.path.join(model_dir, "predictions.json")
        self.write_output(predictions, predictions_file, indent=indent)

    def write_average_results(self, final_output, indent=None):
        check_dir_exists(self.fold_dir)
        info_file = os.path.join(self.fold_dir, "average_results.json")
        self.write_output(final_output, info_file, indent=indent)

    @staticmethod
    def write_output(data, file_path, indent=None):
        with open(file_path, 'w') as output_file:
            json.dump(data, output_file, indent=indent)
