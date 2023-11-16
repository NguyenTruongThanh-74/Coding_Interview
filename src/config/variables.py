import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
# The list of files
feature_file_list = ['date_smedebtsu.csv', 'lag3_smedebtsu.csv', 'lag3_date_smedebtsu.csv']

features_path = '../../data/features'
data_path = '../../data/processed/processed_smedebtsu.csv'
result_path = '../../results'
dataset_name = 'smedebtsu'

# Parameters
N_SPLITS = 5
MAX_TRAIN_SIZE = None
TEST_SIZE = 4
NUM_OF_TEST = N_SPLITS * TEST_SIZE

# Name of metrics
metrics_evaluate = ["MAE", "MSE", "RMSE", "R2_score"]
tested_model_list = {
    'linear_regression': 'Linear Regression',
    'random_forest_regressor': 'Random Forest Regressor',
    'gradient_boosting_regressor' :'Gradient Boosting Regressor',
    'decision_tree_regressor':'Decision Tree Regressor'
}

# The day for the split dataset for the deep-learning model
day_split_test = '2021-04-25'

# Deep Learning parameters
TRAIN_SIZE_RATIO = 0.8
EPOCHS = 10
BATCH_SIZE = 16


ml_models = {
    "linear_regression": LinearRegression(),
    "random_forest_regressor": RandomForestRegressor(),
    "gradient_boosting_regressor" :GradientBoostingRegressor(),
    "decision_tree_regressor":DecisionTreeRegressor()
}
# Dict

ml_name_map = {
    'linear_regression': 'Linear Regression',
    'random_forest_regressor': 'Random Forest Regressor',
    'gradient_boosting_regressor' :'Gradient Boosting Regressor',
    'decision_tree_regressor':'Decision Tree Regressor',
}