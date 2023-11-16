import sys

sys.path.append("../")
import os
import pandas as pd
from config.variables import features_path
from utils.utils import check_dir_exists, create_date_date_features, create_date_lagged_3_features, \
    create_date_lagged_3_features

import warnings

warnings.filterwarnings('ignore')


class DatasetGeneration:
    def __init__(self, path_file):
        self.df = pd.read_csv(path_file)

    def dataset_generation(self):
        df = self.df[["Date_time", "totalU"]]
        df['Date_time'] = pd.to_datetime(df['Date_time'])
        df = df.set_index('Date_time')
        check_dir_exists(features_path)

        # Dataset with new features ['day', 'month', 'year', 'quarter', 'dayofweek', 'dayofyear']
        date_df = df.copy()
        date_df = create_date_date_features(date_df)
        date_df.to_csv(os.path.join(features_path, 'date_smedebtsu.csv'))
        print("File path: ", os.path.join(features_path, 'date_smedebtsu.csv'))

        # Dataset with new features ['totalU_lag1', 'totalU_lag2', 'totalu_lag3']
        lag3_df = df.copy()
        lag3_df = create_date_lagged_3_features(lag3_df)
        lag3_df.to_csv(os.path.join(features_path, 'lag3_smedebtsu.csv'))
        print("File path: ", os.path.join(features_path, 'lag3_smedebtsu.csv'))

        # Dataset with new features ['day', 'month', 'year', 'quarter', 'dayofweek', 'dayofyear'
        #                             'total_debts_lag1', 'total_debts_lag2','total_debts_lag3']
        lag3_df_date_df = create_date_lagged_3_features(date_df)
        lag3_df_date_df.to_csv(os.path.join(features_path, 'lag3_date_smedebtsu.csv'))
        print("File path: ", os.path.join(features_path, 'lag3_date_smedebtsu.csv'))

        return
