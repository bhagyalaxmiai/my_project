import logging
import os
import pandas as pd
from pandas import DataFrame
from source.exception import ChurnException
from pymongo.mongo_client import MongoClient
from source.utility.utility import export_data_csv
from sklearn.model_selection import train_test_split
from source.logger import logging


class DataIngestion:

    def __init__(self, utility_config) -> DataFrame:
        self.utility_config = utility_config

    def export_data_into_feature_store(self, key):
        try:
            logging.info("start : data load from mongodb")
            if key == 'train':
                collection_name = self.utility_config.train_collection_name
                feature_store_dir_path = self.utility_config.train_feature_store_dir_path
            else:
                collection_name = self.utility_config.predict_collection_name
                feature_store_dir_path = self.utility_config.predict_di_feature_store_file_path

            client = MongoClient(self.utility_config.mongodb_url_key)
            database = client[self.utility_config.database_name]
            collection = database[collection_name]
            cursor = collection.find()

            data = pd.DataFrame(list(cursor))

            dir_path = os.path.dirname(feature_store_dir_path)
            os.makedirs(dir_path, exist_ok=True)  # if not exist then create
            data.to_csv(feature_store_dir_path, index=False)

            logging.info("Complete : data load from mongodb")

            return data

        except ChurnException as e:
            logging.error(e)
            raise e

    def split_data_test_train(self, data: DataFrame) -> None:
        try:
            logging.info("start: train, test data split")
            train_set, test_set = train_test_split(data, test_size=self.utility_config.train_test_split_ratio)
            dir_path = os.path.dirname(self.utility_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.utility_config.train_file_path, index=False)
            test_set.to_csv(self.utility_config.test_file_path, index=False)

            logging.info("complete: train, test data split")
        except ChurnException as e:
            raise e

    def clean_data(self, data):
        try:
            logging.info("start : clean data")

            data = data.drop_duplicates()

            data = data.loc[:, data.nunique() > 1]

            drop_column = []

            for col in data.select_dtypes(include=['object']).columns:
                unique_count = data[col].nunique()

                if unique_count / len(data) > 0.5:
                    data.drop(col, axis=1, inplace=True)
                    drop_column.append(col)

            logging.info(f"Dropped columns :{drop_column}")
            logging.info("start : clean data")

            return data

        except ChurnException as e:
            raise e

    def process_data(self, data, key):
        try:
            logging.info("Start : process data")
            if key == 'train':
                mandatory_cols = self.utility_config.mandatory_col_list.copy()
            else:
                mandatory_cols = self.utility_config.mandatory_col_list.copy()
                mandatory_cols.remove(self.utility_config.target_column)
                data = data.drop(self.utility_config.di_col_drop_in_clean, axis=1)

            for col in mandatory_cols:

                if col not in data.columns:
                    raise ChurnException(f"missing mandatory column: {col}")
                if data[col].dtype != self.utility_config.mandatory_col_data_type[col]:
                    try:
                        data[col] = data[col].astype(self.utility_config.mandatory_col_data_type[col])
                    except ValueError as e:
                        raise ChurnException(f"ERROR: converting data type for column: {col}")

            logging.info("complete: process data")

            return data

        except ChurnException as e:
            raise e

    def initiate_data_ingestion(self, key):

        data = self.export_data_into_feature_store(key)

        if key == 'train':
            data = self.clean_data(data)

        data = self.process_data(data, key)

        if key == 'train':
            self.split_data_test_train(data)

        if key == 'predict':
            export_data_csv(data, self.utility_config.predict_file, self.utility_config.predict_file_path)
