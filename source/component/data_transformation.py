import os
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from source.exception import ChurnException
from source.logger import logging
import category_encoders as ce
import warnings

warnings.filterwarnings('ignore')


class DataTransformation:

    def __init__(self, utility_config):
        self.utility_config = utility_config

    def feature_encoding(self, data, target, save_encoder_path=None, load_encoder_path=None, type=None):
        try:
            for col in self.utility_config.dt_binary_class_col:
                data[col] = data[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

            if target != '':
                data[self.utility_config.target_column] = data[self.utility_config.target_column].map(
                    {'Yes': 1, 'No': 0})

            if type == 'test':
                data[self.utility_config.target_column] = data[self.utility_config.target_column].map(
                    {'Yes': 1, 'No': 0})

            if save_encoder_path:
                encoder = ce.TargetEncoder(cols=self.utility_config.dt_multi_class_col)
                data_encoded = encoder.fit_transform(data[self.utility_config.dt_multi_class_col],
                                                     data[self.utility_config.target_column])

                with open(save_encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)

            if load_encoder_path:
                with open(load_encoder_path, 'rb') as f:
                    encoder = pickle.load(f)

                data_encoded = encoder.transform(data[self.utility_config.dt_multi_class_col])

            data = pd.concat([data.drop(columns=self.utility_config.dt_multi_class_col), data_encoded], axis=1)

            return data

        except ChurnException as e:
            raise e

    def min_max_scaling(self, data, type=None):
        if type == 'train':

            numeric_columns = list(data.select_dtypes(include=['float64', 'int64']).columns)

            scaler = MinMaxScaler()

            scaler.fit(data[numeric_columns])

            scaler_details = pd.DataFrame({'feature': numeric_columns,
                                           'Scaler_min': scaler.data_min_,
                                           'Scaler_max': scaler.data_max_})
            scaler_details.to_csv('source/ml/scaler_details.csv', index=False)

            scaled_data = scaler.transform(data[numeric_columns])
            data.loc[:, numeric_columns] = scaled_data
            data['Churn'] = self.utility_config.target_column

            print('done')

        else:
            scaler_details = pd.read_csv('source/ml/scaler_details.csv')

            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                data[col] = data[col].astype('float64')

                temp = scaler_details[scaler_details['feature'] == col]

                if not temp.empty:

                    min = temp.loc[temp.index[0], 'Scaler_min']
                    max = temp.loc[temp.index[0], 'Scaler_max']

                    data[col] = (data[col]-min)/ (max-min)

                else:
                    print(f"No scaling details available for feature {col}")
            data['Churn'] = self.utility_config.target_column

        return data

    def export_data_to_csv(self,train_data,test_data):
        dir_path = os.path.dirname(self.utility_config.dt_train_file_path)
        os.makedirs(dir_path, exist_ok=True)

        train_data.to_csv(self.utility_config.dt_train_file_path, index=False)
        test_data.to_csv(self.utility_config.dt_test_file_path, index=False)

    def export_data_file(self, data, file_name, path):
        try:
            dir_path = os.path.join(path)
            os.makedirs(dir_path, exist_ok=True)

            data.to_csv(path + '\\' + file_name, index=False)
            logging.info("Data transformation files exported")
        except ChurnException as e:
            raise e

    def initiate_data_transformation(self):
        train_data = pd.read_csv(self.utility_config.dv_train_file_path + '\\' + self.utility_config.train_file_name,
                                 dtype={'SeniorCitizen': 'object'})
        test_data = pd.read_csv(self.utility_config.dv_test_file_path + '\\' + self.utility_config.test_file_name,
                                dtype={'SeniorCitizen': 'object'})

        train_data = self.feature_encoding(train_data, target='Churn', save_encoder_path=self.utility_config.dt_multi_class_encoder)
        test_data = self.feature_encoding(test_data, target='', load_encoder_path=self.utility_config.dt_multi_class_encoder, type='test')



        self.utility_config.target_column = train_data['Churn']
        train_data.drop('Churn', axis=1, inplace=True)
        train_data = self.min_max_scaling(train_data, type='train')

        self.utility_config.target_column = test_data['Churn']
        test_data.drop('Churn', axis=1, inplace=True)
        test_data = self.min_max_scaling(test_data,type='test')

        self.export_data_file(train_data, self.utility_config.train_file_name, self.utility_config.dt_train_file_path)
        self.export_data_file(test_data, self.utility_config.test_file_name, self.utility_config.dt_test_file_path)

        print('done')
