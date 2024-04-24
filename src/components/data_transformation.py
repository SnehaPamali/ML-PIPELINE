import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_function

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_features = ['month', 'country', 'page 1 (main category)', 'page 2 (clothing model)', 'colour']
            numerical_features = ['year', 'day', 'order', 'session ID', 'model photography', 'price', 'page']


            # Define the custom ranking for each ordinal variable
            mon = ['April', 'May', 'June', 'July', 'August']
            colour1 = ['beige', 'olive', 'gray', 'brown', 'burgundy', 'of many colors',
       'red', 'pink', 'black', 'blue', 'violet', 'white', 'green',
       'navy blue']
            country1 =['Poland', 'Ireland', 'Czech Republic', 'Germany', 'Switzerland',
       'Lithuania', 'United Kingdom', 'unidentified', 'Romania',
       'com (*.com)', 'Slovakia', 'Norway', 'net (*.net)', 'France',
       'Denmark', 'Netherlands', 'Croatia', 'Luxembourg', 'India',
       'Greece', 'Italy', 'Iceland', 'USA', 'Cyprus', 'Belgium', 'Sweden',
       'Portugal', 'Finland', 'biz (*.biz)', 'Latvia', 'org (*.org)',
       'Spain', 'Austria', 'San Marino', 'Russia', 'int (*.int)',
       'Hungary', 'Estonia', 'United Arab Emirates', 'Christmas Island',
       'Ukraine', 'Australia', 'Slovenia', 'Faroe Islands',
       'British Virgin Islands', 'Mexico', 'Cayman Islands']
            page1 = ['trousers', 'skirts', 'blouses', 'sale']
            page2 = ['A13', 'A16', 'B4', 'B17', 'B8', 'C56', 'C57', 'P67', 'P82', 'B31',
       'B21', 'B24', 'B27', 'A10', 'P1', 'P34', 'P33', 'C4', 'C7', 'C10',
       'C17', 'P77', 'A34', 'A37', 'C25', 'C21', 'C15', 'C53', 'B26',
       'A11', 'C5', 'P60', 'P56', 'P55', 'P48', 'P50', 'P42', 'P23',
       'C49', 'B23', 'C19', 'C34', 'C40', 'C50', 'C42', 'A18', 'A1', 'B1',
       'B16', 'A3', 'B3', 'B30', 'P16', 'A2', 'A5', 'A41', 'B2', 'B13',
       'B15', 'B9', 'B20', 'B25', 'B33', 'B34', 'C2', 'C33', 'C35', 'C55',
       'C59', 'P61', 'P62', 'A6', 'C47', 'B32', 'A17', 'A7', 'A8', 'A9',
       'A20', 'A32', 'C11', 'C22', 'P12', 'A4', 'A23', 'B12', 'B10',
       'C14', 'A12', 'A31', 'A15', 'B7', 'C8', 'C18', 'P2', 'P10', 'C26',
       'C31', 'A35', 'A36', 'C46', 'P29', 'A21', 'P63', 'P30', 'P32',
       'P66', 'P57', 'P43', 'P4', 'A28', 'B28', 'P15', 'A19', 'B19',
       'B14', 'B6', 'A29', 'P49', 'C3', 'C9', 'C36', 'C37', 'C39', 'C51',
       'C52', 'A42', 'B29', 'B11', 'P6', 'P17', 'P11', 'C1', 'C13', 'A14',
       'A24', 'P20', 'P40', 'P51', 'P25', 'C44', 'C45', 'B22', 'A30',
       'C58', 'C54', 'A26', 'A33', 'A38', 'A39', 'P8', 'P19', 'P39',
       'P80', 'P78', 'P76', 'A40', 'B5', 'P3', 'P7', 'P14', 'P18', 'P35',
       'C6', 'C12', 'C30', 'P5', 'P70', 'P36', 'P9', 'P26', 'P52', 'P64',
       'P37', 'P69', 'A22', 'C41', 'P13', 'C16', 'P46', 'C48', 'P81',
       'P38', 'P47', 'P44', 'P59', 'P65', 'C38', 'C20', 'C27', 'C28',
       'C43', 'P58', 'P21', 'P72', 'P73', 'C24', 'P41', 'C29', 'P71',
       'C32', 'C23', 'A27', 'P74', 'P68', 'P75', 'A25', 'P27', 'P24',
       'A43', 'P31', 'P53', 'P45', 'P54', 'P28', 'P22', 'P79']
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[mon, country1, colour1, page1, page2])),
                ('scaler',StandardScaler())
                ]

            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_features),
            ('cat_pipeline',cat_pipeline,categorical_features)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'page']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_function(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)