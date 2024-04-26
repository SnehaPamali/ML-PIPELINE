import os
import pickle
import sys
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from sqlalchemy import create_engine
import pandas as pd
import urllib.parse


def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]
# Train models
            model.fit(X_train, y_train)
# Test data
            y_test_pred = model.predict(X_test)
            #R2 Score 
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except FileNotFoundError:
        error_message = f"File '{file_path}' not found"
        logging.error(error_message)
        raise CustomException(error_message, sys)
    except EOFError:
        error_message = f"Unexpected end of file while loading '{file_path}'"
        logging.error(error_message)
        raise CustomException(error_message, sys)
    except Exception as e:
        error_message = f"Error occurred while loading '{file_path}': {str(e)}"
        logging.error(error_message)
        raise CustomException(error_message, sys)
    
def save_function(file_path, obj): 
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok= True)
    with open (file_path, "wb") as file_obj: 
        pickle.dump(obj, file_obj)

def import_data_from_sql(connection_string, query):
    try:
        password = "Sneha@11"
        encoded_password = urllib.parse.quote_plus(password)
        # Assuming you're using SQLAlchemy
        engine = create_engine(f'mysql://{"root"}:{encoded_password}@{"localhost"}/{"eshopdataset"}')
        # Execute SQL query and load data into a DataFrame
        query = "SELECT * From eshopdata"
        df1 = pd.read_sql(query, engine)
        return df1
    except Exception as e:
        error_message = f"Error occurred while importing data from SQL: {str(e)}"
        logging.error(error_message)
        raise CustomException(error_message, sys)