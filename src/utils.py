import pickle
import sys
from src.exception import CustomException
from src.logger import logging

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
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

