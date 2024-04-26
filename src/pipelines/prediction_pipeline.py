import sys 
import os 
import pandas as pd
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            error_message = f"Error occurred in predict function: {str(e)}"
            logging.error(error_message)
            raise CustomException(error_message, sys)

class CustomData:
    def __init__(self, month:str,
                 day:int,
                 order:int,
                 country:str,
                 sessionID:int,
                 page1_main_category:str,
                 page2_clothing_model:str,
                 colour:str,
                 price:int):
        self.month = month
        self.day = day
        self.order = order
        self.country = country
        self.sessionID = sessionID
        self.page1_main_category= page1_main_category
        self.page2_clothing_category = page2_clothing_model  # Fixed variable name
        self.colour = colour
        self.price = price

    def get_data_as_dataframe(self):
        try: 
            custom_data_input_dict = {
                'month': [self.month],
                'day': [self.day],
                'order': [self.order],
                'country': [self.country],
                'sessionID': [self.sessionID],
                'page1_main_category': [self.page1_main_category],
                'page2_clothing_category': [self.page2_clothing_category],
                'colour': [self.colour],
                'price': [self.price]    
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe created")
            return df
        except Exception as e:
            error_message = f"Error occurred in get_data_as_dataframe function: {str(e)}"
            logging.error(error_message)
            raise CustomException(error_message, sys)
