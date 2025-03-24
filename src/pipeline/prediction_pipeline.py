import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd, numpy as np

# Set the earth's radius (in kilometers)
R = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

# Function to calculate the distance between two points using the haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# class PredictPipeline:
#     def __init__(self):
#         pass

#     def predict(self,features):
#         try:
#             preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
#             model_path = os.path.join('artifacts','model.pkl')

#             preprocessor = load_object(preprocessor_path)
#             model = load_object(model_path)

#             data_scaled = preprocessor.transform(features)

#             pred = model.predict(data_scaled)

#             return pred

#         except Exception as e:
#             logging.info("Exception occured in prediction")
#             raise CustomException(e,sys)
        
#23/02/2025
import os
import sys
import logging
import glob
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.models_dir = os.path.join('artifacts', 'models')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.evaluation_log_path = os.path.join('logs', 'model_predictions.log')
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging for model predictions
        self.prediction_logger = logging.getLogger('model_predictions')
        self.prediction_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.evaluation_log_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.prediction_logger.addHandler(file_handler)

    def load_models(self):
        """Load all models from the models directory"""
        try:
            models = {}
            model_files = glob.glob(os.path.join(self.models_dir, '*.pkl'))
            
            for model_path in model_files:
                model_name = os.path.basename(model_path).replace('.pkl', '')
                models[model_name] = load_object(model_path)
            
            return models
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # Load preprocessor
            preprocessor = load_object(self.preprocessor_path)
            
            # Transform features
            data_scaled = preprocessor.transform(features)
            
            # Load all models
            models = self.load_models()
            
            # Dictionary to store predictions from all models
            predictions = {}
            
            # Make predictions with each model
            for model_name, model in models.items():
                try:
                    pred = model.predict(data_scaled)
                    predictions[model_name] = round(pred[0], 2)
                    
                    # Log the prediction
                    self.prediction_logger.info(
                        f"Model: {model_name}, Prediction: {predictions[model_name]} minutes"
                    )
                except Exception as e:
                    self.prediction_logger.error(
                        f"Error with model {model_name}: {str(e)}"
                    )
                    continue
            
            return predictions

        except Exception as e:
            logging.info("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                Delivery_person_Age:int,
                Delivery_person_Ratings:float,
                Restaurant_latitude:float,
                Restaurant_longitude:float,
                Delivery_location_latitude:float,
                Delivery_location_longitude:float,
                Weather_conditions:str,
                multiple_deliveries:float,
                Festival:str,
                City:str,
                Road_traffic_density:str,
                Vehicle_condition:int,
                Type_of_vehicle:str):
    
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Restaurant_latitude=Restaurant_latitude
        self.Restaurant_longitude=Restaurant_longitude
        self.Delivery_location_latitude=Delivery_location_latitude
        self.Delivery_location_longitude=Delivery_location_longitude
        self.Weather_conditions = Weather_conditions
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_vehicle = Type_of_vehicle
    
    def calculate_distance(self):
        return calculate_distance(
            self.Restaurant_latitude,
            self.Restaurant_longitude,
            self.Delivery_location_latitude,
            self.Delivery_location_longitude
        )

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Restaurant_latitude': [self.Restaurant_latitude],
                'Restaurant_longitude': [self.Restaurant_longitude],
                'Delivery_location_latitude': [self.Delivery_location_latitude],
                'Delivery_location_longitude': [self.Delivery_location_longitude],
                'Weather_conditions': [self.Weather_conditions],
                'multiple_deliveries': [self.multiple_deliveries],
                'Festival': [self.Festival],
                'City': [self.City],
                'Road_traffic_density': [self.Road_traffic_density],
                'Vehicle_condition': [self.Vehicle_condition],
                'Type_of_vehicle': [self.Type_of_vehicle],
                'distance': [self.calculate_distance()]  # Calculate distance here
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.error('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)