# import os,sys,numpy as np,pandas as pd
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# import numpy as np
# # from catboost import CatBoostRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from src.logger import logging
# from src.exception import CustomException
# from src.utils import save_object
# from src.utils import evaluate_model

# from dataclasses import dataclass

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join('artifacts','model.pkl')

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initate_model_training(self,train_array,test_array):
#         try:
#             logging.info('Splitting Dependent and Independent variables from train and test data')
#             X_train, y_train, X_test, y_test = (
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )

#             models = {
#                 'LinearRegression':LinearRegression(),
#                 'Lasso':Lasso(),
#                 'Ridge':Ridge(),
#                 'Elasticnet':ElasticNet(),
#                 'Gradient Boost':GradientBoostingRegressor(),
#                 'RandomForestRegressor' : RandomForestRegressor(),
#                 # 'CatBoost':CatBoostRegressor()
#             }

#             model_report = evaluate_model(X_train,y_train,X_test,y_test,models) 
#             print(model_report)
#             print('\n====================================================================================\n')
#             logging.info(f'Model Report : {model_report}')

#             #TO get best model score from dictionary
#             best_model_score = max(sorted(model_report.values()))

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]

#             best_model = models[best_model_name]

#             print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
#             print('\n====================================================================================\n')
#             logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
#             logging.info(f'Best Model : {best_model}')

#             save_object(
#                 file_path = self.model_trainer_config.trained_model_file_path,
#                 obj = best_model
#             )

#         except Exception as e:
#             logging.info('Exception occured at Model Training')
#             raise CustomException(e,sys)

# #23/02/2025
# import os, sys, numpy as np, pandas as pd
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from src.logger import logging
# from src.exception import CustomException
# from src.utils import save_object

# from dataclasses import dataclass

# @dataclass
# class ModelTrainerConfig:
#     model_directory = os.path.join('artifacts', 'models')

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()
#         # Create models directory if it doesn't exist
#         os.makedirs(self.model_trainer_config.model_directory, exist_ok=True)

#     def initiate_model_training(self, train_array, test_array):
#         try:
#             logging.info('Splitting Dependent and Independent variables from train and test data')
#             X_train, y_train, X_test, y_test = (
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )

#             models = {
#                 'LinearRegression': LinearRegression(),
#                 'Lasso': Lasso(),
#                 'Ridge': Ridge(),
#                 'Elasticnet': ElasticNet(),
#                 'GradientBoost': GradientBoostingRegressor(),
#                 'RandomForestRegressor': RandomForestRegressor(),
#             }

#             # Train and save each model separately
#             for model_name, model in models.items():
#                 try:
#                     logging.info(f'Training {model_name}')
#                     model.fit(X_train, y_train)
                    
#                     # Create file path for each model
#                     model_file_path = os.path.join(
#                         self.model_trainer_config.model_directory,
#                         f'{model_name}.pkl'
#                     )
                    
#                     # Save the model
#                     save_object(
#                         file_path=model_file_path,
#                         obj=model
#                     )
                    
#                     logging.info(f'Successfully saved {model_name} at {model_file_path}')
#                     print(f'Saved {model_name} model')
                    
#                 except Exception as e:
#                     logging.error(f'Error while training {model_name}: {str(e)}')
#                     print(f'Error training {model_name}: {str(e)}')
#                     continue

#             print('\n====================================================================================\n')
#             logging.info('Model training and saving completed')

#         except Exception as e:
#             logging.info('Exception occurred at Model Training')
#             raise CustomException(e, sys)

# ===================================== Sufiyan ============================================================



import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class ModelTrainerConfig:
    def __init__(self):
        self.model_directory = os.path.join('artifacts', 'models')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.model_directory, exist_ok=True)

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'GradientBoost': GradientBoostingRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0)  # Added CatBoost
            }

            for model_name, model in models.items():
                try:
                    logging.info(f'Training {model_name}')
                    model.fit(X_train, y_train)

                    model_file_path = os.path.join(
                        self.model_trainer_config.model_directory,
                        f'{model_name}.pkl'
                    )

                    save_object(
                        file_path=model_file_path,
                        obj=model
                    )

                    logging.info(f'Successfully saved {model_name} at {model_file_path}')
                    print(f'Saved {model_name} model')

                except Exception as e:
                    logging.error(f'Error while training {model_name}: {str(e)}')
                    print(f'Error training {model_name}: {str(e)}')
                    continue

            print('\n====================================================================================\n')
            logging.info('Model training and saving completed')

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)