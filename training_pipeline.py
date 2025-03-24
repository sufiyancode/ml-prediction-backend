import os
import sys
from src.logger import logging
# from exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from model_trainer import ModelTrainer
from src.exception import CustomException
if __name__ == '__main__':
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initaite_data_transformation(train_data_path, test_data_path)
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
    except Exception as e:
        logging.error(f'Error in training pipeline: {str(e)}')
        raise CustomException(e, sys)
    
# import os
# import sys
# from src.logger import logging
# # from exception import CustomException
# import pandas as pd
# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from model_trainer import ModelTrainer
# from src.exception import CustomException
# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
#     model_trainer=ModelTrainer()
#     model_trainer.initiate_model_training(train_arr,test_arr)