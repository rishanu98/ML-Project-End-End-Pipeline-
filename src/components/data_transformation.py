import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # Responsible for handling missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

from src.Exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transfomation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['writing score','reading score']
            categorical_column = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "test preparation course",
            ]

            num_pipeline = Pipeline(
            steps = [
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))
            ]

        )

            cat_pipeline = Pipeline(
            steps= [
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("oen_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False)),
                 
            ]
        ) 
            logging.info(f"Categorical columns: {categorical_column}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipeline",cat_pipeline,categorical_column)
            ]
        )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def intiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtained preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column = "math score"
            numerical_columns = ["writing score","reading score"]
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info(f"Saving preprocessing objects")

            save_object(
                file_path =  self.data_transfomation_config.preprocessor_obj_file_path,  # to save pickle path
                obj = preprocessing_obj,  # to save obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transfomation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)



            

   