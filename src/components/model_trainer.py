import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.Exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModeltrainerConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modeltrainerConfig = ModeltrainerConfig()
    
    def Model_trainer(self,train_data,test_data):
        try:
            logging.info("Split training and test data")
            X_train,y_train,X_test,y_test = (
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1],
            )
            models = {
                    'LinearRegression': LinearRegression(),
                    "Gradient Boosting":GradientBoostingRegressor(),
                    "XGRegressor": XGBRegressor(),
                    'KNeighbors Regressor':KNeighborsRegressor(),
                    'DecisionTree Regressor':DecisionTreeRegressor(),
                    'RandomForest Regressor':RandomForestRegressor(),
                    'CatBoost Regressor': CatBoostRegressor(verbose=False),
                    'AdaBoost Regressor': AdaBoostRegressor()
                }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            ## to get best model score
            best_model_score = max(sorted(model_report.values()))

            ## to get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)
                                                        ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and test dataset")

            save_object(
                file_path =  self.modeltrainerConfig.preprocessor_obj_file_path,  # to save pickle path
                obj = best_model,  # to save best model
            )
            predicted = best_model.predict(X_test)

            r2_Square = r2_score(y_test,predicted)
            return r2_Square
            
        except Exception as e:
            raise CustomException(e,sys)


