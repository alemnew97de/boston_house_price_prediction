import os
import sys
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from src.utils import model_evalution


@dataclass
class ModelTraningConfig:
    train_model_file_path = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_trainer_config =ModelTraningConfig()


    def initate_model_traning(self,train_array,test_array):
        try:
            logging.info("Spliting Dependent and Indipendent Features in Train Test data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            ## Multiple Model Traning
            models = {
            "LinearRegression":LinearRegression(),
            "Ridge":Ridge(),
            "Lasso":Lasso(),
            "ElasticNet":ElasticNet()
        }

            model_report:dict = model_evalution(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n***************************************************************\n")
            logging.info(f"Model Report: {model_report}")

            ## To get Best Model Score from Dict
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")
            print("\n********************************************************************************\n")
            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")


            save_object(file_path=self.model_trainer_config.train_model_file_path,
             obj=best_model
             )


        except Exception as e:
            logging.info("Error occure in model Traning")
            raise CustomException(e,sys)