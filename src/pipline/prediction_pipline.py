import os
import sys
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass



    def predict(self,features):
        try:
            ## This line of path code work i both windos and linex
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            ## lode the object
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            

            pred = model.predict(data_scaled)

            return pred
            

        except Exception as e:
            logging.info("Error Occured in Prediction Pipline")
            raise CustomException(e, sys)



## Create Custom Data Class
class CustomData:
    def __init__(self,
                CRIM:float,
                ZN:float,
                INDUS:float,
                CHAS:int,
                NOX:float,
                RM:float,
                AGE:float,
                DIS:float,
                RAD:int,
                TAX:float,
                PTRATIO:float,
                B:float,
                LSTAT:float):


        self.CRIM = CRIM
        self.ZN = ZN
        self.INDUS = INDUS
        self.CHAS = CHAS
        self.NOX = NOX
        self.RM = RM
        self.AGE = AGE
        self.DIS = DIS
        self.RAD = RAD
        self.TAX = TAX
        self.PTRATIO = PTRATIO
        self.B = B
        self.LSTAT = LSTAT


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CRIM":[self.CRIM],
                "ZN":[self.ZN],
                "INDUS":[self.INDUS],
                "CHAS":[self.CHAS],
                "NOX":[self.NOX],
                "RM":[self.RM],
                "AGE":[self.AGE],
                "DIS":[self.DIS],
                "RAD":[self.RAD],
                "TAX":[self.TAX],
                "PTRATIO":[self.PTRATIO],
                "B":[self.B],
                "LSTAT":[self.LSTAT]
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Ocured in Prediction Pipline")
            raise CustomException(e, sys)







  