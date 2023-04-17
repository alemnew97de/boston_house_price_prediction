import os
import sys
import pandas as pd 
import numpy as np 
from  sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.compoments.data_transformation import DataTransformation

## initalizing The  data Ingestion Configeration

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifcats","train.csv")
    test_data_path:str = os.path.join("artifcats","test.csv")
    raw_data_path:str = os.path.join("artifcats","raw.csv")


## Creating a Class For Data INgestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    ## Data Ingestion MEthod
    def initiate_data_ingetion(self):
        logging.info("Data Ingestion Method Started")
        try:
            data = pd.read_csv(os.path.join("notebook/data","boston.csv"))
            logging.info("Data Reading As Pandas DataFrame")


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Train Test Split")
            train_set,test_set = train_test_split(data,test_size=0.33,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header=True)

            logging.info("Data Ingestion Complited")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception Occure in Data Ingestion Stage")
            raise CustomException(e, sys)



# Run Data INgetion

#if __name__=="__main__":
 #   obj = DataIngestion()
 #   train_data_path,test_data_path = obj.initiate_data_ingetion()
  #  data_transformation = DataTransformation()
  #  train_arr,test_arr,_ = data_transformation.initatie_data_transformation(train_data_path, test_data_path)