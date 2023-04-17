import os
import sys
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from src.compoments.data_ingestion import DataIngestion
from src.compoments.data_transformation import DataTransformation
from src.compoments.model_traning import ModelTraning




if __name__=="__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingetion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initatie_data_transformation(train_data_path, test_data_path)
    model_traning = ModelTraning()
    model_traning.initate_model_traning(train_arr, test_arr)
    