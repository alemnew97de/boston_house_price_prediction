import os
import sys
import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer #handel missing values
#Pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprosser_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Initiated")

            numerical_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
            'PTRATIO', 'B', 'LSTAT']

            logging.info("Pipline Initiated")

            # Numerical Pipline

            num_pipline = Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
        ]
    )

            # Catigorical Pipline
            cato_pipline = Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("scaler",StandardScaler())
        ]
    )

            # Columns Transformer
            preprocessor = ColumnTransformer([
            ("num_pipline",num_pipline,numerical_features),
            ])

            return preprocessor

            logging.info("Pipline Complited")


        except Exception as e:
            logging.info("Error Occured in Data Transformation")
            raise CustomException(e,sys)


    
    def initatie_data_transformation(self,train_path,test_path):
        try:
            ## Read Train and Test Data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning And Test Data Complited")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            logging.info("Ontaning preprosser object")

            prprocessor_obj = self.get_data_transformation_obj()

            target_colum_name = "MEDV"
            drop_columns = [target_colum_name]

            # this is line x & y
            input_feature_train_data = train_data.drop(drop_columns,axis=1)
            target_feature_train_data = train_data[target_colum_name]

            input_feature_test_data = test_data.drop(drop_columns,axis=1)
            target_feature_test_data = test_data[target_colum_name]


            ## Apply Transformation Using Preprocessor Object xtrain and xyest
            input_feature_train_arr = prprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = prprocessor_obj.transform(input_feature_test_data)

            logging.info("Applying Preprossing obj on Train and test data")

            ## Convert into Array To be fast concat
            train_array  = np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]

            ## Calling Save object function and save preprosser pkl
            save_object(file_path=self.data_transformation_config.preprosser_obj_file_path, obj=prprocessor_obj)

            logging.info("Saving Preprocessor Pikel File")


            return (
                train_array,
                test_array,
                self.data_transformation_config.preprosser_obj_file_path
            )

            
        except Exception as e:
            logging.info("Error Occured in the initaie data transformation")
            raise CustomException(e,sys)

    













