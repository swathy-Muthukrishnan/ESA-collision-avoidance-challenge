import os
import pandas as pd
from data_preprocessor import DataPrepocessor
from ML_dataset_creation import CreateDataset
from ML_Model import Risk_Forecasting_model
from data_utils import DataUtils
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, mean_squared_error
from typing import List, Tuple

# Threshold for the risks expressed in logarithmic scale
THRESHOLD = -6


class PredictRisk:

    """
    A class for predicting risk based on input data and evaluating the performance of prediction models.

    Attributes:
    pre_process (DataPrepocessor): An instance of DataPrepocessor for data preprocessing.
    create_dataset (CreateDataset): An instance of CreateDataset for dataset creation.
    MLmodel (Risk_Forecasting_model): An instance of Risk_Forecasting_model for model training.
    processed_data_fpath (str): File path for the processed dataset.
    n_CDMs (int): Number of latest Conjnction Data Messages (CDMs) to consider.
    """


    def __init__(self, latest_CDM = True, n_CDMs = 2):

        """
        Initializes the PredictRisk class.

        Args:
        latest_CDM (bool): Flag indicating whether to use the latest CDM or not.
        n_CDMs (int): Number of latest CDMs to consider default set to 2 

        Returns:
        None
        """
        self.pre_process =  DataPrepocessor(retain_all_features= True)
        self.create_dataset = CreateDataset(latest_CDM, n_CDMs)
        self.MLmodel = Risk_Forecasting_model()
        self.processed_data_fpath = os.path.join(os.getcwd(), 'processed_dataset.feather')
        self.n_CDMs = n_CDMs
        self.run_pipe_line()
        
    def preprocess_data(self) -> pd.DataFrame:

        """
        Preprocesses the data.

        Reads the dataset, converts string values into numerical values, assigns risk category for each CDM,
        and handles missing values.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.

        Information:
        - Step (1): Reads the data using the `get_dataset` method from `pre_process`.
        - Step (2): Converts string values into numerical values using the `process_string` method from `DataUtils`.
        - Step (3): Assigns risk category for each CDM using the `assign_risk_category` method from `pre_process`.
                    Future risk category refers to the risk category assigned to the ground truth (Final risk prior to 2 days of closest approach).
        - Step (4): Removes features with missing values that are neither used in classification nor in regression
                    using the `handle_missing_values` method from `pre_process`.

        Notes: 

        (1) Exploratory Data Analysis was conducted by visulally analysing the distributions of all the features and based on this the following 
        features will be used for classifcation and regression. 

        - classification_features_red = ['max_risk_estimate', 'risk', 'risk_range' , 'mahalanobis_distance' , 'miss_distance', 'c_time_lastob_start']

        - regression_features = ['risk', 'risk_range','max_risk_estimate', 'max_risk_scaling', 'mahalanobis_distance', 'miss_distance',
                    'relative_position_n', 'relative_position_r', 'relative_position_t', 't_j2k_inc', 'c_time_lastob_start', 'risk_category']

        (2) Since `c_time_lastob_start` is an important feature for identifying High and Low risks, only the missing values in this feature is imputed with -1. 
            c_time_lastob_startis a positive value attribute and -1 is chosen instead of 0 as a measure to not introduce any misleading trend in the data 
        
        (3) `risk_range`: These are calulated based on individual CDM risk 

            `risk_range` = 0 for risk values greater than -6
            `risk_range` = 1 for risk values less than or equal to -6 
        
        """

        

        print('\npre-processing the data.........')

        data_frame = self.pre_process.get_dataset() # step (1) 

        data_frame = DataUtils.process_string(data_frame) # step (2) 

        data_frame = self.pre_process.assign_risk_category(data_frame) # step (3) 
        
        data_frame = self.pre_process.handle_missing_values(data_frame)   # step (4)       
    
        return data_frame
    
    def get_train_test_dataset(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        
        """
        Obtains training and testing datasets for classification and regression tasks.

        Args:
        data_frame (pd.DataFrame): The input DataFrame containing the dataset.

        Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: 
        A tuple containing training and testing datasets for classification and regression tasks.
        The structure is ((train_x_c, train_y_c), (train_x_r, train_y_r)), ((test_x_c, test_y_c), (test_x_r, test_y_r)).

        Information:
        (1)  `calssification_dataset` and `regression_dataset` are obtained by calling the `get_train_test_dataset` method from `create_dataset`.
        
        (2)  For classification:

        - `train_x_c` and `test_x_c` represent the features for classification training and testing datasets, respectively.
        - `train_y_c` and `test_y_c` represent the labels for classification training and testing datasets, respectively.
        
        (3) For regression:
        - `train_x_r` and `test_x_r` represent the features for regression training and testing datasets, respectively.
        - `train_y_r` and `test_y_r` represent the target values for regression training and testing datasets, respectively.
        (4) The returned values are structured as follows:

        ((train_x_c, train_y_c), (train_x_r, train_y_r)), ((test_x_c, test_y_c), (test_x_r, test_y_r)).

        Notes: 

        (1) Logic for Splitting the History of CDMs into Input and Output Sequence: 
            - step (1): Identify ground truth `time_to_tca`
                        - ground truth `time_to_tca` is the time to Closest Approach associated with the final risk (ground truth)
                        - This will only be used for dataset creation and is not a part of the Output sequence 
                        - The latest CDM recived exaclty two days before Closest Approach  or on the 2nd Day of Closest Approach is the gorund truth ground truth `time_to_tca`
                           condition for grougnd truth `time_to_tca` is (`time_to_tca` = 2 or `time_to_tca` = Min(`time_to_tca` > 2 ) )
            - Step (2): Remove events with short History
                        - Events that don't have history i.e CDMs before the ground truth are discarded
                        
            - step (3): Determining the Outptut seq
                    - The the CDM(s) where `time_to_tca` is less than ground truth `time_to_tca` is discarded
                    - The CDM that has the ground truth `time_to_tca` is the output sequence 
                       
            - step (2) 
            Input Seq: After determining the Output Seq, the CDMs preceding it are the input sequences
                        Condition is  `time_to_tca`  should be less than ground truth `time_to_tca` 

        (2) Input Sequence: 
            The input sequecnce will include the past risk value along with the selected features 

        (3) `risk_category`: These are calulated based on the `risk_range` associated with ground truth

            eavents are classified into High and Low risk based on `risk_category`
            `risk_category` = 0 for high risk events (risk > -6)
            `risk_category` = 1 for low risk events (risk <= -6)

        """
        calssification_dataset, regression_dataset = self.create_dataset.get_train_test_dataset(data_frame)
        train_x_c, train_y_c, test_x_c, test_y_c = calssification_dataset
        train_x_r, train_y_r, test_x_r, test_y_r = regression_dataset

        train_dataset = (train_x_c, train_y_c), (train_x_r, train_y_r)
        test_dataset = (test_x_c, test_y_c), (test_x_r, test_y_r)
        
        
        return train_dataset, test_dataset 

    def train_model(self, train_data) -> Tuple:

        """
        Trains risk classifier and predictor models.

        Args:
        train_data (Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]):
        A tuple containing training datasets for classification and regression tasks.
        The structure is ((train_x_c, train_y_c), (train_x_r, train_y_r)).

        Returns:
        Tuple[RandomForestClassifier, RandomForestRegressor]: A tuple containing trained classifier and predictor models.

        Information:
            - `calssification_train_dataset` and `regression_train_dataset` are obtained from `train_data`.
            - `ML_model` is an instance of `Risk_Forecasting_model` class 
            - `train_x_c` and `train_y_c` represent the features and labels for classification training dataset, respectively.
            - `train_x_r` and `train_y_r` represent the features and target values for regression training dataset, respectively.
            - `ML_classifier` is trained using `train_risk_classifier`(RandomForestClassifier) method from `ML_model`.
            - `ML_regressor` is trained using `train_risk_predictor` (RandomForestRegressor) method from `ML_model`.

        Notes:
            
            (1) Input Sequence: 

                The input to the model will be the latest CDM(s) recieved 
            (2) Output Sequence: 

            (3 The Classifier first predicts the `risk_category` of the incoming classification fetures of the Input sequence
            (4) The predicted `risk_category` is concatinated with the regression features of the input sequence and then give to the risk predictor i.e regressor model 
            (5) The performance of the risk predictor is heavily dependent on the classifier model 
            
            
        """
        calssification_train_dataset = train_data[0]
        regression_train_dataset = train_data[1]

        ML_model = Risk_Forecasting_model()
        train_x_c, train_y_c = calssification_train_dataset
        train_x_r, train_y_r = regression_train_dataset
        
        ML_classifier = ML_model.train_risk_classifier(train_x_c, train_y_c )
        ML_regressor = ML_model.train_risk_predictor(train_x_r, train_y_r)

        return ML_classifier, ML_regressor

    
    def predict_final_risk(self, trained_classifier, trained_predictor, test_data) -> Tuple:

        """
        Predicts final risk categories and risk values.

        Args:
        trained_classifier (RandomForestClassifier): The trained risk classifier model.
        trained_predictor (RandomForestRegressor): The trained risk predictor model.
        test_data (Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]):
        A tuple containing testing datasets for classification and regression tasks.
        The structure is ((test_x_c, _), (test_x_r, _)).

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing predicted risk categories and risk values.

        Information:

        - `calssification_test_dataset` and `regression_test_dataset` are obtained from `test_data`.
        - `test_x_c` and `test_x_r` represent the features for classification and regression testing datasets, respectively.
        - `predicted_risk_categories` is obtained by testing the classifier model using `test_risk_classifier` method from `MLmodel`.
        - `upated_test_x_r` is updated using `Update_regression_test_data` method from `pre_process`.
        - `predicted_risk` is obtained by testing the predictor model using `test_risk_predictor` method from `MLmodel`.
        
        Notes: 
            - The X_test from the test set for regression is updated with the classification resulst.  

        """
        calssification_test_dataset = test_data[0]
        regression_test_dataset = test_data[1]
        
        test_x_c, _ = calssification_test_dataset
        test_x_r, _ = regression_test_dataset

        predicted_risk_categories = self.MLmodel.test_risk_classifier(test_x_c, trained_classifier)

        upated_test_x_r = self.pre_process.Update_regression_test_data(test_x_r, predicted_risk_categories, self.n_CDMs)

        predicted_risk = self.MLmodel.test_risk_predictor(upated_test_x_r, trained_predictor)

        return predicted_risk_categories, predicted_risk

    def evaluate_classifier_performance(self, ground_truth, predictions) -> str:

        """
        Evaluates the performance of the classifier.

        Args:
        ground_truth (Iterable): Ground truth labels.
        predictions (Iterable): Predicted labels.

        Returns:
        str: A classification report containing precision, recall, F1-score, and support.

        Notes:
        - `report` is generated using the `classification_report` function.
        """
        report = classification_report(ground_truth, predictions)

        return report


    def evaluate_final_risk(self, gt_risk, pred_risk) -> float:
        """
        Evaluates the final risk prediction performance.

        Args:
        gt_risk (np.ndarray): Ground truth risk values.
        pred_risk (np.ndarray): Predicted risk values.
        THRESHOLD (float): Threshold for risk classification.

        Returns:
        float: The ratio of mean squared error to F-beta score (with beta=2).

        Notes:
        - `gt_mask` and `pred_mask` are created based on the threshold value.
        - `mse` is calculated using the `mean_squared_error` function.
        - `f2` is calculated using the `fbeta_score` function with beta=2.
        - The ratio of `mse` to `f2` is returned as the evaluation metric.
        """
        gt_mask = gt_risk >= THRESHOLD
        pred_mask = pred_risk >= THRESHOLD
        mse = mean_squared_error(gt_risk[gt_mask], pred_risk[gt_mask])
        f2 = fbeta_score(gt_mask, pred_mask, beta=2)

        return mse / f2
    
    def run_pipe_line(self):
        """
        Evaluates the entire risk prediction model.

        Returns:
        None

        Notes:
        - The data is preprocessed using the `preprocess_data` method.
        - Training and testing datasets are obtained using the `get_train_test_dataset` method.
        - Classifier and regressor models are trained using the `train_model` method.
        - Classifier and regression test datasets are extracted from the testing data.
        - Predictions for risk category and final risk are obtained using the `predict_final_risk` method.
        - Predictor evaluation is performed using the `evaluate_final_risk` method.
        - Classifier performance is evaluated using the `evaluate_classifier_performance` method.
        - Evaluation scores and classification reports are printed.
        """
            
        dataframe = self.preprocess_data()
        train_data, test_data  = self.get_train_test_dataset(dataframe)
        classifier, regressor = self.train_model(train_data)

        classiifer_test_dataset = test_data[0]
        regression_test_dataset = test_data[1]
        _, y_test_c = classiifer_test_dataset
        _, y_test_r = regression_test_dataset

        predicted_risk_category, predicted_final_risk = self.predict_final_risk(classifier, regressor, test_data)
        
        predictor_evaluation = self.evaluate_final_risk(predicted_final_risk, y_test_r)

        report = self.evaluate_classifier_performance(y_test_c, predicted_risk_category)

        print(f'\n Model Evaluation:\n Evaluation score: {predictor_evaluation} (Predictor Performance)\n')

        print(f"\n Calssification report:\n{report}")

def main():

    PredictRisk()


if __name__ == "__main__":
    main()



