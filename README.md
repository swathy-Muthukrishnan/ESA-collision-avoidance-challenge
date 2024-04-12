# ESA-collision-avoidance-challenge
A two-stage ML model for improved satellite collision avoidance, utilizing a classifier to categorize debris risk (low/high) and a regressor to refine future collision risk predictions based on historical data and conjunction messages (CDMs) (Multivariate time series data)


### README

---
# PredictRisk - Risk Prediction and Evaluation

PredictRisk is a Python class designed for predicting the final risk pior to 2 days of closest approach, This README provides an overview of the class and its functionalities.

## Class Overview

The PredictRisk class consists of several methods for data preprocessing, dataset creation, model training, prediction, and evaluation. Here's a brief overview of each method:

- `__init__(self, latest_CDM=True, n_CDMs=2)`: Initializes the PredictRisk class with options to specify the number of latest Conjunction Data Messages (CDMs) and whether to use the latest CDM.

- `preprocess_data(self)`: Preprocesses the input data, including converting string values into numerical values, assigning risk categories, and handling missing values.

- `get_train_test_dataset(self, data_frame)`: Obtains training and testing datasets for classification and regression tasks.

- `train_model(self, train_data)`: Trains risk classifier and predictor models.

- `predict_final_risk(self, trained_classifier, trained_predictor, test_data)`: Predicts final risk categories and risk values.

- `evaluate_classifier_performance(self, ground_truth, predictions)`: Evaluates the performance of the classifier.

- `evaluate_final_risk(self, gt_risk, pred_risk)`: Evaluates the final risk prediction performance.

- `run_pipe_line(self)`: Executes the entire risk prediction model, including data preprocessing, model training, prediction, and evaluation.

## Folder Structure

The folder structure for this project is as follows:

Risk_prediction/
│
├── data_preprocessor.py
├── ML_dataset_creation.py
├── ML_Model.py
├── data_utils.py
├── predict_risk.py
├── requirements.txt
└── README.md
└── Dataset
	└──train_data.csv


### Dependencies

- pandas
- scikit-learn
- numpy


## Usage

To use the PredictRisk class, follow these steps:

1. Import the necessary modules and classes:

```python
from data_preprocessor import DataPrepocessor
from ML_dataset_creation import CreateDataset
from ML_Model import Risk_Forecasting_model
from data_utils import DataUtils

# Two ways to run the predict_risk.py script  

(1) predict_risk = PredictRisk()
and Run the pipeline to preprocess data, train models, make predictions, and evaluate performance:
predict_risk.run_pipe_line()

or 

(2) Run the following command in the terminal 
python predict_risk.py 

