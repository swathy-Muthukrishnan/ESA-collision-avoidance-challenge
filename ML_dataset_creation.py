import pandas as pd 
import os
import numpy as np
from data_utils import DataUtils
from typing import List, Tuple
from data_viz_utils import DataVizUtils
class CreateDataset:
    
    def __init__(self, latest_cdm = False, n_latest_cdms = 5):
        """
        Initializes an instance of the class.

        Args:
        - latest_cdm (bool): Indicates whether to consider only the latest CDM. Defaults to False.
        - n_latest_cdms (int): Number of latest CDMs to consider if latest_cdm is True. Defaults to 5.

        Returns:
        - None
        """
        self.latest_cdm  = latest_cdm
        self.n_latest_cdms = n_latest_cdms 
    
    def split_input_output_seq(self, data_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Splits the input DataFrame into input and output sequences.

        Args:
        - data_frame (pd.DataFrame): The input DataFrame.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple containing input and output sequences as NumPy arrays.

        Notes:
        - If self.latest_cdm is True, only the latest N CDMs are considered for each event.
        - The method filters out events with only one CDM. This is to avoid creation of Output sequences without input seq
        - Output sequence is extracted using the last row of each event.
        - Evenet that have TCA lesser than ground tuth TCA are dropped from the input DataFrame.
        - The input and output sequences are generated and returned as lists of arrays.
        """
        if self.latest_cdm: 
            grouped_data = data_frame.groupby('event_id').tail(self.n_latest_cdms)

            data_frame = pd.DataFrame(grouped_data)

            grouped = data_frame.groupby('event_id').size()

            # Filter out the groups with length 1
            groups_with_length_1 = grouped[grouped == 1]

            # Get the group ids with length 1
            group_ids_with_length_1 = groups_with_length_1.index.tolist()
            
            data_frame = data_frame[~data_frame['event_id'].isin(group_ids_with_length_1)]
        
        output_sequence = self.last_row_by_event_id(data_frame)[DataUtils.output_features]
        data_frame = self.drop_least_tca(data_frame)

        input_list = data_frame.groupby('event_id').apply(lambda g: g.values).tolist()
        ouput_list = output_sequence.groupby('event_id').apply(lambda g: g.values).tolist()

        return input_list, ouput_list

    def drop_least_tca(self, data_frame):
        """
        Drops rows with the least 'time_to_tca' within each event group.

        Args:
        - data_frame (pd.DataFrame): The input DataFrame containing event data.

        Returns:
        - pd.DataFrame: The DataFrame with rows having the least 'time_to_tca' dropped for each event.

        Notes:
        - The method groups the DataFrame by 'event_id'.
        - For each group, it finds the index of the row with the least 'time_to_tca'.
        - It then drops the rows with these indices.
        """
        grouped_data = data_frame.groupby('event_id')

        # Finding the index of the row with the least 'time_to_tca' within each group
        min_indices = grouped_data['time_to_tca'].idxmin()

        # Dropping the rows with the least 'time_to_tca' within each group
        filtered_data_frame = data_frame.drop(min_indices)

        return filtered_data_frame

    def last_row_by_event_id(self, data_frame):

        """
        Selects the last row for each event group based on the minimum 'time_to_tca'.

        Args:
        - data_frame (pd.DataFrame): The input DataFrame containing event data.

        Returns:
        - pd.DataFrame: The DataFrame with the last row selected for each event group.

        Notes:
        - The method groups the DataFrame by 'event_id'.
        - For each group, it selects the row with the minimum 'time_to_tca'.
        - It returns the DataFrame with these selected rows.
        """
        # Grouping by 'event_id' and selecting the last row in each group
        last_rows = data_frame.loc[data_frame.groupby('event_id')['time_to_tca'].idxmin()]
        return last_rows
    

    def custom_train_test_split(self, data_frame, train_size=0.85, validation_size=0.05, test_size=0.15, random_state=42):

        """
        Custom train-test split function for event data.

        Args:
        - data_frame (pd.DataFrame): The input DataFrame containing event data.
        - train_size (float): The proportion of the dataset to include in the training set.
        - validation_size (float): The proportion of the dataset to include in the validation set.
        - test_size (float): The proportion of the dataset to include in the test set.
        - random_state (int, optional): Random state for reproducibility.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the train, test, and validation DataFrames.

        Notes:
        - The method groups the DataFrame by 'risk_category' and collects unique event IDs in each category.
        - The prorportions are maintained in such a way that the train, test splie is also reflected in the risk category 
        - By default Train data will have  80% of total data, this includes 80% of total High risk events and 80% of total Low risk events 
        - It calculates split proportions for the validation and test sets within the remaining 20%. (20% of total High risk events and 20% of total Low risk events)
        - It performs the split and returns train, test, and validation DataFrames.
        """
        # Group by risk_category and collect unique event IDs in each category
        grouped = data_frame.groupby('risk_category')['event_id'].unique()
        # Calculate split proportions for validation and test sets within the remaining 20%
        val_prop_within_20 = validation_size / (validation_size + test_size)
        test_prop_within_20 = 1 - val_prop_within_20

        #grouped = data_frame.groupby('risk_category')
        grouped = grouped.to_frame()

        #grouped['train_data'], grouped['test_data'] = zip(*grouped['data'].apply(lambda x: split_array(x, test_size=0.2)))
        train_test_events = grouped['event_id'].apply(lambda g: DataUtils.split_array(g, test_size=test_size))
        val_test_events = train_test_events.apply(lambda g: DataUtils.split_array(g[1], test_size=val_prop_within_20))

        train_events = np.hstack(train_test_events.apply(lambda g: g[0]).to_numpy())
        test_events = np.hstack(val_test_events.apply(lambda g: g[0]).to_numpy())
        val_events = np.hstack(val_test_events.apply(lambda g: g[1]).to_numpy())
        
        train_data_frame  = data_frame[data_frame['event_id'].isin(train_events)]
        test_data_frame  = data_frame[data_frame['event_id'].isin(test_events)]
        val_data_frame  = data_frame[data_frame['event_id'].isin(val_events)]

        return train_data_frame, test_data_frame, val_data_frame

    def get_train_test_dataset(self, data_frame):
        
        """
        Get train-test datasets for classification and regression.

        Args:
        - data_frame (pd.DataFrame): The input DataFrame containing event data.

        Returns:
        - Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: 
        A tuple containing classification and regression datasets.

        Notes:
        - The method splits the data into training and testing sets using the custom train-test split function.
        - It concatenates the test and validation datasets into the test set.
        - It splits the input-output sequences for both classification and regression.
        - It prepares classification and regression datasets separately and returns them.
        
        Structure of returned data:
        - classification_dataset: Tuple containing arrays for training and testing classification data:
        - train_x_c: Training input sequences for classification.
        - train_y_c: Training output labels for classification.
        - test_x_c: Testing input sequences for classification.
        - test_y_c: Testing output labels for classification.
        
        - regression_dataset: Tuple containing arrays for training and testing regression data:
        - train_x_r: Training input sequences for regression.
        - train_y_r: Training output values for regression.
        - test_x_r: Testing input sequences for regression.
        - test_y_r: Testing output values for regression.
        """
        train_data, test_data, val_data =  self.custom_train_test_split(data_frame)
        test_data = pd.concat([test_data, val_data])

        calssification_features, regression_features = DataUtils.get_feature_index()

        calssification_output = DataUtils.output_features.index('risk_category')
        regression_output = DataUtils.output_features.index('risk')
        
        train_x, train_y = self.split_input_output_seq(train_data)
        test_x, test_y = self.split_input_output_seq(test_data)

        train_x_c = np.array([DataUtils.pad_matrix(train[:, calssification_features], self.n_latest_cdms-1).ravel() for train in train_x])
        train_y_c = np.array([train[:,calssification_output].ravel()  for train in train_y]).reshape(-1).astype(int)
        test_x_c = np.array([DataUtils.pad_matrix(test[:, calssification_features], self.n_latest_cdms-1).ravel() for test in test_x])
        test_y_c = np.array([test[:,calssification_output].ravel() for test in test_y]).reshape(-1).astype(int)
        
        train_x_r = np.array([DataUtils.pad_matrix(train[:, regression_features], self.n_latest_cdms-1).ravel() for train in train_x])
        train_y_r = np.array([train[:,regression_output].ravel()  for train in train_y]).reshape(-1).astype(int)
        test_x_r = np.array([DataUtils.pad_matrix(test[:, regression_features], self.n_latest_cdms-1).ravel() for test in test_x])
        test_y_r = np.array([test[:,regression_output].ravel() for test in test_y]).reshape(-1).astype(int)

        classifcation_dataset = (train_x_c, train_y_c, test_x_c, test_y_c)
        regression_dataset = (train_x_r, train_y_r, test_x_r, test_y_r)

        return classifcation_dataset, regression_dataset
