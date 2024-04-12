import pandas as pd
import os
from data_utils import DataUtils
import numpy as np 
from data_viz_utils import DataVizUtils
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

class DataPrepocessor: 
    
    """
    A class for preprocessing event data.

    This class provides methods for reading datasets, removing events with insufficient CDM history, 
    assigning risk categories to events, handling missing values, and updating regression test data with forecasted risk categories.

    Attributes:
    - data_file_dir (str): The directory containing the dataset files.
    - dataset (str): The path to the dataset file.
    - retain_all_features (bool): A flag indicating whether to retain all features or not.

    Methods:
    - get_dataset: Reads and returns the dataset.
    - remove_shorter_events: Removes events with insufficient CDM history.
    - assign_risk_category: Assigns risk categories to each event in the dataset.
    - handle_missing_values: Handles missing values in the dataset.
    - assign_risk_change_rate: Assigns risk change rate to each event in the dataset.
    - Update_regression_test_data: Updates regression test data with forecasted risk categories.

    """

    def __init__(self, retain_all_features = False, dataset_dir = 'Dataset', dataset_file_name = 'train_data.csv' ) -> None:
        
        self.data_file_dir = os.path.join(os.getcwd(), dataset_dir)
        self.dataset = os.path.join(self.data_file_dir, dataset_file_name)
        self.retain_all_features = retain_all_features

    def get_dataset(self) -> pd.DataFrame:

        """
        Reads and retuns the dataset

        Returns:
            pd.DataFrame: DataFrame containing the dataset.
        """
        data_frame = pd.read_csv(self.dataset)
        return data_frame
    
    def remove_shorter_events(self, data_frame: pd.DataFrame) -> pd.DataFrame:

        """
        Remove events with insufficient CDM history prior to the closest approach.

        Parameters:
            data_frame (pd.DataFrame): DataFrame containing event data.

        Returns:
            pd.DataFrame: DataFrame with shorter events removed.

        Note: This removal of event's that don't have enough CDMs 
        
        """

        grouped = data_frame.groupby('event_id')

        # Remove events that don't have CDMs avaiable two days piror to closest approach  
        valid_event_ids = grouped.filter(lambda group: (group['time_to_tca'] > 2).any() and (group['time_to_tca'] <= 2).any())['event_id'].unique()

        # Filter dataframe to include only rows that have CDM history for 2 or more than 2 days pior to closet approach
        filtered_data_frame = data_frame[data_frame['event_id'].isin(valid_event_ids)]

        events_in_filtered = filtered_data_frame['event_id'].unique()

        data_frame['short_event'] = 1

        # Set the value of 'short_event' column to 0 for events in 'events_in_filtered'
        data_frame.loc[data_frame['event_id'].isin(events_in_filtered), 'short_event'] = 0
        DataVizUtils.plot_pie_chart(data_frame, class_name = 'Class of CDM length ',grouping_feature = 'short_event', title =  '')


        return filtered_data_frame

    def assign_risk_category(self, data_frame: pd.DataFrame) -> pd.DataFrame:

        """
        Assigns risk categories to each event in the dataset.

        Args:
        data_frame (pd.DataFrame): The input DataFrame containing the dataset.

        Returns:
        pd.DataFrame: The DataFrame with risk categories assigned to each event.

        Notes:
        - The function splits the 'risk_range' column using the `split_risk_range` method from `DataUtils`.
        - It removes events with shorter histories using the `remove_shorter_events` method.
        - Initializes the 'risk_category' column with a default value of -1.
        - Finds the row indices with 'time_to_tca' equal to 2 or the next immediate least value lesser than 2 for each event.
        - Updates the 'risk_category' column based on the chosen rows.
        - Drops unnecessary rows and ensures alignment on 'event_id'.
        - Filters the rows to retain only those relevant for assigning risk categories.
        """
        
        data_frame = DataUtils.split_risk_range(data_frame)

        data_frame = self.remove_shorter_events(data_frame)

        data_frame['risk_category'] = -1

        # Step 2: Find the row indices with 'time_to_tca' equal to 2 or the next immediate least value lesser than 2 for each event

        chosen_indices = (
            data_frame[data_frame['time_to_tca'] >= 2].groupby('event_id')['time_to_tca']
            .idxmin()  # Find max index where time_to_tca <= 2
            .fillna(data_frame[data_frame['time_to_tca'] > 2].groupby('event_id')['time_to_tca'].idxmin())  # Fill NaNs with min index where time_to_tca < 2
        )

        #chosen_indices = chosen_indices.dropna().astype(int)  # Drop NaNs and convert to int

        # Step 3: Use the chosen indices to update 'risk_category' and drop unnecessary rows
        chosen_rows = data_frame.loc[chosen_indices]

        # Align chosen_rows on event_id to ensure the length matches
        chosen_rows = chosen_rows.set_index('event_id')
        data_frame = data_frame.set_index('event_id')

        # The risk range of the grounf truth will be used as risk category 
        # Update 'risk_category' based on chosen rows
        data_frame['risk_category'] = chosen_rows['risk_range']

        data_frame = data_frame.reset_index(level=0)
        grouped_data = data_frame.groupby('event_id')
        
        def filter_rows(group):
            Two_days_before_tca = group[group['time_to_tca'] > 2]
            return group[group['time_to_tca'] >= Two_days_before_tca['time_to_tca'].min()]

        filtered_data_frame = grouped_data.apply(filter_rows)


        return filtered_data_frame.reset_index(drop=True) 
    
    def handle_missing_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        """
        Handles missing values in the dataset.

        Args:
        dataframe (pd.DataFrame): The input DataFrame containing the dataset.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.

        Notes:
        - Missing values in the 'c_time_lastob_end' and 'c_time_lastob_start' columns are filled with -1.
        - Columns with any remaining missing values are dropped.
        - The processed DataFrame is saved to a feather file named 'processed_dataset.feather'.
        """

        dataframe['c_time_lastob_end'].fillna(-1, inplace=True)
        dataframe['c_time_lastob_start'].fillna(-1, inplace=True)
        dataframe = dataframe.dropna(axis = 1)
        dataframe.to_feather('processed_dataset.feather')
        return dataframe
    
    def assign_risk_change_rate(self, data_frame: pd.DataFrame) -> pd.DataFrame:

        """
        Assigns risk change rate to each event in the dataset.

        Args:
        data_frame (pd.DataFrame): The input DataFrame containing the dataset.

        Returns:
        pd.DataFrame: The DataFrame with risk change rate assigned to each event.

        Notes:
        - The risk change rate is calculated as the difference of risk divided by the difference of time to Closest Approach (TCA)
        for each event using the `diff` method on grouped 'risk' and 'time_to_tca' columns.
        - Missing values in the calculated risk change rate are filled with 0.
        """
        data_frame['risk_change_rate'] = data_frame.groupby('event_id')['risk'].diff() / data_frame.groupby('event_id')['time_to_tca'].diff()
        data_frame['risk_change_rate'].fillna(0, inplace=True)

        return data_frame 
    
    def Update_regression_test_data(self, test_x_r: np.ndarray, forecast_c: np.ndarray, n_cdm: int) -> np.ndarray:

        """
        Updates the regression test data with forecasted risk categories.

        Args:
        test_x_r (np.ndarray): The input test data for regression.
        forecast_c (np.ndarray): The forecasted risk categories.
        n_cdm (int): The number CDMs to consider.

        Returns:
        np.ndarray: The updated regression test data.

        Notes:
        - The method updates each test data instance in test_x_r with the forecasted risk category.
        """
        updated_test_x = []
        _, regression_features = DataUtils.get_feature_index()
        risk_category_idx = regression_features.index(DataUtils.feature_order.index('risk_category'))


        for i, test_x in enumerate(test_x_r):
            category_array = np.full( shape = (n_cdm-1, 1),fill_value= forecast_c[i])
            test_x = test_x.reshape(n_cdm-1, len(regression_features))
            test_x = np.delete(test_x, risk_category_idx, axis=1)
            test_x = np.concatenate((test_x, category_array), axis=1)
            updated_test_x.append(test_x.ravel())
        updated_test_x = np.array(updated_test_x)

        return updated_test_x



def main():

    preprocess = DataPrepocessor(False)
    data_frame = preprocess.get_dataset()
    data_frame = DataUtils.process_string(data_frame)
    filtered_df = preprocess.assign_risk_category(data_frame)
    filtered_df = preprocess.assign_risk_change_rate(filtered_df)
    filtered_df = preprocess.handle_missing_values(filtered_df)



if __name__ == "__main__":
    main()