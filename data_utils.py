import pandas as pd

import numpy as np 
class DataUtils: 

    """
    A utility class for data preprocessing and manipulation.

    Static Attributes:
    - columns_to_drop (list): Columns to be dropped from the DataFrame.
    - classification_features_red (list): Selected features for classification.
    - regression_features (list): Selected features for regression.
    - feature_order (list): Order of features in the DataFrame.
    - output_features (list): Output features of interest.

    Methods:
    - process_string(df: pd.DataFrame) -> pd.DataFrame: 
      Processes string columns in the DataFrame.
    - normalization(data_frame): 
      Normalizes selected features in the DataFrame.
    - split_risk_range(data_frame: pd.DataFrame) -> pd.DataFrame: 
      Splits the 'risk' column into risk ranges.
    - drop_least_tca(data_frame): 
      Drops rows with the least 'time_to_tca' within each event group.
    - custom_collate_fn(batch): 
      Custom collate function for PyTorch DataLoader.
    - pad_matrix(matrix, max_length): 
      Pads a matrix to a desired length.
    - get_feature_index(): 
      Retrieves feature indices for classification and regression.
    - split_array(data, test_size): 
      Splits the input array into training and testing sets.
    """

    columns_to_drop = ['mission_id']

    classification_features_red = ['max_risk_estimate', 'risk', 'risk_range' , 'mahalanobis_distance' , 'miss_distance', 'c_time_lastob_start']

    regression_features = ['risk', 'risk_range','max_risk_estimate', 'max_risk_scaling', 'mahalanobis_distance', 'miss_distance',
                    'relative_position_n', 'relative_position_r', 'relative_position_t', 't_j2k_inc', 'c_time_lastob_start', 'risk_category']
    

    feature_order =['event_id', 'time_to_tca', 'mission_id', 'risk', 'max_risk_estimate',
       'max_risk_scaling', 'miss_distance', 'relative_speed',
       'relative_position_r', 'relative_position_t', 'relative_position_n',
       'relative_velocity_r', 'relative_velocity_t', 'relative_velocity_n',
       't_time_lastob_start', 't_time_lastob_end', 't_recommended_od_span',
       't_actual_od_span', 't_obs_available', 't_obs_used',
       't_residuals_accepted', 't_weighted_rms', 't_cd_area_over_mass',
       't_cr_area_over_mass', 't_sedr', 't_j2k_sma', 't_j2k_ecc', 't_j2k_inc',
       't_ct_r', 't_cn_r', 't_cn_t', 'c_object_type', 'c_time_lastob_start',
       'c_time_lastob_end', 'c_cd_area_over_mass', 'c_cr_area_over_mass',
       'c_sedr', 'c_j2k_sma', 'c_j2k_ecc', 'c_j2k_inc', 't_span', 'c_span',
       't_h_apo', 't_h_per', 'c_h_apo', 'c_h_per', 'geocentric_latitude',
       'azimuth', 'elevation', 'mahalanobis_distance',
       't_position_covariance_det', 'c_position_covariance_det', 't_sigma_r',
       't_sigma_t', 't_sigma_n', 'risk_range', 'risk_category']
    
    output_features = ['risk', 'risk_category', 'event_id']
    
    @staticmethod
    def process_string(df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes string columns in the DataFrame.

        Args:
        - df (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with string columns processed.
        """
        string_columns = df.select_dtypes(include='object').columns

        # Create object mapping for each string column using dictionary comprehension
        object_mappings = {feature: {key: i for i, key in enumerate(df[feature].unique())} for feature in string_columns}

        # Map values for each string column
        for feature, mapping in object_mappings.items():
            df[feature] = df[feature].map(mapping)

        return df



    @staticmethod
    def normalization(data_frame):
        """
        Normalizes selected features in the DataFrame.

        Args:
        - data_frame (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Normalized DataFrame.
        """
        selected_features = ['max_risk_estimate', 'max_risk_scaling', 'mahalanobis_distance', 'miss_distance', 'c_time_lastob_start', 'c_position_covariance_det']
        
        # Compute mean and standard deviation for each group
        group_means = data_frame.groupby('event_id')[selected_features].transform('mean')
        group_stds = data_frame.groupby('event_id')[selected_features].transform('std')
        
        # Perform normalization for selected features
        normalized_data_selected = (data_frame[selected_features] - group_means) / group_stds
        normalized_data_selected = normalized_data_selected
        # Assign normalized values back to the specific columns using index
        data_frame.loc[:, selected_features] = normalized_data_selected
        
        return data_frame

    def split_risk_range(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the 'risk' column into risk ranges.

        Args:
        - data_frame (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with 'risk_range' column added.
        """
        # Create a new column 'risk_range' with default value -1
        data_frame['risk_range'] = -1
        
        # Assign risk ranges based on conditions
        data_frame.loc[data_frame['risk'] > -6, 'risk_range'] = 0
        data_frame.loc[(data_frame['risk'] <= -6) & (data_frame['risk'] >= -20), 'risk_range'] = 1
        data_frame.loc[data_frame['risk'] <= -20, 'risk_range'] = 1
        data_frame.loc[data_frame['risk'] == -30, 'risk_range'] = 1
        
        return data_frame
    
    @staticmethod
    def drop_least_tca(data_frame):
        """
        Drops rows with the least 'time_to_tca' within each event group.

        Args:
        - data_frame (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with rows having the least 'time_to_tca' dropped for each event.
        """

        grouped_data = data_frame.groupby('event_id')

        # Finding the index of the row with the least 'time_to_tca' within each group
        min_indices = grouped_data['time_to_tca'].idxmin()

        # Dropping the rows with the least 'time_to_tca' within each group
        filtered_data_frame = data_frame.drop(min_indices)

        return filtered_data_frame
    
    @staticmethod
    def pad_matrix(matrix, max_length):
        """
        Pad a matrix to a desired length.

        Parameters:
            matrix (numpy.ndarray): The matrix to pad.
            max_length (int): The desired length to pad the matrix to.

        Returns:
            numpy.ndarray: The padded matrix.
        """
        padded_matrix = np.zeros((max_length, matrix.shape[1]))
        padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        return padded_matrix

    @staticmethod
    def get_feature_index():
        """
        Retrieves feature indices for classification and regression.

        Returns:
        - Tuple: Tuple containing feature indices for classification and regression.
        """
        feature_order_dict = {key: idx for idx, key in zip(np.arange(0, len(DataUtils.feature_order)), DataUtils.feature_order)}
        classi_feature_indices = [feature_order_dict[feature] for feature in DataUtils.classification_features_red]
        reg_feature_indices = [feature_order_dict[feature] for feature in DataUtils.regression_features]
        return classi_feature_indices, reg_feature_indices
    
    @staticmethod
    def split_array(data, test_size):
        """
        Split the input array into training and testing sets.

        Parameters:
        - data: The input array to split.
        - test_size: The proportion of the data to include in the test set.

        Returns:
        - train_data: The training set.
        - test_data: The testing set.
        """
        # Shuffle the array
        np.random.seed(42)
        np.random.shuffle(data)

        # Calculate the split index
        split_index = int(len(data) * (1 - test_size))

        # Split the array into training and testing sets
        train_data = data[:split_index]
        test_data = data[split_index:]

        return train_data, test_data
