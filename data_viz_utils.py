
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import seaborn as sns
import os
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np 

class DataVizUtils:

    save_folder = "data_plots"
    risk_ranges = {
        0 : 'High Risk (Risk >= -9)',
        1 : 'Medium Risk(-20 >= Risk < -9)',
        2 : 'Low Risk (-30 > Risk < -20)',
        3 : 'Low Risk (Risk = -30)'
    }
    @staticmethod
    def plot_distribution(data, feature_name):
        """
        Plot the kernel density estimation (KDE) for a given feature in the data.

        Parameters:
        - data: Pandas Series or array-like
            The data to plot the distribution for.
        - feature_name: str
            The name of the feature (used for labeling the plot).

        Returns:
        - None
        """
        # Create a figure and axes
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot KDE on the first axis
        sns.kdeplot(data, fill=True, ax=axs[0])
        axs[0].set_title(f'KDE for {feature_name}')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Density')
        
        # Plot histogram on the second axis
        axs[1].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axs[1].set_title(f'Histogram for {feature_name}')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

        os.makedirs(DataVizUtils.save_folder, exist_ok=True)
        file_name = f'distribution_of_{feature_name}.png'
        plt.savefig(os.path.join(DataVizUtils.save_folder, file_name))
        
    @staticmethod
    def plot_risk_range_3Dscatter(data_frame, feature_1='risk', feature_2='max_risk_estimate', feature_3='max_risk_scaling'):
        
        df_ranges = {
            'Range 1 : High Risk (Risk >= -9)': data_frame[data_frame['risk'] >= -9],
            'Range 2 : Medium Risk(-20 >= Risk < -9)': data_frame[(data_frame['risk'] < -9) & (data_frame['risk'] >= -20)],
            'Range 3 : Low Risk (-30 > Risk < -20)': data_frame[(data_frame['risk'] < -20) & (data_frame['risk'] > -30)],
            'Range 4 : Low Risk (Risk = -30)': data_frame[data_frame['risk'] == -30]
        }

        color_map = {0: 'red', 1: 'green'}  # Assuming only 2 labels
        unique_labels = data_frame['label'].unique()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={'projection': '3d'})


        for (title, df), ax in zip(df_ranges.items(), axs.flatten()):
            scatter = sns.scatter(df[feature_1], df[feature_2], df[feature_3], c=df['label'].map(color_map))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Risk [base 10 log]', fontsize=10)
            ax.set_ylabel('Max Risk Estimate', fontsize=10)
            ax.set_zlabel('Max Risk Scaling', fontsize=10)
    
        # Create custom legend patches
        legend_patches = [mpatches.Patch(color=color_map[label], label=f'Class: {label}') for label in unique_labels]
        fig.legend(handles=legend_patches, loc='upper right', fontsize=8)

        os.makedirs(DataVizUtils.save_folder, exist_ok=True)

        file_name = f'3DScatter plots with different risk ranges.png'
        plt.savefig(os.path.join(DataVizUtils.save_folder, file_name))
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_class_percentage(data_frame, class_column):
    # Count the number of rows for each class
        class_counts = data_frame[class_column].value_counts()

        # Calculate total count
        total_count = class_counts.sum()

        # Plot a pie chart
        plt.figure(figsize=(6, 6))
        patches, texts, autotexts = plt.pie(class_counts, labels=[f'Class {label}' for label in class_counts.index], colors=['red', 'green'], autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Classes')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Add annotations
        annotations = [f'Class: {label}: {count} ({percent:.1f}%)' for label, count, percent in zip(class_counts.index, class_counts, class_counts / total_count * 100)]
        plt.text(0, 0, '\n'.join(annotations), horizontalalignment='center', verticalalignment='center', fontsize=10)

        # Add total count note
        plt.text(0, -1.5, f'Total count: {total_count}', horizontalalignment='center', verticalalignment='center', fontsize=10)

        os.makedirs(DataVizUtils.save_folder, exist_ok=True)

        file_name = f'Distribution of Classes.png'
        plt.savefig(os.path.join(DataVizUtils.save_folder, file_name))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_risk_range_boxplots(data_frame: pd.DataFrame):
        # Define the risk ranges

        df_range_1 =  data_frame[data_frame['risk'] >= -9]
        df_range_2 =  data_frame[(data_frame['risk'] < -9) & (data_frame['risk'] >= -20)]
        df_range_3 =  data_frame[(data_frame['risk'] < -20) & (data_frame['risk'] > -30)]
        df_range_4 =  data_frame[data_frame['risk'] == -30]

        df_range_1['risk_range'] = np.full(len(df_range_1), 0)
        df_range_2['risk_range'] = np.full(len(df_range_2), 1)
        df_range_3['risk_range'] = np.full(len(df_range_3), 2)
        df_range_4['risk_range'] = np.full(len(df_range_4), 3)

        data_frame_labelled = pd.concat([df_range_1, df_range_2, df_range_3, df_range_4], axis= 0)

        # Features to include in box plots
        features = ['time_to_tca', 'max_risk_estimate']
        fig, axs = plt.subplots( len(features), 1,  figsize=(10, 6*len(features)))

        # Iterate over features
        for i, feature in enumerate(features):
            # Create box plot for the current feature
            sns.boxplot(data=data_frame_labelled, x=feature, y = 'risk_range', orient="h",ax=axs[i])
            axs[i].set_ylabel('Risk Range')
            axs[i].set_xlabel(feature)

        plt.tight_layout()
        plt.show()

    @staticmethod   
    def plot_pie_chart(data_frame, **kwargs):

        """
        Plot a pie chart based on a DataFrame.

        Parameters:
        data_frame (pandas.DataFrame): The DataFrame containing the data to be visualized.
        grouping_feature (str, optional): The column name to group the data by. Default is 'risk_category'.
        feature_to_count (str, optional): The column name to count unique occurrences. Default is 'event_id'.
        class_name (str, optional): The name to represent the grouped feature in the chart labels. Default is 'Category'.
        title (str, optional): The title of the pie chart. Default is 'Events by Risk Category'.
        """

        # Set default values for optional parameters
        grouping_feature = kwargs.get('grouping_feature', 'risk_category')
        feature_to_count = kwargs.get('feature_to_count', 'event_id')
        class_name = kwargs.get('class_name', 'Category')
        title = kwargs.get('title', 'Events by Risk Category')

        grouped = data_frame.groupby(grouping_feature)[feature_to_count].nunique()

        plt.figure(figsize=(10, 6))
        plt.pie(grouped, labels=[f'{class_name}: {category}\n Count: {count}\n Percentage: {count / grouped.sum() * 100:.2f}%' for category, count in zip(grouped.index, grouped)],
                autopct='%1.1f%%', startangle=140)
        plt.title(title)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the image
          # Assuming DataVizUtils.save_folder holds the desired storage path
        plt.tight_layout()
        plt.savefig(os.path.join(DataVizUtils.save_folder, f'{title}.png'))

        plt.show()

    @staticmethod
    def display_box_plot(data_frame, x_feature = 'risk', y_feature = 'risk_category', group_by_feature = 'risk_category', title = 'Box Plot for feature'):

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data_frame, x = x_feature, y= y_feature, orient="h", hue= data_frame[group_by_feature])

        def calculate_limits(group):
            quartiles = group.quantile([0.0, 1.0]).to_dict()
            return quartiles[0.0], quartiles[1.0]

        # Group the data and apply the limit calculation function
        grouped_data = data_frame.groupby(group_by_feature)
        results = grouped_data[x_feature].apply(calculate_limits)
        lower_limits, upper_limits = zip(*results) 

        #plt.xlim( left = np.array(lower_limits).min(),  right = np.array(upper_limits).max())
        plt.title(f'Box plot of {title}')
        plt.xlabel(f'{x_feature}')
        plt.ylabel(f'{y_feature}')

        plt.tight_layout()
        plt.savefig(os.path.join(DataVizUtils.save_folder, f'BP {title}.png'))

        plt.show()