import pandas as pd
import numpy as np
from scipy import stats


def calculate_prediction_error(actual_df
                               , predicted_df
                               , actual_col
                               , predicted_col):
    """Calculate basic error metrics between predicted and actual values."""
    """Parameters:
        actual_df (pd.DataFrame): dataframe containing actual values.
        predicted_df (pd.DataFrame): dataframe containing predicted values.
        actual_col (str): name of the column with actual values.
        predicted_col (str): name of the column with predicted values."""
    """Returns: (dict): Dictionary with MAE, RMSE, MAPE."""
    
    actual = actual_df[actual_col].values
    predicted = predicted_df[predicted_col].values

    errors = predicted - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / actual)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def describe_distribution(series
                          , skip_first_in_min = False):
    """Generate distribution statistics for a pandas series."""
    """Parameters:
        series (pd.Series): numerical data.
        skip_first_in_min (bool): skip the first entry when computing minimum."""
    """Returns: (dict): summary statistics including skewness and kurtosis."""
    
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series[1:].min() if skip_first_in_min else series.min(),
        'max': series.max(),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'skew': stats.skew(series.dropna()),
        'kurtosis': stats.kurtosis(series.dropna())
    }


def print_stat(stat
               , title):
    """Prints formatted distribution summary statistics."""
    """Parameters:
        stat (dict): dictionary with summary statistics
        title (str): distribustion title"""
    
    print("\nDistribution Characteristics ({}):".format(title))
    print(f"Min: {stat['min']:.4f}")
    print(f"Max: {stat['max']:.4f}")
    print(f"Mean: {stat['mean']:.4f}")
    print(f"Median: {stat['median']:.4f}")
    print(f"Standard Deviation: {stat['std']:.4f}")
    print(f"Interquartile Range (IQR): {stat['iqr']:.4f}")
    print(f"Skewness: {stat['skew']:.4f}")
    print(f"Kurtosis: {stat['kurtosis']:.4f}")
    
    
def display_df_info(df,
                   timestamp_col):
    """Print a quick overview of a dataframe with respect to its timestamp range and structure."""
    """Parameters:
        df (pd.DataFrame): dataframe to be described.
        timestamp_col (str): column name containing timestamps."""
    
    print("The dataframe contains total of {} entries from {} to {}".format(len(df), df[timestamp_col].min(), df[timestamp_col].max()))
    display(df.head())
