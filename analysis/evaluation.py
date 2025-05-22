import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu


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
                          , confidence=0.95
                          , skip_first_in_min = False):
    """Generate distribution statistics for a pandas series."""
    """Parameters:
        series (pd.Series): numerical data.
        confidence (float): confidence level for Confidence Interval. Default: 0.95.
        skip_first_in_min (bool): skip the first entry when computing minimum."""
    """Returns: (dict): summary statistics including skewness and kurtosis."""
    
    n = len(series)
    mean = series.mean()
    sem = stats.sem(series)
    ci_low, ci_high = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
    
    return {
        'mean': mean,
        'median': series.median(),
        'std': series.std(),
        'cv': series.std()/series.mean(),
        'mad': np.median(np.abs(series - series.median())),
        'min': series[1:].min() if skip_first_in_min else series.min(),
        'max': series.max(),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'skew': stats.skew(series),
        'kurtosis': stats.kurtosis(series),
        f'{int(confidence*100)}% CI lower': ci_low,
        f'{int(confidence*100)}% CI upper': ci_high
    }


def compare_distributions(actual_df
                          , predicted_df
                          , confidence=0.95
                          , directions = ['Up', 'Down']
                          , is_mod_reset = False):
    """Compare statistical characteristics between actual and predicted values."""
    """Parameters:
        actual (pd.DataFrame): actual data with 'reset', 'trade_count', and 'direction' columns.
        predicted (pd.DataFrame): predicted data with 'reset', 'trade_count', and 'direction' columns.
        confidence (float): confidence level for CI. Default 0.95."""
    """Returns: (pd.DataFrame): comparison table of metrics."""
    actual = None
    if is_mod_reset:
        actual = actual_df.loc[actual_df['reset_mod'], 'trade_count_mod']
    else:
        actual = actual_df.loc[actual_df['reset'], 'trade_count']
        
    predicted = predicted_df.loc[predicted_df['reset'], 'trade_count']
    
    actual_stats = describe_distribution(actual, confidence)
    predicted_stats = describe_distribution(predicted, confidence)

    stats_df = pd.DataFrame([actual_stats, predicted_stats])
    stats_df['Number of trades'] = [len(actual_df), len(predicted_df)]
    stats_df['Number of resets'] = [len(actual), len(predicted)]
    stats_df['% resets'] = 100*stats_df['Number of resets']/stats_df['Number of trades']
    for direction in directions:
        stats_df['Count {}'.format(direction)] = [len(actual_df[actual_df['direction'] == direction])
                                                        , len(predicted_df[predicted_df['direction'] == direction])]
    stats_df = stats_df.T
    stats_df.columns = ['Actual', 'Predicted']
    return stats_df

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

    
def perform_hypothesis_tests(actual
                             , predicted
                             , alpha=0.05):
    """Perform t-test and Mann–Whitney U test between actual and predicted values."""
    """Parameters:
        actual (pd.Series): actual numerical data.
        predicted (pd.Series): predicted numerical data.
        alpha (float): significance level. Default 0.05."""
    """Returns: (dict): p-values and interpretation for both tests."""
    actual = pd.Series(actual).dropna()
    predicted = pd.Series(predicted).dropna()

    # Welch’s t-test (does not assume equal variance)
    t_stat, t_p = ttest_ind(actual, predicted, equal_var=False)

    # Mann–Whitney U test (non-parametric)
    u_stat, u_p = mannwhitneyu(actual, predicted, alternative='two-sided')

    return {
        't-test p-value': t_p,
        't-test significant': t_p < alpha,
        'Mann–Whitney U p-value': u_p,
        'Mann–Whitney U significant': u_p < alpha
    }