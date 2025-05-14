import matplotlib.pyplot as plt
import seaborn as sns
import utils
from scipy.stats import expon, gamma
import numpy as np
import pandas as pd

def plot_trade_time_hist(df
                         , time_unit = 'min'
                         , bins = 30
                         , kde=True
                         , fit_exp = True
                         , fig_size=(10, 5)):
    """Plot histogram of time between trades and overlay fitted Exponential distribution."""
    """Parameters:
        df (pd.DataFrame): dataframe with a 'timestamp_diff' column in seconds.
        time_unit (str): unit of time to convert time window between transactions into.
        bins (int): number of bins for the histogram.
        kde (bool): whether to include KDE curve.
        fit_exp (bool): whether to include fitted Exponential curve.
        fig_size (tuple): size of the plot."""
    
    TIME_COL = 'timestamp_diff'
    time_diff = utils.convert_time(df[TIME_COL], time_unit)

    # Plot histogram
    plt.figure(figsize=fig_size)
    sns.histplot(time_diff, bins=bins, kde=kde, stat='density', color='skyblue', label='Empirical')
    
    if fit_exp:
        # Fit and overlay exponential distribution
        exp_params = expon.fit(time_diff)
        x_vals = np.linspace(time_diff.min(), time_diff.max(), 100)
        exp_df = pd.DataFrame({'x':x_vals, 'y':expon.pdf(x_vals, *exp_params)})
        plt.plot(exp_df['x'], exp_df['y'], 'r-', lw=2, label='Exponential fit')

    # Labels and legend
    plt.title(f"Time Between Trades ({time_unit})")
    plt.xlabel(f"Time between trades ({time_unit})")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_reset_trades_hist(actual_df
                         , predicted_df = None
                         , bins = 30
                         , bin_width = None
                         , fig_size=(10, 5)):
    """Plot histogram of trades between resets"""
    """Parameters:
        df (pd.DataFrame): dataframe with columns 'reset' and 'trade_count'.
        bins (int): number of bins for the histogram.
        bin_width (int): width of bins for the histogram. If specified the bins count is overwritten. 
        fig_size (tuple): size of the plot."""
    
    actual_trades_per_reset = actual_df.loc[actual_df['reset'], 'trade_count']
    plt.figure(figsize=fig_size)
    
    if predicted_df is None:
        if bin_width != None:
            bins = np.arange(max(0, actual_trades_per_reset.min()-bin_width), actual_trades_per_reset.max() + bin_width, bin_width)
        # Plot histogram if only one dataframe
        sns.histplot(actual_trades_per_reset, bins=bins, kde=False)
    else:
        predicted_trades_per_reset = predicted_df.loc[predicted_df['reset'], 'trade_count']
        if bin_width != None:
            bins = np.arange(max(0, min(actual_trades_per_reset.min(), predicted_trades_per_reset.min())-bin_width)
                             , max(actual_trades_per_reset.max(), predicted_trades_per_reset.max()) + bin_width, bin_width)
        df = pd.DataFrame({'value': actual_trades_per_reset.to_list() + predicted_trades_per_reset.to_list(),
                            'type': ['Actual'] * len(actual_trades_per_reset) + ['Predicted'] * len(predicted_trades_per_reset)})
        # Plot histogram for two dataframes
        sns.histplot(data=df, x='value', hue='type', bins=bins, kde=False, element='step', stat='count', common_bins=True)
   
    # Labels and legend
    plt.title(f"Trades between resets")
    plt.xlabel(f"Trades between resets")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

def plot_price_variation(df
                         , x_unit = 'min'
                         , x_step_size = None
                         , y_step_size = None
                         , fig_size=(10, 5)):

    """Plot token price vatiation over time."""
    """Parameters:
        df (pd.DataFrame): dataframe with 'token_price_a' and 'timestamp_diff' columns.
        x_unit (str): unit to convert time axis into.
        x_step_size (int): step size for x-axis ticks.
        y_step_size (float): step size for y-axis ticks.
        fig_size (tuple): size of the matplotlib figure."""
    
    PRICE_COL = 'token_price_a'
    TIME_COL = 'timestamp_diff'
    trade_df = df[[PRICE_COL, TIME_COL]]
    trade_df['tx_time'] = utils.convert_time(trade_df[TIME_COL].cumsum(), x_unit)
    
    # Plot price over time
    plt.figure(figsize=fig_size)
    sns.lineplot(data=trade_df, x='tx_time', y=PRICE_COL)
    plt.title("Token A Price Over Time")
    plt.xlabel("Time from first transaction ({})".format(x_unit))
    plt.ylabel("Token A price")
    
    # Optional tick control
    ax = plt.gca()  # get current axis
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    if x_step_size is not None:
        plt.xticks(ticks=range(0, trade_df['tx_time'].max() + x_step_size, x_step_size))
        ax.set_xlim(x_limits)
    if y_step_size is not None:
        plt.yticks(ticks=range(trade_df[PRICE_COL].min() - y_step_size, trade_df[PRICE_COL].max() + y_step_size, y_step_size))
        ax.set_ylim(y_limits)
    plt.show()
    
def plot_error_distribution(actual_df
                            , predicted_df
                            , actual_col
                            , predicted_col
                            , title='Prediction Error Distribution'
                            , fig_size=(8, 5)):
    """Plot histogram of prediction errors."""
    """Parameters:
        actual_df (pd.DataFrame): dataframe containing actual values.
        predicted_df (pd.DataFrame): dataframe containing predicted values.
        actual_col (str): name of the column with actual values.
        predicted_col (str): name of the column with predicted values.
        title (str): title of the plot.
        fig_size (tuple): size of the matplotlib figure."""
    
    errors = predicted_df[predicted_col].values - actual_df[actual_col].values
    plt.figure(figsize=fig_size)
    sns.histplot(errors, bins=20, kde=True, color='steelblue')
    plt.title(title)
    plt.xlabel('Error (Prediction - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_predictions_vs_actual(actual_df
                            , predicted_df
                            , actual_col
                            , predicted_col
                            , title
                            , fig_size=(6, 6)):
    """Scatter plot of predicted vs actual trade counts."""
    """Parameters:
        actual_df (pd.DataFrame): dataframe containing actual values.
        predicted_df (pd.DataFrame): dataframe containing predicted values.
        actual_col (str): name of the column with actual values.
        predicted_col (str): name of the column with predicted values.
        title (str): title of the plot.
        fig_size (tuple): size of the matplotlib figure."""
    
    plt.figure(figsize=fig_size)
    sns.scatterplot(x=actual_df[actual_col].values, y=predicted_df[predicted_col].values)
    plt.plot([actual_df[actual_col].min(), actual_df[actual_col].max()],
             [actual_df[actual_col].min(), actual_df[actual_col].max()], 'r--')
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.show()
