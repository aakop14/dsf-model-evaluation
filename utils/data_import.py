import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np


def select_file(title="Select a CSV file"
           , file_types=[("CSV files", "*.csv"), ("All files", "*.*")]
           , initial_dir=None):
    """Open a file dialog and return the selected csv file path."""
    """Parameters: 
        title (str): the window title.
        file_types (list:tuple): allowed file formats.
        initial_dir (str): the initial directory to open the window at. Default:the current folder."""
    """Returns: (str) the file location if chosen else an empty string."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring the file dialog to the front
    if initial_dir is None:
        initial_dir = os.getcwd()  # Start in the current working directory if the location is not already given
    file_path = filedialog.askopenfilename(
        parent=root,
        title=title,
        initialdir=initial_dir,
        filetypes=file_types
    )
    root.destroy()
    return file_path

def read_csv(csv_path=None
            , column_names=None
            , restrict_name_size=True):
    """Open a csv file and read into pandas dataframe. 
        If custom column names are chosen the names will be assigned to columns in the same order.
        If the csv file location is not specified, ask user to choose the file."""
    """Parameters:
       csv_path (str): location of the csv file. Default: prompts user to choose file.
       column_names (list:str): column names that will be assigned to columns in the same order.
       restrict_name_size (bool): restricts column size to the original dataframe size. 
           If False empty columns with the extra names will be added to the end of dataframe."""
    """Returns: (pd.DataFrame): dataframe with the data from the csv file."""
    # Ask user for csv file location if not specified
    if csv_path is None:
        csv_path = select_file()
    if csv_path == '':
        print('File not chosen.')
        return None 
    
    # Read csv into dataframe with the first row as headers
    df = pd.read_csv(csv_path, header=0)
    # Rename column names if custom names are given and handle size restrictions
    if column_names is not None:
        df = rename_columns(df, column_names, restrict_name_size)
    return df

def rename_columns(df
            , custom_names
            , truncate_extra=True):
    """Rename columns of the dataframe using custom_names list.
    If custom_names has fewer entries than the number of columns in the dataframe,
    only the first len(custom_names) columns are renamed. Else all column names are replaced.
    The size of column names is truncated to the original dataframe column size if truncate_extra=True."""
    """Parameters:
        df (pd.DataFrame): the dataframe whose columns are to be renamed.
        custom_names (list): list of new column names.
        truncate_extra (bool): if True and the size of custom_names is more than dataframe names, 
            then the size of custom_names is truncated to the size of dataframe columns."""
    """Returns: (pd.DataFrame): dataframe with updated column names."""
    current_cols = list(df.columns)
    num_custom = len(custom_names)
    num_existing = len(current_cols)
    
    # If fewer custom names than existing columns, rename those and keep the rest
    if num_custom < num_existing:
        df.columns = custom_names + current_cols[num_custom:]
    else:
        # If more and truncate_extra is True, ignore extra names
        df.columns = custom_names[:num_existing]
        # If truncate_extra is False, add extra columns initialized as empty
        if not truncate_extra:
            for extra_col in custom_names[num_existing:]:
                df[extra_col] = ''
    return df

def drop_tkn_amount_rows(df
            , token_column_name
            , amount_above
            , drop_nan=True):
    """Drop rows from the dataframe where the specified column has amount less or equal to amount_above."""
    """Parameters:
        df (pd.DataFrame): the dataframe to be filtered.
        token_column_name (str): the column to check amount.
        amount_above (float): rows with greater amount than specified will remain.
        drop_nan (bool): drops NaN valued rows."""
    """Returns: (pd.DataFrame): filtered dataframe."""
    
    # Filter rows based on amount threshold and optionally remove NaN values
    if drop_nan:
        return df[(df[token_column_name] > amount_above) & (~df[token_column_name].isna())]
    return df[(df[token_column_name] > amount_above)]

def read_trading_data(column_names
            , truncate_column_size = False
            , csv_path=None
            , token_amount_column_names = ['token_amount_a', 'token_amount_b']
            , grouping_interval = ''):
    """Open trading data csv file and read into pandas dataframe. 
        Clean and process the dataframe to use as an input in the model."""
    """Parameters:
       column_names (list:str): column names that will be assigned to the dataframe in the same order. 
           If less column names are given compared to the original csv column size, then only that portion will be renamed. 
       truncate_column_size (bool): restricts the size of the dataframe to the columns specified in the function.
       csv_path (str): location of the csv file. Default: prompts user to choose file.
       token_amount_column_names (list:str): names of columns containing token in/out amounts.
       grouping_interval (str): the interval to group transactions. T for minutes, H for hours. Default: emptry string, that skips this step."""
    """Returns: (pd.DataFrame): processed dataframe with the correct column names."""
    
    trading_df = read_csv(csv_path=csv_path, column_names=column_names)
    if trading_df is None:
        print('Trading csv file was not chosen.')
        return trading_df
    # Reduce columns to only once that are given
    if truncate_column_size:
        trading_df = trading_df[column_names]
        
    # Filter rows with token amount 0 or not a number
    # Drop rows with zero or NaN in token amount A (required for price calculation)
    trading_df = drop_tkn_amount_rows(trading_df, token_amount_column_names[0], 0, True) 
    # Drop rows with zero or NaN in token amount B (optional, but helps remove noise)
    trading_df = drop_tkn_amount_rows(trading_df, token_amount_column_names[1], 0, True) 
    
    #Calculate token b price with respect to token a price
    trading_df['token_price_a'] = abs(trading_df[token_amount_column_names[1]]/trading_df[token_amount_column_names[0]])
    trading_df['abs_token_b_amount'] = abs(trading_df[token_amount_column_names[1]])
    trading_df['timestamp_date'] = pd.to_datetime(trading_df['timestamp_int'], unit='s')
    # Sort by timestamp asc
    trading_df = trading_df.sort_values(by='timestamp_date').reset_index(drop=True)
    if grouping_interval != '':
        #Floor the timestamp to the nearest grouping_interval
        trading_df['timestamp_floored'] = trading_df['timestamp_date'].dt.floor(grouping_interval)
        # Factorize to assign group numbers starting from 1
        trading_df['tx_group'] = pd.factorize(trading_df['timestamp_floored'])[0] + 1
    return trading_df

def add_reset_to_strat_df(df):
    """ Adds reset detection logic to a trading strategy dataframe.
    Tracks when vault balances for both tokens become positive (at least the starting balance in each),
    and annotates each row with vault state, reset flags, and recent/last reset indices."""
    """Parameters:
        df (pd.DataFrame): dataframe with strategy trade data. Must include:
            - 'token_in', 'token_out'
            - 'token_amount_in', 'token_amount_out'
            - 'token_in_usd', 'token_out_usd'"""
    """Returns: (pd.DataFrame): modified df with added reset representative columns."""
    
    # Initialize new columns for reset tracking and vault balances
    df['last_reset'] = -1
    df['last_reset_mod'] = -1
    df['recent_reset'] = -1
    df['recent_reset_mod'] = -1
    df['reset'] = False
    df['reset_mod'] = False
    df['vault_a'] = 0
    df['vault_b'] = 0
    df['vault_a_usd'] = 0
    df['vault_b_usd'] = 0

    # Initialize running vault balances for token A and B
    vault_a = 0
    vault_a_usd = 0
    vault_b = 0
    vault_b_usd = 0
    
    # Identify token A and B using the first row
    token_a = df.loc[0, 'token_in']
    token_b = df.loc[0, 'token_out']

    for index in range(len(df)):
        # Sanity check: tokens shouldn't be the same
        if df.loc[index, 'token_in'] == df.loc[index, 'token_out']:
            raise ValueError("Token IN and OUT are the same")
            return
        
        # Update vaults based on direction of the trade
        if token_a == df.loc[index, 'token_in']:
            vault_a = vault_a + df.loc[index, 'token_amount_in']
            vault_b = vault_b - df.loc[index, 'token_amount_out']
            
            vault_a_usd = vault_a_usd + df.loc[index, 'token_in_usd']
            vault_b_usd = vault_b_usd - df.loc[index, 'token_out_usd']
        if token_b == df.loc[index, 'token_in']:
            vault_a = vault_a - df.loc[index, 'token_amount_out']
            vault_b = vault_b + df.loc[index, 'token_amount_in']
            
            vault_a_usd = vault_a_usd - df.loc[index, 'token_out_usd']
            vault_b_usd = vault_b_usd + df.loc[index, 'token_in_usd']
        
        # Store running vault balances in the DataFrame
        df.loc[index, 'vault_a'] = vault_a
        df.loc[index, 'vault_b'] = vault_b  
        df.loc[index, 'vault_a_usd'] = vault_a_usd
        df.loc[index, 'vault_b_usd'] = vault_b_usd
        # Define a reset when both vaults are positive
        df.loc[index, 'reset'] = vault_a > 0 and vault_b > 0
        # If a reset occurred at this index
        if df.loc[index, 'reset']:
            # Record the reset index
            df.loc[index, 'recent_reset'] = index
            df.loc[index, 'last_reset'] = df.loc[:index-1, 'recent_reset'].max()
            
            # Define a "modified" reset only if the previous row did not qualify
            # Avoids setting reset to a row if vault balances never reduced below starting
            if index > 0:
                if df.loc[index-1, 'vault_a'] > 0 and df.loc[index-1, 'vault_b'] > 0:
                    df.loc[index,'reset_mod'] = False
                else:
                    df.loc[index,'reset_mod'] = True
                    df.loc[index, 'recent_reset_mod'] = index
                    df.loc[index, 'last_reset_mod'] = df.loc[:index-1, 'recent_reset_mod'].max()
        
    return df

def read_raindex_data(csv_path=None
            , grouping_interval = None):
    """Open raindex trading data csv file and read into pandas dataframe. 
        Clean and process the dataframe to use in the model as an input."""
    """Parameters:
       csv_path (str): location of the csv file. Default: prompts user to choose file.
       grouping_interval (str): the interval to group transactions. T for minutes, H for hours. 
       Default: None since the transactions are representing real trading frequency."""
    """Returns: (pd.DataFrame): processed dataframe with the correct column names."""
    
    # Column names to be assigned to the dataframe. 
    #---!!!IF COLUMNS CHANGED IN THE CSV FILE THIS NEEDS TO BE ALSO CHANGED ACCORDINGLY!!!--- 
    raindex_columns =  ['timestamp_int', 'order_hash', 'order_type', 'token_in', 'token_out', 'token_amount_in', 'token_amount_out', 
               'token_in_usd', 'token_out_usd', 'io_ratio', 'sender', 'tx_hash' ]
    
    raindex_df = read_trading_data(raindex_columns, True, csv_path, ['token_amount_in', 'token_amount_out']
                                   , ('' if grouping_interval is None else grouping_interval))
    if raindex_df is None:
        print('Raindex csv was not chosen.')
        return raindex_df
    
    if grouping_interval is None:
        # Use timestamp as-is for grouping; assume no need for further time bucketing since this is real trading data
        raindex_df['timestamp_floored'] = raindex_df['timestamp_date']
        # Factorize to assign group numbers starting from 1
        raindex_df['tx_group'] = pd.factorize(raindex_df['timestamp_floored'])[0] + 1
        
    raindex_df = add_reset_to_strat_df(raindex_df)
    
    return raindex_df

def read_dex_data(csv_path=None
            , grouping_interval = '15T'):
    """Open dex pool data specific csv file and read into pandas dataframe. 
        Clean and process the dataframe to use in the model as an input."""
    """Parameters:
       csv_path (str): location of the csv file. Default: prompts user to choose file.
       grouping_interval (str): the interval to group transactions. T for minutes, H for hours. Default: '15T' 15 min."""
    """Returns: (pd.DataFrame): processed dataframe with the correct column names."""
    
    # Column names to be assigned to the dataframe. 
    #---!!!IF COLUMNS CHANGED IN THE CSV FILE THIS NEEDS TO BE ALSO CHANGED ACCORDINGLY!!!--- 
    dex_columns =  ['block_number', 'tx_hash', 'token_amount_a', 'token_amount_b', 
               'timestamp_int']
    
    dex_df = read_trading_data(dex_columns, True, csv_path, ['token_amount_a', 'token_amount_b'], grouping_interval)
    if dex_df is None:
        print('Dex csv was not chosen.')
        return dex_df
    return dex_df

def read_model_predictions(file_path = None
            , sheet_name = '15M'):
    """Open model prediction Excel file and return processed dataframe.
       Cleans and transforms the prediction data for analysis."""
    """Parameters:
       file_path (str): location of the Excel file. Default: prompts user to choose file.
       sheet_name (str): name of the sheet in the Excel file to read. Default: '15M'."""
    """Returns: (pd.DataFrame): cleaned and structured dataframe with model prediction data."""
    # Ask user for csv file location if not specified
    if file_path is None:
        file_path = select_file(title="Select the model xlsx file"
           , file_types=[("Excel files", "*.xlsx"), ("All files", "*.*")])
    # Exit function if file dialog was canceled
    if file_path == '':
        print('File not chosen.')
        return None 
    
    # Read Excel file from specified sheet, selecting columns B to U, skipping 22 metadata rows
    # If model file is changed 'usecols' and 'skiprows' arguments need to change accordingly
    df = pd.read_excel(file_path,
                        sheet_name=sheet_name,
                        usecols="B:U",      # Excel-style columns
                        skiprows=22)        # Skip first 22 rows (row 23 is index 0 after skipping)
    
    # Column names to assign to the dataframe
    # ---!!! IF EXCEL FILE STRUCTURE CHANGES, THIS LIST MUST BE UPDATED !!!---
    column_names = ['time_interval','executed_auction_price','amount','cumulative_amount', 'strategy_auctions', 'date_time','average_price', 
                    'cumulative_reset', 'gap', 'price_gap', 'reset_str', 'direction', 'duration', 'price', 'amount1', 'duration_x', 
                    'auction_count_in_reset', 'last_reset_sp', 'recent_reset_sp', 'moving_average']

    df = rename_columns(df, column_names, True)
    # Keep only rows with valid time interval and amount values
    df = df[(~df['time_interval'].isna()) & (~df['amount'].isna())]
    
    # Replace NaN values in categorical columns with empty strings
    for col in ['reset_str', 'direction']:
        df[col].fillna('', inplace = True)
    
    # Initialize reset tracking columns. The values from Excel file do not match df indices
    df['last_reset'] = -1
    df['recent_reset'] = -1
    
    # Identify reset row indices
    # 'reset_str' column marks rows where a reset occurred. Changing str to bool simplifies future use of this column. 
    df['reset'] = df['reset_str'] == 'Yes'    
    # Create boolean mask for resets
    reset_indices = df[df['reset']].index
    
    # Fill columns to track reset indices
    # For each reset row, assign its own index as 'recent_reset'
    # and find most recent prior reset index for 'last_reset' (used for modeling state transitions)
    for index in reset_indices:
        df.loc[index, 'recent_reset'] = index
        df.loc[index, 'last_reset'] = df.loc[:index-1, 'recent_reset'].max()
        
    return df