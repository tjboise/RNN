# Simple UI to get the columns and the first few rows of a selected dataframe

import os
import pandas as pd

DATA_FOLDER = './datasets/'

def clear_output():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_dataframe_and_columns():
    files = os.listdir(DATA_FOLDER)
    print('Select a dataframe (-1 to exit):')
    for i, file in enumerate(files):
        print(f'{i}: {file}')
    file_index = int(input('Enter the index of the dataframe: '))
    selected_data = {} #this will be a dictionary of {dataframename: ["list of column names"])}    
    while file_index >= 0 and file_index < len(files):        
        # clear the output
        os.system('cls' if os.name == 'nt' else 'clear')
        dataframe = pd.read_parquet(DATA_FOLDER + files[file_index])
        print(dataframe.head())
        print("")
        for i, column in enumerate(dataframe.columns):
            print(f'{i}: {column}')
        columns = input("enter a csv list of columns to select (enter 'all' to select all columns or None to select none of them): ")
        if columns == 'all':
            columns = dataframe.columns
        elif columns.lower() == 'none' or columns == '':
            columns = []
        else:
            columns = dataframe.columns[[int(i) for i in columns.split(',')]]
        selected_data[files[file_index]] = columns
        clear_output()
        print('Select a dataframe (-1 to exit):')
        for i, file in enumerate(files):
            print(f'{i}: {file}')
        file_index = int(input('Enter the index of the dataframe: '))
    for key in list(selected_data.keys()):
        if len(selected_data[key]) == 0:
            del selected_data[key]
    return selected_data

if __name__ == '__main__':  
    print(get_dataframe_and_columns())