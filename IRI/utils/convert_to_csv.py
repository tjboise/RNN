# this script will load all of the .parquet files from the datasets folder and convert them to .csv files

import os
import pandas as pd
import tqdm

DATA_FOLDER = './datasets/'

if __name__ == '__main__':
    files = os.listdir(DATA_FOLDER)
    for file in tqdm.tqdm(files):
        if file[-8:] != '.parquet':
            continue
        dataframe = pd.read_parquet(DATA_FOLDER + file)
        dataframe.to_csv(DATA_FOLDER + file[:-8] + '.csv')