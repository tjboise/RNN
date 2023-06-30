from typing import Tuple
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import pandas as pd

max_time_delta = 0
mean_iri = 0
iri_range = 0

class IRIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_length=10, padding_value=-1):
        self.__raw_data = data
        self.__seq_length = seq_length
        self.__padding_value = padding_value

        self._inputs = []
        self._outputs = []

        # This will split each sequence (SHRP_ID, STATE_CODE) into sequences of length seq_length
        # that contain between 1 and seq_length-1 rows from the original dataset
        main_bar = tqdm(self.__raw_data.groupby(["SHRP_ID", "STATE_CODE"]), leave=False, desc="Generating sequences")
        global max_time_delta
        for group in main_bar:
            data = group[1].sort_values(["VISIT_DATE"], ascending=True)
            data["RELATIVE_TIME"] = pd.to_datetime(data["VISIT_DATE"])

            # subtract the first date from all dates to get a relative date
            data["RELATIVE_TIME"] = data["RELATIVE_TIME"] - data["RELATIVE_TIME"].iloc[0]
            data["RELATIVE_TIME"] = data["RELATIVE_TIME"].dt.days
            local_max = data["RELATIVE_TIME"].max()
            if local_max > max_time_delta:
                max_time_delta = local_max

            for i in range(2, len(data)):
                tmp = pd.DataFrame(data.iloc[:i])
                # Pad the sequence to length seq_length
                pad = pd.DataFrame(self.__padding_value, index=range(seq_length - len(tmp)), columns=tmp.columns)
                tmp = pd.concat([pad, tmp])[-seq_length:]

                inputs = tmp[['IRI_LEFT_WHEEL_PATH', 'IRI_RIGHT_WHEEL_PATH', 'RELATIVE_TIME']].to_numpy(dtype=float)
                inputs[-1, 0] = -1
                inputs[-1, 1] = -1

                self._inputs.append(inputs)
                self._outputs.append(data.iloc[i-1][['IRI_LEFT_WHEEL_PATH', 'IRI_RIGHT_WHEEL_PATH']].to_numpy(dtype=float))
            
            data.drop(columns=["VISIT_DATE"], inplace=True)

    def __len__(self):
        return len(self._inputs)
    
    def __getitem__(self, idx):
        normalized = self._inputs.copy()[idx]
        normalized[:, 2] = normalized[:, 2] # / max_time_delta
        return torch.from_numpy(normalized.T).float(), torch.from_numpy(self._outputs[idx]).float()
    

class IRIBucketDataset(IRIDataset):    
    def __getitem__(self, idx):
        # mean of the output vector from the parent
        mean = self._outputs[idx].mean()
        # the boundries for good, acceptable and poor IRI
        # good < boundries[0], 
        # boundries[0] <= acceptable < boundries[1],
        # boundries[1] <= poor
        boundries = [1.5, 2.68]
        target = [0] * 3 # good, acceptable, poor one-hot encoded
        if mean < (boundries[0] - mean_iri) / iri_range:
            target[0] = 1
        elif mean < (boundries[1] - mean_iri) / iri_range:
            target[1] = 1
        else:
            target[2] = 1
        return torch.from_numpy(self._inputs[idx].T).float(), torch.tensor(target).float()

def normalize_columns(df: pd.DataFrame, columns: list, iri_range: float) -> pd.DataFrame:
    """
    Normalizes the given columns in the given dataframe to (-1, 1)
    """
    for col in columns:
        df[col] = (df[col] - df[col].min()) / iri_range
    return df

def mean_center_columns(df: pd.DataFrame, columns: list, mean_iri: float) -> pd.DataFrame:
    """
    Mean centers the given columns in the given dataframe.
    """
    for col in columns:
        df[col] = df[col] - mean_iri
    return df

def load_iri_datasets(seed=42,
                      train_split=0.8,
                      path="./training_data/IRI-only.parquet",
                      seq_length=10,
                      one_hot=False
                      ) -> Tuple[Dataset, Dataset]:    
    global mean_iri, iri_range
    """
    Loads the IRI dataset and returns it as train and test pytorch datasets.
    """
    raw_data = pd.read_parquet(path)
    raw_data.fillna(1, inplace=True)

    # Group by SHRP_ID and STATE_CODE
    grouped_data = raw_data.groupby(["SHRP_ID", "STATE_CODE"], as_index=False)

    # Sorts by date
    sorted_data = grouped_data.apply(
        lambda x: x.sort_values(["VISIT_DATE"], ascending=True))
    sorted_data.reset_index(inplace=True)

    # Mean Center and Normalize IRI
    mean_iri = (sorted_data["IRI_LEFT_WHEEL_PATH"].mean() + sorted_data["IRI_RIGHT_WHEEL_PATH"].mean()) / 2


    sorted_data = mean_center_columns(sorted_data, ["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH"], mean_iri)

    iri_range = max(sorted_data["IRI_LEFT_WHEEL_PATH"].max() - sorted_data["IRI_LEFT_WHEEL_PATH"].min(),
                    sorted_data["IRI_RIGHT_WHEEL_PATH"].max() - sorted_data["IRI_RIGHT_WHEEL_PATH"].min())
    
    sorted_data = normalize_columns(sorted_data, ["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH"], iri_range)

    # Split into train and test
    ids = sorted_data[["SHRP_ID", "STATE_CODE"]].drop_duplicates()
    train_ids = ids.sample(frac=train_split, random_state=seed)
    test_ids = ids.drop(train_ids.index)

    train_data = pd.merge(sorted_data, train_ids, on=["SHRP_ID", "STATE_CODE"])
    test_data = pd.merge(sorted_data, test_ids, on=["SHRP_ID", "STATE_CODE"])

    # Create datasets
    if one_hot:
        train_dataset = IRIBucketDataset(train_data, seq_length)
        test_dataset = IRIBucketDataset(test_data, seq_length)
    else:
        train_dataset = IRIDataset(train_data, seq_length)
        test_dataset = IRIDataset(test_data, seq_length)
    return (train_dataset, test_dataset)