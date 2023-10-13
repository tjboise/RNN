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
    def __init__(self, measurement_info : pd.DataFrame = None, seq_length=10, padding_value=-1, parquet=False):
        global max_time_delta

        self.__raw_data = measurement_info
        self.__padding_value = padding_value

        self._inputs = []
        self._outputs = []
        nskipped = 0

        if parquet is not False:  
            metadata = pd.read_parquet(parquet + "_metadata.parquet")        
            for i in tqdm(range(metadata["length"]), desc="Loading from parquet"):
                self._inputs.append(pd.read_parquet(parquet + f"_inputs_{i}.parquet").to_numpy(dtype=float))
                self._outputs.append(pd.read_parquet(parquet + f"_outputs_{i}.parquet").to_numpy(dtype=float))

        # This will split each sequence (SHRP_ID, STATE_CODE) into sequences of length seq_length
        # that contain between 1 and seq_length-1 rows from the original dataset
        main_bar = tqdm(self.__raw_data.groupby(["SHRP_ID", "STATE_CODE"]), leave=False, desc="Generating sequences")
        for group in main_bar:
            measurement_info = group[1].sort_values(["VISIT_DATE"], ascending=True)
            measurement_info["RELATIVE_TIME"] = pd.to_datetime(measurement_info["VISIT_DATE"])

            # subtract the first date from all dates to get a relative date
            measurement_info["RELATIVE_TIME"] = measurement_info["RELATIVE_TIME"] - measurement_info["RELATIVE_TIME"].iloc[0]
            measurement_info["RELATIVE_TIME"] = measurement_info["RELATIVE_TIME"].dt.days
            local_max = measurement_info["RELATIVE_TIME"].max()
            if local_max > max_time_delta:
                max_time_delta = local_max

            for i in range(2, len(measurement_info)):
                # If the last element in the sequence is a construction element, skip it
                # if measurement_info.iloc[i-1]['IRI_LEFT_WHEEL_PATH'] <= 0 or measurement_info.iloc[i-1]['IRI_RIGHT_WHEEL_PATH'] <= 0:
                #     nskipped += 1
                # else:
                tmp = pd.DataFrame(measurement_info.iloc[:i])
                # Pad the sequence to length seq_length
                pad = pd.DataFrame(self.__padding_value, index=range(seq_length - len(tmp)), columns=tmp.columns)
                tmp = pd.concat([pad, tmp])[-seq_length:]
                inputs = tmp[['MAX_MEAN_DEPTH_WIRE_REF',
                            'IRI_LEFT_WHEEL_PATH',
                            'IRI_RIGHT_WHEEL_PATH',
                            'AADT_ALL_VEHIC',
                            'MEPDG_TRANS_CRACK_LENGTH_AC',
                            'RELATIVE_TIME',
                            'IMP_TYPE',
                            'MAX_ANN_HUM_AVG',
                            'MIN_ANN_HUM_AVG',
                            'MEAN_ANN_TEMP_AVG',
                            'FREEZE_THAW_YR',
                            'TOTAL_ANN_PRECIP',
                            'TOTAL_SNOWFALL_YR',
                            'LAYER_TYPE']].to_numpy(dtype=float)
                inputs[-1, 0] = -1

                self._inputs.append(inputs)
                self._outputs.append(measurement_info.iloc[i-1][['MAX_MEAN_DEPTH_WIRE_REF']].to_numpy(dtype=float))
            
            measurement_info.drop(columns=["VISIT_DATE"], inplace=True)

    def __len__(self):
        return len(self._inputs)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self._inputs[idx].T).float(), torch.from_numpy(self._outputs[idx]).float()
    
    def to_parquet(self, path):
        for i in tqdm(range(len(self._inputs)), desc="Saving to parquet"):
            # write a parquet file for each sequence
            pd.DataFrame(self._inputs[i]).to_parquet(path + f"_inputs_{i}.parquet", index=False)
            pd.DataFrame(self._outputs[i]).to_parquet(path + f"_outputs_{i}.parquet", index=False) 
        #write metadata file. This file contains the number of sequences and the length of each sequence
        pd.DataFrame({"length": len(self._inputs), "seq_length": len(self._inputs[0])}).to_parquet(path + "_metadata.parquet", index=False)

    

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

def load_rut_datasets(seed=42,
                      train_split=0.8,
                      path="./training_data/final_data.parquet",
                      construction_path="./training_data/construction_data.parquet",
                      seq_length=10,
                      one_hot=False
                      ) -> Tuple[Dataset, Dataset]:    
    global mean_iri, iri_range
    """
    Loads the RUT dataset and returns it as train and test pytorch datasets.
    """
    cache_file_name = str(seed) + "_" + str(train_split) + "_" + str(seq_length) + "_" + str(one_hot) + ".parquet"
    try:
        print("Loading cached data:")
        print("train_" + cache_file_name)
        test_dataset = IRIDataset(parquet="./training_data/test_" + cache_file_name)
        train_dataset = IRIDataset(parquet="./training_data/train_" + cache_file_name)
        return (train_dataset, test_dataset)

    except FileNotFoundError:
        print("Cached data not found, regenerating...")
        raw_data = pd.read_parquet(path)

        construction_data = pd.read_parquet(construction_path)
        construction_data.fillna(1, inplace=True)
        construction_data.rename(columns={"IMP_DATE": "VISIT_DATE"}, inplace=True)
        construction_data["VISIT_DATE"] = pd.to_datetime(construction_data["VISIT_DATE"], format="%m/%d/%Y")
        raw_data = pd.merge(raw_data, construction_data, on=["SHRP_ID", "STATE_CODE", "VISIT_DATE"], how="outer")
        raw_data.fillna(-1, inplace=True)

        # Group by SHRP_ID and STATE_CODE
        grouped_data = raw_data.groupby(["SHRP_ID", "STATE_CODE"], as_index=False)

        # Sorts by date
        sorted_data = grouped_data.apply(
            lambda x: x.sort_values(["VISIT_DATE"], ascending=True))
        sorted_data.reset_index(inplace=True)

        # Mean Center and Normalize IRI
        mean_iri = (sorted_data["IRI_LEFT_WHEEL_PATH"].mean() + sorted_data["IRI_RIGHT_WHEEL_PATH"].mean()) / 2
        mean_aadt = sorted_data["AADT_ALL_VEHIC"].mean()
        mean_transverse_cracks = sorted_data["MEPDG_TRANS_CRACK_LENGTH_AC"].mean()
        mean_max_annual_humidity = sorted_data["MAX_ANN_HUM_AVG"].mean()
        mean_min_annual_humidity = sorted_data["MIN_ANN_HUM_AVG"].mean()
        mean_mean_annual_temp = sorted_data["MEAN_ANN_TEMP_AVG"].mean()
        mean_freeze_thaw_cycles = sorted_data["FREEZE_THAW_YR"].mean()
        mean_total_annual_precipitation = sorted_data["TOTAL_ANN_PRECIP"].mean()
        mean_total_snowfall = sorted_data["TOTAL_SNOWFALL_YR"].mean()
        mean_rut = sorted_data["MAX_MEAN_DEPTH_WIRE_REF"].mean()


        sorted_data = mean_center_columns(sorted_data, ["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH"], mean_iri)
        sorted_data = mean_center_columns(sorted_data, ["AADT_ALL_VEHIC"], mean_aadt)
        sorted_data = mean_center_columns(sorted_data, ['MEPDG_TRANS_CRACK_LENGTH_AC'], mean_transverse_cracks)
        sorted_data = mean_center_columns(sorted_data, ['MAX_ANN_HUM_AVG'], mean_max_annual_humidity)
        sorted_data = mean_center_columns(sorted_data, ['MIN_ANN_HUM_AVG'], mean_min_annual_humidity)
        sorted_data = mean_center_columns(sorted_data, ['MEAN_ANN_TEMP_AVG'], mean_mean_annual_temp)
        sorted_data = mean_center_columns(sorted_data, ['FREEZE_THAW_YR'], mean_freeze_thaw_cycles)
        sorted_data = mean_center_columns(sorted_data, ['TOTAL_ANN_PRECIP'], mean_total_annual_precipitation)
        sorted_data = mean_center_columns(sorted_data, ['TOTAL_SNOWFALL_YR'], mean_total_snowfall)
        sorted_data = mean_center_columns(sorted_data, ['MAX_MEAN_DEPTH_WIRE_REF'], mean_rut)

        iri_range = max(sorted_data["IRI_LEFT_WHEEL_PATH"].max() - sorted_data["IRI_LEFT_WHEEL_PATH"].min(),
                        sorted_data["IRI_RIGHT_WHEEL_PATH"].max() - sorted_data["IRI_RIGHT_WHEEL_PATH"].min())
        aadt_range = sorted_data["AADT_ALL_VEHIC"].max() - sorted_data["AADT_ALL_VEHIC"].min()
        transverse_cracking_range = sorted_data["MEPDG_TRANS_CRACK_LENGTH_AC"].max() - sorted_data["MEPDG_TRANS_CRACK_LENGTH_AC"].min()
        max_annual_humidity_range = sorted_data["MAX_ANN_HUM_AVG"].max() - sorted_data["MAX_ANN_HUM_AVG"].min()
        min_annual_humidity_range = sorted_data["MIN_ANN_HUM_AVG"].max() - sorted_data["MIN_ANN_HUM_AVG"].min()
        mean_annual_temp_range = sorted_data["MEAN_ANN_TEMP_AVG"].max() - sorted_data["MEAN_ANN_TEMP_AVG"].min()
        freeze_thaw_cycles_range = sorted_data["FREEZE_THAW_YR"].max() - sorted_data["FREEZE_THAW_YR"].min()
        total_annual_precipitation_range = sorted_data["TOTAL_ANN_PRECIP"].max() - sorted_data["TOTAL_ANN_PRECIP"].min()
        total_snowfall_range = sorted_data["TOTAL_SNOWFALL_YR"].max() - sorted_data["TOTAL_SNOWFALL_YR"].min()
        rut_range = sorted_data["MAX_MEAN_DEPTH_WIRE_REF"].max() - sorted_data["MAX_MEAN_DEPTH_WIRE_REF"].min()
        
        
        sorted_data = normalize_columns(sorted_data, ["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH"], iri_range)
        sorted_data = normalize_columns(sorted_data, ["AADT_ALL_VEHIC"], aadt_range)
        sorted_data = normalize_columns(sorted_data, ["MEPDG_TRANS_CRACK_LENGTH_AC"], transverse_cracking_range)
        sorted_data = normalize_columns(sorted_data, ["MAX_ANN_HUM_AVG"], max_annual_humidity_range)
        sorted_data = normalize_columns(sorted_data, ["MIN_ANN_HUM_AVG"], min_annual_humidity_range)
        sorted_data = normalize_columns(sorted_data, ["MEAN_ANN_TEMP_AVG"], mean_annual_temp_range)
        sorted_data = normalize_columns(sorted_data, ["FREEZE_THAW_YR"], freeze_thaw_cycles_range)
        sorted_data = normalize_columns(sorted_data, ["TOTAL_ANN_PRECIP"], total_annual_precipitation_range)
        sorted_data = normalize_columns(sorted_data, ["TOTAL_SNOWFALL_YR"], total_snowfall_range)
        sorted_data = normalize_columns(sorted_data, ["MAX_MEAN_DEPTH_WIRE_REF"], rut_range)

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

        #save cache files
        # train_dataset.to_parquet("./training_data/train_" + cache_file_name)
        # test_dataset.to_parquet("./training_data/test_" + cache_file_name)

        return (train_dataset, test_dataset)


if __name__ == "__main__":
    tmp = load_rut_datasets()
    for i in range(100):
        print(tmp[0][i])