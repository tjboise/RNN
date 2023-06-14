from typing import Tuple
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import pandas as pd

max_time_delta = 0

class IRIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_length=10, padding_value=-1):
        self.__raw_data = data
        self.__seq_length = seq_length
        self.__padding_value = padding_value

        self.__inputs = []
        self.__outputs = []

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

            for i in range(1, len(data)):
                tmp = pd.DataFrame(data.iloc[:i])
                # Pad the sequence to length seq_length
                pad = pd.DataFrame(self.__padding_value, index=range(seq_length - len(tmp)), columns=tmp.columns)
                tmp = pd.concat([pad, tmp])[-seq_length:]
                # Add the sequence to the inputs
                self.__inputs.append(tmp[['IRI_LEFT_WHEEL_PATH', 'IRI_RIGHT_WHEEL_PATH', 'RELATIVE_TIME']].to_numpy(dtype=float))
                # Add the next row to the outputs
                self.__outputs.append(data.iloc[i][['IRI_LEFT_WHEEL_PATH', 'IRI_RIGHT_WHEEL_PATH']].to_numpy(dtype=float))
            
            data.drop(columns=["VISIT_DATE"], inplace=True)

    def __len__(self):
        return len(self.__inputs)
    
    def __getitem__(self, idx):
        normalized = self.__inputs.copy()[idx]
        normalized[:, 2] = normalized[:, 2] / max_time_delta
        return torch.from_numpy(normalized.T).float(), torch.from_numpy(self.__outputs[idx]).float()

def load_iri_datasets(seed=42,
                      train_split=0.8,
                      path="./training_data/IRI-only.parquet",
                      seq_length=10
                      ) -> Tuple[Dataset, Dataset]:
    """
    Loads the IRI dataset and returns it as train and test pytorch datasets.
    """
    raw_data = pd.read_parquet(path)
    # set NANs to -1
    raw_data.fillna(-1, inplace=True)
    # Group by SHRP_ID and STATE_CODE
    grouped_data = raw_data.groupby(["SHRP_ID", "STATE_CODE"], as_index=False)
    # Sort by date
    sorted_data = grouped_data.apply(
        lambda x: x.sort_values(["VISIT_DATE"], ascending=True))
    sorted_data.reset_index(inplace=True)
    # Mean Center and Normalize IRI
    mean_iri = (sorted_data["IRI_LEFT_WHEEL_PATH"].mean() + sorted_data["IRI_RIGHT_WHEEL_PATH"].mean()) / 2
    sorted_data["IRI_LEFT_WHEEL_PATH"] = sorted_data["IRI_LEFT_WHEEL_PATH"].replace(-1, mean_iri) - mean_iri
    sorted_data["IRI_RIGHT_WHEEL_PATH"] = sorted_data["IRI_RIGHT_WHEEL_PATH"].replace(-1, mean_iri) - mean_iri
    iri_range = max(sorted_data["IRI_LEFT_WHEEL_PATH"].max() - sorted_data["IRI_LEFT_WHEEL_PATH"].min(),
                    sorted_data["IRI_RIGHT_WHEEL_PATH"].max() - sorted_data["IRI_RIGHT_WHEEL_PATH"].min())
    sorted_data["IRI_LEFT_WHEEL_PATH"] = sorted_data["IRI_LEFT_WHEEL_PATH"] / iri_range
    sorted_data["IRI_RIGHT_WHEEL_PATH"] = sorted_data["IRI_RIGHT_WHEEL_PATH"] / iri_range
    print("Mean IRI: ", mean_iri)
    print("IRI Range: ", iri_range)
    # Split into train and test
    ids = sorted_data[["SHRP_ID", "STATE_CODE"]].drop_duplicates()
    train_ids = ids.sample(frac=train_split, random_state=seed)
    test_ids = ids.drop(train_ids.index)

    train_data = pd.merge(sorted_data, train_ids, on=["SHRP_ID", "STATE_CODE"])
    test_data = pd.merge(sorted_data, test_ids, on=["SHRP_ID", "STATE_CODE"])
    # Create datasets
    train_dataset = IRIDataset(train_data, seq_length)
    test_dataset = IRIDataset(test_data, seq_length)
    return (train_dataset, test_dataset)


if __name__ == "__main__":
    first, _ = load_iri_datasets()
    for i in range(10):
        print(first[i])