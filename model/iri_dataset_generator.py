from typing import Tuple
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd


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

        for group in main_bar:
            data = group[1].sort_values(["VISIT_DATE"], ascending=True)
            data["RELATIVE_TIME"] = pd.to_datetime(data["VISIT_DATE"])

            # subtract the first date from all dates to get a relative date
            data["RELATIVE_TIME"] = data["RELATIVE_TIME"] - data["RELATIVE_TIME"].iloc[0]
            data["RELATIVE_TIME"] = data["RELATIVE_TIME"].dt.days

            for i in range(1, len(data)):
                tmp = pd.DataFrame(data.iloc[:i])
                # Pad the sequence to length seq_length
                pad = pd.DataFrame(self.__padding_value, index=range(seq_length - len(tmp)), columns=tmp.columns)
                tmp = pd.concat([pad, tmp])
                # Add the sequence to the inputs
                assert tmp is not None
                self.__inputs.append(tmp[['IRI_LEFT_WHEEL_PATH', 'IRI_RIGHT_WHEEL_PATH', 'RELATIVE_TIME']].to_numpy())
                # Add the next row to the outputs
                self.__outputs.append(data.iloc[i][['IRI_LEFT_WHEEL_PATH', 'IRI_RIGHT_WHEEL_PATH']].to_numpy())
            
            data.drop(columns=["VISIT_DATE"], inplace=True)

    def __len__(self):
        return len(self.__inputs)
    
    def __getitem__(self, idx):
        return self.__inputs[idx], self.__outputs[idx]

def load_iri_datasets(seed=42,
                      train_split=0.8,
                      path="./training_data/IRI-only.parquet",
                      seq_length=10
                      ) -> Tuple[Dataset, Dataset]:
    """
    Loads the IRI dataset and returns it as train and test pytorch datasets.
    """
    raw_data = pd.read_parquet(path)
    # Group by SHRP_ID and STATE_CODE
    grouped_data = raw_data.groupby(["SHRP_ID", "STATE_CODE"], as_index=False)
    # Sort by date
    sorted_data = grouped_data.apply(
        lambda x: x.sort_values(["VISIT_DATE"], ascending=True))
    sorted_data.reset_index(inplace=True)
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