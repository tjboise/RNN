import torch.nn as nn
import torch
import full_iri_dataset_generator as iri
from training_loop import train_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys
import json

SEQUENCE_LENGTH = 10
NUM_FEATURES_PER_SAMPLE = 14
NUM_LAYERS = 3

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=SEQUENCE_LENGTH,
                          hidden_size=SEQUENCE_LENGTH * NUM_FEATURES_PER_SAMPLE,
                          num_layers=NUM_LAYERS,
                          batch_first=True)
        self.final = nn.Linear(SEQUENCE_LENGTH * NUM_FEATURES_PER_SAMPLE, 2)

    def forward(self, x):
        hidden = torch.zeros(NUM_LAYERS,
                             x.size(0),
                             SEQUENCE_LENGTH * NUM_FEATURES_PER_SAMPLE).to(x.device)
        cell = torch.zeros(NUM_LAYERS,
                            x.size(0),
                            SEQUENCE_LENGTH * NUM_FEATURES_PER_SAMPLE).to(x.device)
        out, _ = self.rnn(x, (hidden, cell))
        out = self.final(out[:, -1, :])
        return out
    


POSSIBLE_SEQUENCE_LENGTHS = [2,3,5,10]
POSSIBLE_LR = [0.01, 0.001, 0.0001]
POSSIBLE_GAMMA = [0.5, 0.75, 0.9]
POSSIBLE_NUM_LAYERS = [1,2,3,4,5]
MAX_EPOCHS = 200
MODEL_FOLDER = ".tuning/LSTM"

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# pbar_SEQUENCE_LENGTH = tqdm(POSSIBLE_SEQUENCE_LENGTHS, leave=False)
# for SEQUENCE_LENGTH in pbar_SEQUENCE_LENGTH:
#     pbar_SEQUENCE_LENGTH.set_description(f'SEQUENCE_LENGTH: {SEQUENCE_LENGTH}')
#     train, test = iri.load_iri_datasets(path="../training_data/final_data.parquet",
#                                         construction_path="../training_data/construction_data.parquet",
#                                         seq_length=SEQUENCE_LENGTH)
#     pbar_LR = tqdm(POSSIBLE_LR, leave=False)
#     for LR in pbar_LR:
#         pbar_LR.set_description(f'LR: {LR}')        
#         pbar_GAMMA = tqdm(POSSIBLE_GAMMA, leave=False)
#         for GAMMA in pbar_GAMMA:
#             pbar_GAMMA.set_description(f'GAMMA: {GAMMA}')
#             pbar_NUM_LAYERS = tqdm(POSSIBLE_NUM_LAYERS, leave=False)
#             for NUM_LAYERS in pbar_NUM_LAYERS:
#                 if os.path.exists(f"{MODEL_FOLDER}/{SEQUENCE_LENGTH}_{NUM_LAYERS}_{LR}_{GAMMA}.pth"):
#                     continue
#                 pbar_NUM_LAYERS.set_description(f'NUM_LAYERS: {NUM_LAYERS}')
#                 model = LSTM()
#                 loss = nn.MSELoss()
#                 optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#                 lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=GAMMA)
#                 train_info = train_model(model, train, test, loss, optimizer, epochs=MAX_EPOCHS, test_every_n=1, batch_size=512, lr_scheduler=lr_scheduler)
#                 torch.save(model.state_dict(), f"{MODEL_FOLDER}/{SEQUENCE_LENGTH}_{NUM_LAYERS}_{LR}_{GAMMA}.pth")
#                 with open(f"{MODEL_FOLDER}/{SEQUENCE_LENGTH}_{NUM_LAYERS}_{LR}_{GAMMA}.json", "w") as f:
#                     json.dump(train_info, f)

#print the model and epoch with the lowest test loss and the model and epoch with the highest test r2
current_lowest_loss = sys.maxsize
current_highest_r2 = -sys.maxsize
lowest_loss = None
highest_r2 = None
for file in tqdm(os.listdir(MODEL_FOLDER), leave=False):
    if file.endswith(".json"):
        with open(f"{MODEL_FOLDER}/{file}", "r") as f:
            train_info = json.load(f)
            if train_info["test_losses"][-1] < current_lowest_loss:
                current_lowest_loss = train_info["test_losses"][-1]
                lowest_loss = file
            if train_info["test_r2s"][-1] > current_highest_r2:
                current_highest_r2 = train_info["test_r2s"][-1]
                highest_r2 = file

print(f"Lowest Loss: {current_lowest_loss} in {lowest_loss}")
print(f"Highest R2: {current_highest_r2} in {highest_r2}")