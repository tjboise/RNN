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
DROPOUT = 0.5
EMBEDDING_DIM = SEQUENCE_LENGTH * NUM_FEATURES_PER_SAMPLE

class LSTMAttention(nn.Module):
    def __init__(self):
        super(LSTMAttention, self).__init__()
        self.encoder = nn.LSTM(SEQUENCE_LENGTH, EMBEDDING_DIM, batch_first=True, num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.decoder = nn.LSTM(SEQUENCE_LENGTH, EMBEDDING_DIM, batch_first=True, num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.attention = nn.MultiheadAttention(EMBEDDING_DIM, 1)
        self.linear = nn.Linear(EMBEDDING_DIM * NUM_FEATURES_PER_SAMPLE, 2)

    def forward(self, x):
        encoder_outputs, hidden = self.encoder(x)
        output, hidden = self.decoder(x, hidden)

        output, _ = self.attention(output, encoder_outputs, encoder_outputs)
        output = self.linear(output.view(output.shape[0], -1))
        return output
    


POSSIBLE_SEQUENCE_LENGTHS = [5]
POSSIBLE_LR = [0.01]
POSSIBLE_DROPOUTS = [0.5, 0.25]
POSSIBLE_NUM_LAYERS = [3]
POSSIBLEGAMMAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_EPOCHS = 200
MODEL_FOLDER = ".tuning/LSTMAttentionFine4"

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
#         pbar_DROP = tqdm(POSSIBLE_DROPOUTS, leave=False)
#         for CURRENT_DROPOUT in pbar_DROP:
#             pbar_DROP.set_description(f'DROPOUT: {CURRENT_DROPOUT}')
#             pbar_NUM_LAYERS = tqdm(POSSIBLE_NUM_LAYERS, leave=False)
#             for NUM_LAYERS in pbar_NUM_LAYERS:
#                 pbar_GAMMA = tqdm(POSSIBLEGAMMAS, leave=False)
#                 for GAMMA in pbar_GAMMA:
#                     pbar_GAMMA.set_description(f'GAMMA: {GAMMA}')
#                     if os.path.exists(f"{MODEL_FOLDER}/{SEQUENCE_LENGTH}_{NUM_LAYERS}_{LR}_{CURRENT_DROPOUT}_{GAMMA}.pth"):
#                         continue
#                     EMBEDDING_DIM = SEQUENCE_LENGTH * NUM_FEATURES_PER_SAMPLE
#                     pbar_NUM_LAYERS.set_description(f'NUM_LAYERS: {NUM_LAYERS}')
#                     model = LSTMAttention()
#                     loss = nn.MSELoss()
#                     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#                     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=GAMMA)
#                     train_info = train_model(model, train, test, loss, optimizer, epochs=MAX_EPOCHS, test_every_n=1, batch_size=512, lr_scheduler=lr_scheduler)
#                     torch.save(model.state_dict(), f"{MODEL_FOLDER}/{SEQUENCE_LENGTH}_{NUM_LAYERS}_{LR}_{CURRENT_DROPOUT}_{GAMMA}.pth")
#                     with open(f"{MODEL_FOLDER}/{SEQUENCE_LENGTH}_{NUM_LAYERS}_{LR}_{CURRENT_DROPOUT}_{GAMMA}.json", "w") as f:
#                         json.dump(train_info, f)

#print the model and epoch with the lowest test loss and the model and epoch with the highest test r2
current_lowest_loss = sys.maxsize
current_highest_r2 = -sys.maxsize
lowest_loss = None
highest_r2 = None
for file in tqdm(os.listdir(MODEL_FOLDER), leave=False):
    if file.endswith(".json"):
        with open(f"{MODEL_FOLDER}/{file}", "r") as f:
            train_info = json.load(f)
            if train_info["train_losses"][-1] < current_lowest_loss:
                current_lowest_loss = train_info["train_losses"][-1]
                lowest_loss = file
            if train_info["train_r2s"][-1] > current_highest_r2:
                current_highest_r2 = train_info["train_r2s"][-1]
                highest_r2 = file

print(f"Lowest Loss: {current_lowest_loss} in {lowest_loss}")
print(f"Highest R2: {current_highest_r2} in {highest_r2}")