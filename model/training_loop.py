from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

def train_model(model, train, test, loss, optimizer, epochs=20, batch_size=2048, test_every_n=1):
    """
        Trains a model using the given loss and optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    bar = tqdm(range(epochs), desc="Training Epoch")
    test_loss=0
    for i in bar:
        avgloss = 0
        num_losses = 0
        for _, data in enumerate(train_data):
            inputs, goal = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            computed_loss=loss(model(inputs), goal)
            computed_loss.backward()
            optimizer.step()

            num_losses += 1
            avgloss += computed_loss.item()
        if i % test_every_n == 0:
            test_data = DataLoader(test, batch_size=batch_size, shuffle=True)
            test_losss = 0
            test_num_losses = 0
            for _, data in enumerate(test_data):
                inputs, goal = data[0].to(device), data[1].to(device)
                test_losss += loss(model(inputs), goal).item()
                test_num_losses += 1
            test_loss = test_losss/test_num_losses
        bar.set_postfix({"Train Loss": avgloss/num_losses, "Test Loss": test_loss})