from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import numpy as np

# if windows import directml backend for pytorch
try:
    import torch_directml
except:
    pass

def train_model(model,
                train,
                test,
                loss,
                optimizer,
                epochs=20,
                batch_size=2048,
                test_every_n=1,
                lr_scheduler=None):
    """
        Trains a model using the given loss and optimizer
    """
    results = {}
    train_losses = []
    test_losses = []
    lrs = []
    train_r2s = []
    test_r2s = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch_directml != None and torch_directml.is_available():
        device = torch_directml.device()
    print("Using device: ", device)
    model.to(device)
    model.train()
    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    bar = tqdm(range(epochs), desc="Training Epoch", leave=False)
    test_loss=0
    test_r2=0
    for i in bar:
        avgloss = 0
        num_losses = 0
        all_pred = []
        all_goal = []
        for _, data in enumerate(train_data):
            inputs, goal = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            computed_loss=loss(outputs, goal)
            computed_loss.backward()
            optimizer.step()

            num_losses += 1
            avgloss += computed_loss.item()

            # Store predictions and true values for r2 calculation
            all_pred.append(outputs.detach().cpu().numpy())
            all_goal.append(goal.detach().cpu().numpy())

        train_r2 = r2_score(np.concatenate(all_goal), np.concatenate(all_pred))
        train_r2s.append(train_r2)

        if i % test_every_n == 0:
            test_data = DataLoader(test, batch_size=batch_size, shuffle=True)
            test_losss = 0
            test_num_losses = 0
            all_pred_test = []
            all_goal_test = []
            for _, data in enumerate(test_data):
                inputs, goal = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                test_losss += loss(outputs, goal).item()
                test_num_losses += 1
                
                # Store predictions and true values for r2 calculation
                all_pred_test.append(outputs.detach().cpu().numpy())
                all_goal_test.append(goal.detach().cpu().numpy())

            test_loss = test_losss/test_num_losses
            test_r2 = r2_score(np.concatenate(all_goal_test), np.concatenate(all_pred_test))
            test_r2s.append(test_r2)

        bar.set_postfix({"Train Loss": avgloss/num_losses, "Test Loss": test_loss, "Train R2": train_r2, "Test R2": test_r2})

        train_losses.append(avgloss/num_losses)
        test_losses.append(test_loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        if lr_scheduler is not None:
            lr_scheduler.step()
    results["train_losses"] = train_losses
    results["test_losses"] = test_losses
    results["train_r2s"] = train_r2s
    results["test_r2s"] = test_r2s
    results["lrs"] = lrs
    return results