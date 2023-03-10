import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from data import CustomDataset, preprocess
from model1 import build_model
from tqdm import tqdm
from evalu import evalu


def options():
    # Set up argument parser for input CSV file, batch size, learning rate, number of epochs, input dimension, and output dimension
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('--input_dim', type=int, default=1024,
                        help='dimension of hidden layers in the neural network')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='dimension of the output layer in the neural network')
    # data related
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument('--dataset_name', type=str, default='tox_data_clean.csv')
    parser.add_argument('--split', nargs='+', type=float, default=(8, 2))
    parser.add_argument('--save_path', type=str, default='saved_models/model1.pth', help='path to save trained model')
    args = parser.parse_args()

    return args

def main():
    args = options()
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Assign the arguments to variables
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    input_dim = args.input_dim
    output_dim = args.output_dim

    # Load the data
    X_train, X_test, y_train, y_test, X, y = preprocess(args.data_root, args.dataset_name, args.split)

    # Set up custom dataset and data loaders for training and testing sets
    train_set = CustomDataset(X_train, y_train, num_samples=X_train.shape[0], input_dim=X_train.shape[1], output_dim=output_dim)
    val_set = CustomDataset(X_test, y_test, num_samples=X_test.shape[0], input_dim=X_test.shape[1], output_dim=output_dim)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size)

    # Initialize the neural network and the optimizer
    model = build_model(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the loss function
    criterion = nn.BCELoss()

    # Train the neural network
    for epoch in range(epochs):
        # Train for one epoch
        model.train()
        train_loss = 0
        train_correct = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            y_pred = model(x_batch).flatten()
            loss = criterion(y_pred, y_batch)
            train_loss += loss.item()
            train_correct += ((y_pred > 0.5) == y_batch).sum().item()
            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                y_pred = model(x_batch).flatten()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_correct += ((y_pred > 0.5) == y_batch).sum().item()

        # Print the loss and accuracy for the training and validation sets
        train_loss /= len(train_set)
        train_acc = train_correct / len(train_set)
        val_loss /= len(val_set)
        val_acc = val_correct / len(val_set)
        print("Epoch {}: Train Loss {:.4f}, Train Acc {:.2f}%, Val Loss {:.4f}, Val Acc {:.2f}%".format(
            epoch+1, train_loss, 100*train_acc, val_loss, 100*val_acc))
        
        # Save the model every 10 epochs
        if (epoch+1) % 10 == 0:
            model_dir = os.path.join('saved_models', f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), model_dir)
        
    # Save the final trained model
    model_dir = os.path.join('saved_models', f"model_final.pt")
    torch.save(model.state_dict(), model_dir)
        
    evalu(X, y, model)


if __name__ == '__main__':
    main()


