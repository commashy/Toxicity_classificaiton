import os

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess(data_root='dataset', dataset_name='tox_data_clean.csv', split=(8, 1, 1)):
    # Read the input CSV file
    df = pd.read_csv(os.path.join(data_root, dataset_name))

    # Create a list of SMILES strings
    smiles_list = list(df['smiles'])

    # Convert SMILES strings to molecular fingerprints
    X = torch.zeros((len(smiles_list), 1024))
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            bi = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bi)
            X[i] = torch.tensor(fp)
        else:
            print(f"Invalid SMILES string at index {i}: {smi}")

    # Extract toxicity labels from the input CSV file
    y = torch.tensor(list(df['NR.AhR']))

    # Split the data into training and testing sets
    ratio = (split[1] + split[2])/10
    ratio2 = split[2]/(split[1] + split[2])
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=ratio2)

    # Create dataframes
    # train_df = pd.concat([X_train, y_train], axis=1)
    # val_df = pd.concat([X_val, y_val], axis=1)
    # test_df = pd.concat([X_test, y_test], axis=1)

    # Save the dataframes as CSV files
    # train_df.to_csv(os.path.join(data_root, f'train_{dataset_name}.csv'), index=False)
    # val_df.to_csv(os.path.join(data_root, f'val_{dataset_name}.csv'), index=False)
    # test_df.to_csv(os.path.join(data_root, f'test_{dataset_name}.csv'), index=False)
    # return X_train, X_val, y_train, y_val, X, y
    return X_train, X_val, X_test, y_train, y_val, y_test

# def main():
#     # Set up argument parsers for data_root, dataset_name, and split
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='dataset')
#     parser.add_argument('--dataset_name', type=str, default='tox_data_clean.csv')
#     parser.add_argument('--split', nargs='+', type=float, default=(8, 1, 1))
#     args = parser.parse_args()

#     # Parse the arguments
#     X_train, X_val, y_train, y_val = preprocess(args.data_root, args.dataset_name, args.split)
#     return X_train, X_val, y_train, y_val

# if __name__ == '__main__':
#     main()
