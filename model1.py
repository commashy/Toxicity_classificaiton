import torch
import torch.nn as nn

# Define the neural network architecture
class Model1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512,256) 
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x
    
def build_model(input_dim, output_dim):
    return Model1(input_dim, output_dim)