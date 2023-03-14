import torch
import torch.nn as nn

# Define the neural network architecture
class Model1(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2, l2_reg=0.001):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512,256) 
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc6(x))
        return x

    def l2_regularization(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2) ** 2
        return l2_loss

    def loss_function(self, outputs, targets):
        # Binary cross entropy loss
        bce_loss = nn.BCELoss()(outputs, targets)
        # L2 regularization loss
        l2_loss = self.l2_regularization()
        # Total loss
        total_loss = bce_loss + self.l2_reg * l2_loss
        return total_loss
    
def build_model(input_dim, output_dim):
    return Model1(input_dim, output_dim)