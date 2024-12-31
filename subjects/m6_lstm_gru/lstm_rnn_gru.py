import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import pprint

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import kagglehub


# Convert to sequences for time series modeling
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length, 0]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# Define cell
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        """
        Custom implementation of an RNN cell. 
        Refer to the PyTorch documentation for nn.RNNCell for more details. 
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#RNNCell

        Args:
            input_size: Number of input features.
            hidden_size: Number of features in the hidden state.
            bias: Whether to include a bias term (default: True).
            nonlinearity: Nonlinearity to use ('tanh' or 'relu').
        """
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        if nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity. Choose 'tanh' or 'relu'.")

        self.nonlinearity = nonlinearity

        # Linear transformations
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the parameters using a uniform distribution.
        """
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden_state_input=None):
        """
        Forward pass for the RNN cell.

        Args:
            input: Input tensor of shape (batch_size, input_size).
            hidden_state_input: Hidden state tensor of shape (batch_size, hidden_size). Defaults to zeros if not provided.

        Returns:
            hidden_state_input: Updated hidden state (batch_size, hidden_size).
        """
        if hidden_state_input is None:
            hidden_state_input = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        # Compute the new hidden state
        hidden_state = self.input_to_hidden(input) + self.hidden_to_hidden(hidden_state_input)
        hidden_state = torch.tanh(hidden_state) if self.nonlinearity == "tanh" else torch.relu(hidden_state)

        return hidden_state
    

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        Custom implementation of an LSTM cell.

        Args:
            input_size: Number of input features.
            hidden_size: Number of features in the hidden state.
            bias: Whether to include a bias term (default: True).
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Linear layers for input-to-hidden and hidden-to-hidden transformations
        self.input_to_hidden = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size * 4, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the parameters using a uniform distribution.
        """
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden_state_tuple=None):
        """
        Forward pass for the LSTM cell.

        Args:
            input: Input tensor of shape (batch_size, input_size).
            hidden_state_tuple: Tuple (hidden state, cell state), each of shape (batch_size, hidden_size).
                Defaults to zeros if not provided.

        Returns:
            hidden_state_next: Updated hidden state (batch_size, hidden_size).
            cell_state_next: Updated cell state (batch_size, hidden_size).
        """
        # If hidden state is not provided, initialize it to zeros (the first time step)
        if hidden_state_tuple is None:
            hidden_state_tuple = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hidden_state_tuple = (hidden_state_tuple, hidden_state_tuple)

        hidden_state_tuple, cell_state_prev = hidden_state_tuple

        # Compute gates
        # TODO: Should we use a single linear layer and split the output? (for efficiency). Or create separate weights for each gate?
        gates = self.input_to_hidden(input) + self.hidden_to_hidden(hidden_state_tuple)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

        # Apply nonlinearities
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        # Update cell state and hidden state
        # k_t = f_t * cell_state_prev
        # j_t = i_t * g_t
        cell_state_next = f_t * cell_state_prev + i_t * g_t
        hidden_state_next = o_t * torch.tanh(cell_state_next)

        return hidden_state_next, cell_state_next


if __name__ == "__main__":
    path = kagglehub.dataset_download(
        "uciml/electric-power-consumption-data-set"
    )

    print("Path to dataset files:", path)

    data = pd.read_csv(
        path + '/household_power_consumption.txt',
        sep=';',
        low_memory=False,
        na_values=['nan', '?']
    )

    # Combine 'Date' and 'Time' into a single datetime column
    data['dt'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    # Drop the original 'Date' and 'Time' columns
    data.drop(columns=['Date', 'Time'], inplace=True)
    # Set 'dt' as the index
    data.set_index('dt', inplace=True)

    # Display the first few rows to confirm the result
    print(data.head())
    print(data.columns)

    # Drop rows with NaN values
    data.dropna(inplace=True)
    columns = ['Global_active_power']

    # Scale the data
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])

    seq_length = 10  # Number of time steps in each sequence
    X, y = create_sequences(data[columns].values, seq_length)

    print(f"Sequences shape: {X.shape}, Targets shape: {y.shape}")

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader for training
    batch_size = 256
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    RNNCell(input_size=3, hidden_size=4)
