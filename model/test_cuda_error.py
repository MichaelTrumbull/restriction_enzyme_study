import torch
import torch.nn as nn
import torch.nn.functional as F

print('after imports')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Net_Linear(nn.Module):
    def __init__(self, input_size=10, output_size=10, hidden_layers=0, connections_between_layers=256):
        super().__init__()

        self.i = input_size #len(train_x[0])
        self.o = output_size #len(train_y[0])
        self.hid = hidden_layers
        self.con = connections_between_layers

        self.fc11 = nn.Linear(self.i, self.o)

        print('ending init for network')

    def forward(self, x):
        print('starting forward')
        x = F.relu(self.fc11(x))
        print('running if stateament ')
        return x

net = Net_Linear( 7000, 300, 0, 256).to(device=device)

print('done')