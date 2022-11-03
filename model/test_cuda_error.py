import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from datetime import datetime
import os
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
        #print(use_softmax)
        #self.use_softmax = use_softmax
        #print(self.use_softmax)

        if self.hid > 0: self.fc1 = nn.Linear(self.i, self.con)
        if self.hid > 1: self.fc2 = nn.Linear(self.con, self.con)
        if self.hid > 2: self.fc3 = nn.Linear(self.con, self.con)
        if self.hid > 3: self.fc4 = nn.Linear(self.con, self.con)
        if self.hid > 4: self.fc5 = nn.Linear(self.con, self.con)
        if self.hid > 5: self.fc6 = nn.Linear(self.con, self.con)
        if self.hid > 6: self.fc7 = nn.Linear(self.con, self.con)
        if self.hid > 7: self.fc8 = nn.Linear(self.con, self.con)
        if self.hid > 8: self.fc9 = nn.Linear(self.con, self.con)
        if self.hid > 9: self.fc10 = nn.Linear(self.con, self.con)

        if self.hid > 0: self.fc11 = nn.Linear(self.con, self.o)
        if self.hid == 0: self.fc11 = nn.Linear(self.i, self.o) # no connections
        print('ending init for network')
    def forward(self, x):
        print('starting forward')
        if self.hid > 0: x = F.relu(self.fc1(x))
        if self.hid > 1: x = F.relu(self.fc2(x))
        if self.hid > 2: x = F.relu(self.fc3(x))
        if self.hid > 3: x = F.relu(self.fc4(x))
        if self.hid > 4: x = F.relu(self.fc5(x))
        if self.hid > 5: x = F.relu(self.fc6(x))
        if self.hid > 6: x = F.relu(self.fc7(x))
        if self.hid > 7: x = F.relu(self.fc8(x))
        if self.hid > 8: x = F.relu(self.fc9(x))
        if self.hid > 9: x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        print('running if stateament ')
        if False:
            return split_softmax(x)
        return x

net = Net_Linear( len(train_x[0]), len(train_y[0]), args.hid, args.connections).to(device=device)

print('done')