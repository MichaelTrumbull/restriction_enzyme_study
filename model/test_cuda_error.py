import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        print('Start of init')

        self.i = input_size
        self.o = output_size

        self.fc11 = nn.Linear(self.i, self.o)
        
        print('End of init')
    def forward(self, x):
        x = F.relu(self.fc11(x))
        return x

device = torch.device("cuda")

net = Net_Linear( 100, 100).to(device)
print('Finished')
