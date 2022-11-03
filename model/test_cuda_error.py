import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_Linear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        print('Start of init')
        self.fc11 = nn.Linear(input, output)
        print('End of init')

    def forward(self, x):
        x = F.relu(self.fc11(x))
        return x

device = torch.device("cuda")

net = Net_Linear(10, 10).to(device)
print('Finished')
