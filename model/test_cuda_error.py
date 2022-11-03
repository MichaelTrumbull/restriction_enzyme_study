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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net_Linear( 7000, 300).to(device=device)
print('Finished')
