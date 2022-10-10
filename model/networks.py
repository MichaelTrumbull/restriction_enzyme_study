import torch
import torch.nn as nn
import torch.nn.functional as F

m = nn.Softmax(dim=1)
def split_softmax_105(a): # NOTE: this is build for 105 length data that is 4 bit one hot and 9 bits at the end one hot.
    bases = a[:,0:105-9] # len=96 = 24 * 4base
    spaces = a[:,105-9:105] # 9 digit one hot representing number of spaces
    hold = m(bases[:,0:4]) # first base
    for i in range(int(len(bases[0])/4)):
        hold = torch.cat((hold, m(bases[:,(i+1)*4:(i+2)*4])), 1) #cat the soft max of a set of 4 onto hold
    hold = torch.cat((hold, m(spaces)), 1) #cat the soft max of the number of spaces
    return hold
def split_softmax_136(a): # NOTE: this is build for 136 length data that is 4 bit one hot all the way
    bases = a[:,:] # len=96 = 24 * 4base
    hold = m(bases[:,0:4]) # first base
    for i in range(int(len(bases[0])/4)):
        hold = torch.cat((hold, m(bases[:,(i+1)*4:(i+2)*4])), 1) #cat the soft max of a set of 4 onto hold
    return hold
class Net_Linear(nn.Module):
    def __init__(self, input_size=10, output_size=10, hidden_layers=0, connections_between_layers=512):
        super().__init__()

        self.i = input_size #len(train_x[0])
        self.o = output_size #len(train_y[0])
        self.hid = hidden_layers
        self.con = connections_between_layers

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
    def forward(self, x):
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
        x = self.fc11(x)
        if len(x[0])==136: return split_softmax_136(x)
        return split_softmax_105(x)

class Net_Conv1d_funnel(nn.Module): #uses conv3 to flatten the kernel feature back to 1
    def __init__(self, in_len, out_len, k=5, ft=2, con=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1,ft,k)
        self.conv2 = nn.Conv1d(ft,2*ft,k)
        self.conv3 = nn.Conv1d(2*ft,1,k)
        self.fc1 = nn.Linear( in_len - ((k-1)*3) , con)
        self.fc2 = nn.Linear(con, out_len)
    def forward(self, x):
        #print(x.size())
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return split_softmax_105(x)

class Net_Conv1d_flatten(nn.Module): #uses torch.flatten(x, start_dim=1) to return to originial shape
    def __init__(self, in_len, out_len, k=5, ft=2, con=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1,ft,k)
        self.conv2 = nn.Conv1d(ft,2*ft,k)
        self.fc1 = nn.Linear(  ( in_len - ((k-1)*2) )*ft*2  , con) # -(k-1) dimension size for each conv layer. This has 2 layers. 
        #Then multiply ft*2 as that is the feature size from conv2
        self.fc2 = nn.Linear(con, out_len)
    def forward(self, x):
        #print(x.size())
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return split_softmax_105(x)

class Net_Conv2d(nn.Module):
    def __init__(self, in_len1, in_len2, out_len, k=5, ft=2, con=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1,ft,k)
        self.conv2 = nn.Conv2d(ft,2*ft,k)
        self.fc1 = nn.Linear(  ( ( in_len1-((k-1)*2) ) * ( in_len2-((k-1)*2) ) )*ft*2  , con) 
        self.fc2 = nn.Linear(con, out_len)
    def forward(self, x):
        #print(x.size())
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return split_softmax_105(x)
