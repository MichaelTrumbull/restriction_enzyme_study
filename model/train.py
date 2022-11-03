'''
This is a rewrite of trainmodel.py and networks.py. It has gotten away from me and rewriting previous build-code has led to better results and clarity.
To use implementations of conv1d or conv2d see networks.py
'''
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

################################################################################################################################################################################
# Network block
################################################################################################################################################################################


softmax = nn.Softmax(dim=1)
relu = nn.ReLU()
'''
def split_softmax_105(a): # NOTE: this is build for 105 length data that is 4 bit one hot and 9 bits at the end one hot.
    bases = a[:,0:105-9] # len=96 = 24 * 4base
    spaces = a[:,105-9:105] # 9 digit one hot representing number of spaces
    hold = softmax(bases[:,0:4]) # first base
    for i in range(int(len(bases[0])/4)):
        hold = torch.cat((hold, softmax(bases[:,(i+1)*4:(i+2)*4])), 1) #cat the soft max of a set of 4 onto hold
    hold = torch.cat((hold, softmax(spaces)), 1) #cat the soft max of the number of spaces
    return hold
def split_softmax_136(a): # NOTE: this is build for 136 length data that is 4 bit one hot all the way
    bases = a[:,:] # len=96 = 24 * 4base
    hold = softmax(bases[:,0:4]) # first base
    for i in range(int(len(bases[0])/4)):
        hold = torch.cat((hold, softmax(bases[:,(i+1)*4:(i+2)*4])), 1) #cat the soft max of a set of 4 onto hold
    return hold
'''
def split_softmax(a):
    if len(a[0]) == 140 or len(a[0]) == 136 or len(a[0]) == 8: #methylation_motif_(padlast *or* padmiddle) is 4bit coding only. So take softmax of each 4bit
        hold = softmax(a[:,0:4]) # first base
        for i in range(int(len(a[0])/4)):
            hold = torch.cat((hold, softmax(a[:,(i+1)*4:(i+2)*4])), 1) #cat the soft max of a set of 4 onto hold
        return hold
    if len(a[0]) == 97: #motif1sthalf2ndhalf_padmiddle_numN contains a single value at the end representing num N. This is treated different from the 4bit encoding
        bases = a[:,0:96] # len=96 = 24 * 4base
        spaces = a[:,96] # 9 digit one hot representing number of spaces
        hold = softmax(bases[:,0:4]) # first base
        for i in range(int(len(bases[0])/4)):
            hold = torch.cat((hold, softmax(bases[:,(i+1)*4:(i+2)*4])), 1) #cat the soft max of a set of 4 onto hold
        hold = torch.cat((hold, relu(spaces)), 1) #cat the soft max of the number of spaces # !!! Relu is used here because the value is usually more than 1.
        return hold
    print('ERROR: split_softmax failed. Wrong data size')

class Net_Linear(nn.Module):
    def __init__(self, input_size=10, output_size=10, hidden_layers=0, connections_between_layers=256, use_softmax = False):
        super().__init__()

        self.i = input_size #len(train_x[0])
        self.o = output_size #len(train_y[0])
        self.hid = hidden_layers
        self.con = connections_between_layers
        print(use_softmax)
        self.use_softmax = use_softmax
        print(self.use_softmax)

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
        if self.use_softmax:
            return split_softmax(x)
        return x


################################################################################################################################################################################
# Training functions block
################################################################################################################################################################################


mse = nn.MSELoss()
crossentropy = nn.CrossEntropyLoss()
def split_crossentropy(a,target):
    if len(a[0]) == 140 or len(a[0]) == 136 or len(a[0]) == 8:
        bases = a[:,:] # All are bases in this 136 bit encoding. 
        target_bases = target[:,:] # All are bases. There are no 'spaces' accounted for at the end
        hold = []
        hold.append( crossentropy(bases[:,0:4], target_bases[:,0:4]).item() )# first base
        for i in range(int(len(bases[0])/4) - 1):
            temp =  crossentropy(bases[:,(i+1)*4:(i+2)*4], target_bases[:,(i+1)*4:(i+2)*4]).item()
            hold.append( temp ) #cat the soft max of a set of 4 onto hold
        return torch.tensor( sum(hold)/34 , requires_grad=True).to(device=device) # , dtype=torch.float )
    if len(a[0]) == 97:
        bases = a[:,0:96] # len=96 = 24 * 4base
        target_bases = target[:,0:96]
        spaces = a[:,96] # 9 digit one hot representing number of spaces
        target_spaces = target[:,96]
        hold = []
        hold.append( crossentropy(bases[:,0:4], target_bases[:,0:4]).item() )# first base
        for i in range(int(len(bases[0])/4) - 1):
            temp =  crossentropy(bases[:,(i+1)*4:(i+2)*4], target_bases[:,(i+1)*4:(i+2)*4]).item()
            hold.append( temp ) #cat the soft max of a set of 4 onto hold
        hold.append( crossentropy(spaces, target_spaces).item() ) #cat the soft max of the number of spaces
        return torch.tensor( sum(hold)/25 , requires_grad=True).to(device=device) # , dtype=torch.float )
    print('ERROR: split_crossentropy')
def split_mse(a,target):
    if len(a[0]) == 140 or len(a[0]) == 136 or len(a[0]) == 8: #len 8 is used for testing
        bases = a[:,:] # All are bases in this 136 bit encoding. 
        target_bases = target[:,:] # All are bases. There are no 'spaces' accounted for at the end
        hold = []
        hold.append( mse(bases[:,0:4], target_bases[:,0:4]).item() )# first base
        for i in range(int(len(bases[0])/4) - 1):
            temp =  mse(bases[:,(i+1)*4:(i+2)*4], target_bases[:,(i+1)*4:(i+2)*4]).item()
            hold.append( temp ) #cat the soft max of a set of 4 onto hold
        return torch.tensor( sum(hold)/34 , requires_grad=True).to(device=device) # , dtype=torch.float )
    if len(a[0]) == 97:
        bases = a[:,0:96] # len=96 = 24 * 4base
        target_bases = target[:,0:96]
        spaces = a[:,96] # 9 digit one hot representing number of spaces
        target_spaces = target[:,96]
        hold = []
        hold.append( mse(bases[:,0:4], target_bases[:,0:4]).item() )# first base
        for i in range(int(len(bases[0])/4) - 1):
            temp =  mse(bases[:,(i+1)*4:(i+2)*4], target_bases[:,(i+1)*4:(i+2)*4]).item()
            hold.append( temp ) #cat the soft max of a set of 4 onto hold
        hold.append( mse(spaces, target_spaces).item() ) #cat the soft max of the number of spaces
        return torch.tensor( sum(hold)/25 , requires_grad=True).to(device=device) # , dtype=torch.float )
    print('ERROR: split_mse')
'''
def split_crossentropy_motif1st2ndhalf(a, target):
    bases = a[:,0:105-9] # len=96 = 24 * 4base
    target_bases = target[:,0:105-9]
    spaces = a[:,105-9:105] # 9 digit one hot representing number of spaces
    target_spaces = target[:,105-9:105]
    hold = []
    hold.append( crossentropy(bases[:,0:4], target_bases[:,0:4]).item() )# first base
    for i in range(int(len(bases[0])/4) - 1):
        temp =  crossentropy(bases[:,(i+1)*4:(i+2)*4], target_bases[:,(i+1)*4:(i+2)*4]).item()
        hold.append( temp ) #cat the soft max of a set of 4 onto hold
    hold.append( crossentropy(spaces, target_spaces).item() ) #cat the soft max of the number of spaces
    return torch.tensor( sum(hold)/25 , requires_grad=True).to(device=device) # , dtype=torch.float )
def split_crossentropy_met_mot(a, target): # this should be used if target data is from "data/metalation_motifs_onehot_pad.pt"
    bases = a[:,:] # All are bases in this 136 bit encoding. 
    target_bases = target[:,:] # All are bases. There are no 'spaces' accounted for at the end
    hold = []
    hold.append( crossentropy(bases[:,0:4], target_bases[:,0:4]).item() )# first base
    for i in range(int(len(bases[0])/4) - 1):
        temp =  crossentropy(bases[:,(i+1)*4:(i+2)*4], target_bases[:,(i+1)*4:(i+2)*4]).item()
        hold.append( temp ) #cat the soft max of a set of 4 onto hold
    return torch.tensor( sum(hold)/34 , requires_grad=True).to(device=device) # , dtype=torch.float )
def split_mse_136(a, target): # this should be used if target data is from "data/metalation_motifs_onehot_pad.pt"
    bases = a[:,:] # All are bases in this 136 bit encoding. 
    target_bases = target[:,:] # All are bases. There are no 'spaces' accounted for at the end
    hold = []
    hold.append( mse(bases[:,0:4], target_bases[:,0:4]).item() )# first base
    for i in range(int(len(bases[0])/4) - 1):
        temp =  mse(bases[:,(i+1)*4:(i+2)*4], target_bases[:,(i+1)*4:(i+2)*4]).item()
        hold.append( temp ) #cat the soft max of a set of 4 onto hold
    return torch.tensor( sum(hold)/34 , requires_grad=True).to(device=device) # , dtype=torch.float )
'''



################################################################################################################################################################################
# Main
################################################################################################################################################################################
if __name__ == "__main__":
    print('starting main')
    ################################################################################################################################################################################
    # Setup block
    ################################################################################################################################################################################
    rungroup = "VeryLongRun_nosoftmax_nosplit"
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=9, help="integer value of number of epochs to run for")
    parser.add_argument('--connections', type=int, default=256, help="number of connections between nodes in linear layers")
    parser.add_argument('--hid', type=int, default=0, help="number of hidden linear layers in the network")
    parser.add_argument('--lrval', type=float, default=0.001, help="lrval jump value during training")
    parser.add_argument('--type', type=str, default="lin", help="network being used (lin, conv1d, conv2d)")
    parser.add_argument('--batch', type=int, default=32, help="batch size. total len of dataset=600")
    parser.add_argument('--input_path', type=str, default='data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt', help="location of input tensor for training")
    parser.add_argument('--target_path', type=str, default="data/target_Methylation_Motif_padmiddle.pt", help="location of input tensor for training")
    parser.add_argument('--lf',type=str,default='mse', choices=['crossent', 'split_crossent', 'mse', 'split_mse'], help="Loss function to be used")
    parser.add_argument('--use_softmax',type=bool,default=False, help="Should split_softmax be applied at the final network layer?") # try not using this in the final layer https://discuss.pytorch.org/t/activation-function-for-last-layer/41151
    args = parser.parse_args()


    run_name = datetime.now().strftime("%m_%d_%H_%M_%S_%f")
    savepath = "runs/" + rungroup + "/" + run_name
    if not os.path.exists("runs/"): os.mkdir("runs/")
    if not os.path.exists("runs/" + rungroup): os.mkdir("runs/" + rungroup)
    os.mkdir(savepath)

    train_x = torch.load(args.input_path)
    train_y = torch.load(args.target_path)

    with open(savepath + "/setup.log", "w") as f: 
        f.write(run_name + "\n")
        f.write('--epochs:' + str(args.epochs) + "\n")
        f.write('--connections:' + str(args.connections) + "\n")
        f.write('--hid:' + str(args.hid) + "\n")
        f.write('--lrval:' + str(args.lrval) + "\n")
        f.write('--type:' + str(args.type) + "\n")
        f.write('--batch:' + str(args.batch) + "\n")
        f.write('--input_path:' + str(args.input_path) + "\n")
        f.write('--target_path:' + str(args.target_path) + "\n")
        f.write('--lf:' + str(args.lf) + "\n")
        f.write('--use_softmax:' + str(args.use_softmax) + "\n")
        f.write('--rungroup:' + rungroup + "\n")
        f.write("train_x.size():" + str(train_x.size()) + "\n")
        f.write("train_y.size():" + str(train_y.size()) + "\n")


    net = Net_Linear( len(train_x[0]), len(train_y[0]), args.hid, args.connections, args.use_softmax).to(device=device)
    optimizer = optim.Adam(net.parameters(), lr=args.lrval)
    BATCH_SIZE = args.batch
    EPOCHS = args.epochs

    hold_losses = []
    hold_losses_epoch = []
    for epoch in range(EPOCHS):
        print('epoch',epoch)
        hold_losses_epoch.append(0)
        for i in range(0, len(train_x), BATCH_SIZE): 
            batch_x = train_x[i:i+BATCH_SIZE]
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)

            net.zero_grad()
            outputs = net(batch_x)

            if args.lf == "mse": loss = mse(outputs, batch_y)
            if args.lf == "split_mse": loss = split_mse(outputs, batch_y)
            if args.lf == "crossent": loss = crossentropy(outputs, batch_y)
            if args.lf == "split_crossent": loss = split_crossentropy(outputs, batch_y)
            
            loss.backward()
            optimizer.step()

            hold_losses.append(loss.item()) # was originally outside batch loop...
            hold_losses_epoch[epoch] = hold_losses_epoch[epoch] + loss.item()
    
    ################################################################################################################################################################################
    # Data saving block
    ################################################################################################################################################################################

    with open(savepath + "/loss.txt", "w") as f: 
        f.write(str(hold_losses))
    with open(savepath + "/epochloss.txt", "w") as f: 
        f.write(str(hold_losses_epoch))
    #torch.save(net.state_dict(), run_name + ".statedict" )

    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.plot(hold_losses)
    plt.title('b' + str(args.batch) + 'c' + str(args.connections) + 'h' + str(args.hid) + 'target' + str(args.target_path))
    plt.savefig(savepath + "/loss.png")

    plt.figure(1)
    plt.plot(hold_losses_epoch)
    plt.title('epochs:b' + str(args.batch) + 'c' + str(args.connections) + 'h' + str(args.hid) + 'target' + str(args.target_path))
    plt.savefig(savepath + "/epochloss.png")
    print('finished')