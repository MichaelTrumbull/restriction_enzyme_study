import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from datetime import datetime
import networks
import os

# ! maybe put all of this in functions. Utilize if __name__ = "main":


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=9, help="integer value of number of epochs to run for")
parser.add_argument('--connections', type=int, default=256, help="number of connections between nodes in linear layers")
parser.add_argument('--hid', type=int, default=0, help="number of hidden linear layers in the network")
parser.add_argument('--lrval', type=float, default=0.001, help="lrval jump value during training")
parser.add_argument('--type', type=str, default="lin", help="network being used (lin, conv1d, conv2d)")
parser.add_argument('--batch', type=int, default=32, help="batch size. total len of dataset=600")
parser.add_argument('--input_path', type=str, default='data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt', help="location of input tensor for training")
parser.add_argument('--target_path', type=str, default="data/target_Methylation_Motif_padmiddle.pt", help="location of input tensor for training")
parser.add_argument('--lf',type=str,default='basic_mse', choices=['basic_crossent', 'crossent_105', 'crossent_136', 'basic_mse', 'mse_136'], help="Loss function to be used")
parser.add_argument('--note', type=str, default='', help="any details you want to save about the run?")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! add statement to save details of the run. such as layers, dimensions, epochs, batches... Use this info file instead the run name. save such that these details can be accessed in the future by a script


hid = args.hid
con = args.connections
lrval = args.lrval


run_name = datetime.now().strftime("%m_%d_%H_%M_%S_%f") + str(args.note)
savepath = "runs/" + run_name
os.mkdir(savepath)
    


input_data_path = args.input_path
input_data_path_2d = args.input_path #"../data/msr-esmb1.pt" # maybe get rid of this line and modify later code?
target_data_path = args.target_path

train_x = torch.load(input_data_path)
train_y = torch.load(target_data_path)


if args.type == "lin": net = networks.Net_Linear( len(train_x[0]), len(train_y[0]), hid, con).to(device=device)
if args.type == "conv1d": net = networks.Net_Conv1d_flatten( len(train_x[0]), len(train_y[0]),k=10,ft=1,con=con ).to(device=device)
#if args.type == "conv2d": 
#    train_x = torch.load(input_data_path_2d).to(device=device) #might have memory issues
#    net = networks.Net_Conv2d( len(train_x[0]), len(train_x[0,0]), len(train_y[0]),k=10,ft=1,con=con ).to(device=device)

with open(savepath + "/setup.log", "w") as f: 
    f.write(run_name + "\n")
    f.write("Note:" + str(args.note) + "\n")
    f.write('--epochs:' + str(args.epochs) + "\n")
    f.write('--connections:' + str(args.connections) + "\n")
    f.write('--hid:' + str(args.hid) + "\n")
    f.write('--lrval:' + str(args.lrval) + "\n")
    f.write('--type:' + str(args.type) + "\n")
    f.write('--batch:' + str(args.batch) + "\n")
    f.write('--input_path:' + str(args.input_path) + "\n")
    f.write('--target_path:' + str(args.target_path) + "\n")
    f.write("train_x.size():" + str(train_x.size()) + "\n")
    f.write("train_y.size():" + str(train_y.size()) + "\n")
    f.write(str(net) + "\n")

def save_losses(temp_hold_losses):
    with open(savepath + "/loss.txt", "w") as f: 
        f.write(str(temp_hold_losses))


mse = nn.MSELoss()
crossentropy = nn.CrossEntropyLoss()
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


#########################
### !!! Need more loss functions for new in/target data. until then use standard mse or cross ent.
#########################



optimizer = optim.Adam(net.parameters(), lr=lrval)
BATCH_SIZE = args.batch
EPOCHS = args.epochs

hold_losses = []
hold_losses_epoch = []
for epoch in range(EPOCHS):
    print(epoch, '/', EPOCHS, end='\r')
    hold_losses_epoch.append(0)
    for i in range(0, len(train_x), BATCH_SIZE): 
        batch_x = train_x[i:i+BATCH_SIZE]
        batch_y = train_y[i:i+BATCH_SIZE]

        batch_x = batch_x.to(device=device)
        batch_y = batch_y.to(device=device)

        save_losses(hold_losses)

        net.zero_grad()
        outputs = net(batch_x)

        if args.lf == "basic_mse": loss = mse(outputs, batch_y)
        if args.lf == "mse_136": loss = split_mse_136(outputs, batch_y)
        if args.lf == "basic_crossent": loss = crossentropy(outputs, batch_y)
        if args.lf == "crossent_105": loss = split_crossentropy_motif1st2ndhalf(outputs, batch_y)
        if args.lf == "crossent_136": loss = split_crossentropy_met_mot(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        hold_losses.append(loss.item()) # was originally outside batch loop...
        hold_losses_epoch[epoch] = hold_losses_epoch[epoch] + loss.item()

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