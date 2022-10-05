import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from datetime import datetime
import networks

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=99, help="integer value of number of epochs to run for")
parser.add_argument('--connections', type=int, default=256, help="number of connections between nodes in linear layers")
parser.add_argument('--hid', type=int, default=0, help="number of hidden linear layers in the network")
parser.add_argument('--lrval', type=float, default=0.001, help="lrval jump value during training")
parser.add_argument('--type', type=str, default="lin", help="network being used (lin, conv1d, conv2d)")
parser.add_argument('--batch', type=int, default=32, help="batch size. total len of dataset=600")
parser.add_argument('--device', type=str, default="", help="Specify which gpu. Defaults to trying any gpu, then uses cpu")
args = parser.parse_args()

if args.device == "": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
print('using ', device)

hid = args.hid
t = datetime.now().strftime("%m%d%H%M%S")
con = args.connections
lrval = args.lrval

run_name = t+"-FC"+"-C"+str(con)+"-Hid"+str(hid)+"-LR"+str(lrval)+str(args.type)
print('run_name: ', run_name)

input_data_path = "../data/msr-esmb1-flat-padded.pt"
input_data_path_2d = "../data/msr-esmb1.pt"
target_data_path = "../data/motifs-base4-numN.pt"
train_x = torch.load(input_data_path).to(device=device)
train_y = torch.load(target_data_path).to(device=device)

###############
#print("train_x.size()",train_x.size())
#print("train_y.size()",train_y.size())

# Target data supplied is cut off after 600 sequences...
train_x = torch.load(input_data_path)[0:600].to(device=device)
train_y = torch.load(target_data_path)[0:600].to(device=device)

#print("train_x.size()",train_x.size())
#print("train_y.size()",train_y.size())
##############

if args.type == "lin": net = networks.Net_Linear( len(train_x[0]), len(train_y[0]), hid, con).to(device=device)
if args.type == "conv1d": net = networks.Net_Conv1d_flatten( len(train_x[0]), len(train_y[0]),k=10,ft=1,con=con ).to(device=device)
if args.type == "conv2d": 
    train_x = torch.load(input_data_path_2d).to(device=device)
    net = networks.Net_Conv2d( len(train_x[0]), len(train_x[0,0]), len(train_y[0]),k=10,ft=1,con=con ).to(device=device)

print(net)

'''
crossentropy = nn.CrossEntropyLoss()
def split_crossentropy(a, target):
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
'''
split_crossentropy = networks.split_crossentropy() # test if I can import this from networks to clean up training script

def save_losses(temp_hold_losses):
    #print('Saving hold_losses. Len = ', len(temp_hold_losses))
    with open("loss/" + run_name + ".txt", "w") as f: 
        f.write(str(temp_hold_losses))

optimizer = optim.Adam(net.parameters(), lr=lrval)
BATCH_SIZE = args.batch
EPOCHS = args.epochs

hold_losses = []
for epoch in range(EPOCHS):
    for i in range(0, len(train_x), BATCH_SIZE): 
        batch_x = train_x[i:i+BATCH_SIZE]
        batch_y = train_y[i:i+BATCH_SIZE]

        save_losses(hold_losses)

        net.zero_grad()
        outputs = net(batch_x)
        loss = split_crossentropy(outputs, batch_y)
        loss.backward()
        optimizer.step()
    hold_losses.append(loss.item())

with open("loss/" + run_name + ".txt", "w") as f: 
    f.write(str(hold_losses))

#torch.save(net.state_dict(), run_name + ".statedict" )

