import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from datetime import datetime
import networks

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=9, help="integer value of number of epochs to run for")
parser.add_argument('--connections', type=int, default=256, help="number of connections between nodes in linear layers")
parser.add_argument('--hid', type=int, default=0, help="number of hidden linear layers in the network")
parser.add_argument('--lrval', type=float, default=0.001, help="lrval jump value during training")
parser.add_argument('--type', type=str, default="lin", help="network being used (lin, conv1d, conv2d)")
parser.add_argument('--batch', type=int, default=32, help="batch size. total len of dataset=600")
parser.add_argument('--device', type=str, default="", help="Specify which gpu. Defaults to trying any gpu, then uses cpu")
parser.add_argument('--input_path', type=str, default="/data/msr-esmb1-flat-padded.pt", help="location of input tensor for training")
parser.add_argument('--target_path', type=str, default="/data/motifs-base4-numN.pt", help="location of input tensor for training")
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

input_data_path = args.input_path
input_data_path_2d = args.input_path #"../data/msr-esmb1.pt" # maybe get rid of this line and modify later code?
target_data_path = args.target_path
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

split_crossentropy = networks.split_crossentropy() # test if I can import this from networks to clean up training script IF THIS FAILS JUST COPY AND PASTE FROM NETWORKS BACK IN

def save_losses(temp_hold_losses):
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

