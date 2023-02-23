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
import networks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mse = nn.MSELoss()
crossentropy = nn.CrossEntropyLoss()
def split_crossentropy(input,target):
    print('WARNING: DO NOT USE THIS FUNC. I DONT UNDERSTAND HOW IT WORKS SO IT PROBABLY DOESNT WORK AS INTENDED. ')
    '''
    Because this one-hot scheme is not mutually exclusive, cross entropy would not work. 
    Instead, we can treat each bit (not 4bit residue) as the mutually exclusive states.
    '''
    hold =  crossentropy(input[:,0], target[:,0]).item()
    hold = 0
    for i in range(int( len( input[0] ) )):
        hold = hold + crossentropy(input[:,i], target[:,i]).item()
    print(len(input[0]))
    print(hold)
    return torch.tensor( hold/len(input[0]) , requires_grad=True).to(device=device)

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=9, help="integer value of number of epochs to run for")
    parser.add_argument('--connections', type=int, default=256, help="number of connections between nodes in linear layers")
    parser.add_argument('--hid', type=int, default=3, help="number of hidden linear layers in the network")
    parser.add_argument('--lrval', type=float, default=0.001, help="lrval jump value during training")
    parser.add_argument('--batch', type=int, default=32, help="batch size. total len of dataset=600")
    parser.add_argument('--input_path', type=str, default='data/esm2_3B_avg.pt', help="location of input tensor for training")
    parser.add_argument('--target_path', type=str, default="data/Methylation_Motif_oneside.pt", help="location of input tensor for training")
    parser.add_argument('--lf',type=str,default='mse', choices=['crossent', 'mse'], help="Loss function to be used")
    parser.add_argument('--group', type=str, default='NOT_SPECIFIED', help="dir to group runs in")
    parser.add_argument('--run_name', type=str, default='NOT_SPECIFIED', help="dir to group runs in")
    args = parser.parse_args()

    rungroup = args.group

    run_name = args.run_name #datetime.now().strftime("%m_%d_%H_%M_%S_%f")
    savepath = "runs/" + rungroup + "/" + run_name
    if not os.path.exists("runs/"): os.mkdir("runs/")
    if not os.path.exists("runs/" + rungroup): os.mkdir("runs/" + rungroup)
    os.mkdir(savepath)

    train_x = torch.load(args.input_path)[0:500]
    train_y = torch.load(args.target_path)[0:500]

    valid_x = torch.load(args.input_path)[500:]
    valid_y = torch.load(args.target_path)[500:]

    with open(savepath + "/setup.log", "w") as f: 
        f.write(run_name + "\n")
        f.write('--epochs:' + str(args.epochs) + "\n")
        f.write('--connections:' + str(args.connections) + "\n")
        f.write('--hid:' + str(args.hid) + "\n")
        f.write('--lrval:' + str(args.lrval) + "\n")
        f.write('--batch:' + str(args.batch) + "\n")
        f.write('--input_path:' + str(args.input_path) + "\n")
        f.write('--target_path:' + str(args.target_path) + "\n")
        f.write('--lf:' + str(args.lf) + "\n")
        f.write('--rungroup:' + rungroup + "\n")
        f.write("--train_x_size:" + str(train_x.size()) + "\n")
        f.write("--train_y_size:" + str(train_y.size()) + "\n")


    net = networks.Net_Linear( len(train_x[0]), len(train_y[0]), args.hid, args.connections).to(device=device)
    optimizer = optim.Adam(net.parameters(), lr=args.lrval)
    BATCH_SIZE = args.batch
    EPOCHS = args.epochs

    hold_losses = []
    hold_losses_epoch = []
    validloss = []
    evalidloss = []
    for epoch in range(EPOCHS):
        hold_losses_epoch.append(0)
        evalidloss.append(0)
        for i in range(0, len(train_x), BATCH_SIZE): 
            batch_x = train_x[i:i+BATCH_SIZE]
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)

            net.zero_grad()
            outputs = net(batch_x)

            if args.lf == "mse": loss = mse(outputs, batch_y)
            if args.lf == "crossent": loss = split_crossentropy(outputs, batch_y)
            
            loss.backward()
            optimizer.step()

            hold_losses.append(loss.item()) # was originally outside batch loop...
            hold_losses_epoch[epoch] = hold_losses_epoch[epoch] + loss.item()
        for i in range(0, len(valid_x), BATCH_SIZE): 
            batch_x = valid_x[i:i+BATCH_SIZE]
            batch_y = valid_y[i:i+BATCH_SIZE]

            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)

            outputs = net(batch_x)
            vloss = mse(outputs, batch_y).item()
            validloss.append( vloss )
            evalidloss[epoch] = evalidloss[epoch] + vloss
    

    with open(savepath + "/loss.txt", "w") as f: 
        f.write(str(hold_losses))
    with open(savepath + "/epochloss.txt", "w") as f: 
        f.write(str(hold_losses_epoch))
    with open(savepath + "/validloss.txt", "w") as f: 
        f.write(str(validloss))
    with open(savepath + "/evalidloss.txt", "w") as f: 
        f.write(str(evalidloss))
    #torch.save(net.state_dict(), run_name + ".statedict" )
