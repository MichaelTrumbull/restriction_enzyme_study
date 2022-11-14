'''
Use this script to test the input and target data has the correct dimension. 
'''
import torch
import os

print('Input tensor:')
print('Needs to be 2d tensor of 600 sequences (other dim doesn\'t matter)')
try:
    i = torch.load("data/msr-esmb1-flat.pt")
    print('msr-esmb1-flat.pt size:', i.size())
except:
    print('FAILED: data/msr-esmb1-flat.pt not found')

print('Target tensor:')
print('Should also be 600 sequences (other dim doesn\'t matter')
try:
    t = torch.load("data/motifs-base4-numN.pt")
    print('motifs-base4-numN.pt size: ', t.size())
except:
    print('FAILED: motifs-base4-numN.pt not found')