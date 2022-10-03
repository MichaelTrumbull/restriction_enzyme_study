'''
Script to build torch tensors for: Type I RMS metalation motifs
Before starting: 
    Use ESM-b1 language model to convert 'm', 'r', and 's' FASTA files to richer data tensor
    ESM-b1: https://github.com/facebookresearch/esm *Should use the script provided for conversion
    Need metalation motifs in csv form


'''

# each esmb1 tensor is flattened and padded to get a 2d tensor

BUILD_PARTIAL = False # when any of the builds run it will also save a tensor with len=2 (instead of len=600) for trial runs on personal computer

INPUT_DIRS = [ "data/m1split/", "data/m2split/", "data/r1split/", "data/r2split/", "data/s1split/", "data/s2split/" ]

import os
if not os.path.exists("data/"): os.mkdir("data/") # we will save the outputs to this directory


print('RUNNING BUILD_ESMB1_PAD_FLAT')
'''
- Data came from Dr. Gang Fang
- Data was 3 files of input data (split into M R and S) and 1 target file (only 600 seq long containing metalation motifs)
- M R and S sequence lengths were too big (>1024) for ESM-b1 protien language model processing so these files were split into 2
- prebuilt script to translate seq's into language model tensors
    - python scripts/extract.py esm1b_t33_650M_UR50S filename.faa newsavelocation/ --repr_layers 0 32 33 --include mean per_tok
'''
import os
import torch
import torch.nn.functional as F
import numpy as np

# .pt files built using premade script: python scripts/extract.py esm1b_t33_650M_UR50S filename.faa newsavelocation/ --repr_layers 0 32 33 --include mean per_tok
# .pt files were generated for each enzyme for each file
m1s = []
m2s = []
r1s = []
r2s = []
s1s = []
s2s = []
directory = INPUT_DIRS
for filename in os.listdir(directory[0]):
    f = os.path.join(directory[0], filename)
    if f[-3:len(f)] == ".pt":
        m1s.append(torch.load(f)["representations"][33])
for filename in os.listdir(directory[1]):
    f = os.path.join(directory[1], filename)
    if f[-3:len(f)] == ".pt":
        m2s.append(torch.load(f)["representations"][33])
for filename in os.listdir(directory[2]):
    f = os.path.join(directory[2], filename)
    if f[-3:len(f)] == ".pt":
        r1s.append(torch.load(f)["representations"][33])
for filename in os.listdir(directory[3]):
    f = os.path.join(directory[3], filename)
    if f[-3:len(f)] == ".pt":
        r2s.append(torch.load(f)["representations"][33])
for filename in os.listdir(directory[4]):
    f = os.path.join(directory[4], filename)
    if f[-3:len(f)] == ".pt":
        s1s.append(torch.load(f)["representations"][33])
for filename in os.listdir(directory[5]):
    f = os.path.join(directory[5], filename)
    if f[-3:len(f)] == ".pt":
        s2s.append(torch.load(f)["representations"][33])

print("Checking torch.cat on [0], [1], [2]")
i = 0
print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())
i = 1
print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())
i = 2
print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())

print("make data_list by: data_list.append(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) )) ")
data_list = []
for i in range(len(m1s)):
    data_list.append(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ))

#### flatten
data_list_flat = []
for line in data_list:
    data_list_flat.append(torch.flatten(line))
print("data_list_flat[0].size()", data_list_flat[0].size())

#### get padding size
len_list = []
for list in data_list_flat:
    len_list.append(len(list))
max_len_val = max(len_list)
print("max_len_val", max_len_val)

#### PAD

data_list_flat_padded = []
for i, line in enumerate(data_list_flat):
    data_list_flat_padded.append( F.pad(data_list_flat[i], ( max_len_val-len(line), 0), "constant" ) )
print("data_list_flat_padded[0].size()", data_list_flat_padded[0].size())

#### build list of tensors into single torch tensor
print("torch stacking the list of tensors into single tensor")
data = torch.stack(data_list_flat_padded)

print("data.size()", data.size())

print('saving')
torch.save(data, "data/msr-esmb1-flat.pt")
if BUILD_PARTIAL: torch.save(data[0:2], "data/PARTIALDATAmsr-esmb1-flat.pt")
