'''
This will onehot encode the motif 1st half and 2nd half columns in metadata_600.csv.
To keep each sub sequence (left or right of /) in the same location, each will be padded the same before combining.
Pad size for sub seq: 6
'''
from re import M
from tkinter import RIGHT
import pandas as pd
import torch
# label pseudo one hot values for each residue
bas4_dict = {
    'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
    'R': [1,0,1,0], 'Y': [0,1,0,1], 'K': [0,0,1,1], 'M': [1,1,0,0], 'S': [0,1,1,0], 'W': [1,0,0,1],
    'B': [0,1,1,1], 'D': [1,0,1,1], 'H': [1,1,0,1], 'V': [1,1,1,0], 'N': [1,1,1,1],
    '0': [0,0,0,0]# the space between 
}

df = pd.read_csv("raw_data/metadata_600.csv") #remove ../
first = df['Motif_1stHalf'].tolist()
second = df['Motif_2ndHalf'].tolist()

max_len = 6
padded = []
for i, line in enumerate(first):
    padded.append(line.split('/')[0] + '0'*(max_len - len(line.split('/')[0])))
    try: padded[i] = padded[i] + line.split('/')[1] + '0'*(max_len - len(line.split('/')[1]))
    except: padded[i] = padded[i] + '000000'
for i, line in enumerate(second):
    padded[i] = padded[i] + line.split('/')[0] + '0'*(max_len - len(line.split('/')[0]))
    try: padded[i] = padded[i] + line.split('/')[1] + '0'*(max_len - len(line.split('/')[1]))
    except: padded[i] = padded[i] + '000000'

onehot_list = []
for line in padded:
    onehot_line = []
    for char in line:
        onehot_line = onehot_line + bas4_dict[char]
    onehot_list.append(onehot_line)

### NOTE the issue with counting the number of Ns is there are Ns present between regular residues.
### this means i would need to remove the corresponding start and end of each sequence given
### to be able to count the number of Ns that are only for spacing. 
methylationmotif = df['Methylation Motif'].tolist()

# Test if both sites have the same spacing of Ns. 
'''
for i, line in enumerate(full):
    try: 
        l = line.split('/')[0]
        r = line.split('/')[1]
        left_ns = l[ len( first[i].split('/')[0] ) : -len(second[i].split('/')[1]) ].count('N')
        right_ns = r[ len( second[i].split('/')[0] ) : -len(first[i].split('/')[1]) ].count('N')
    except:
        l = line.split('/')[0]
        right_ns = 0
        left_ns = l[ len( first[i].split('/')[0] ) : -len(first[i].split('/')[1]) ].count('N')
    if left_ns != right_ns:
        print()
        print(line)
        print(first[i].split('/')[0], first[i].split('/')[1])
        print(l[ len( first[i].split('/')[0] ) : -len(first[i].split('/')[1]) ])
        print(left_ns, right_ns)
'''# Both sites always have the same number of N spacing

# Sol: Count all Ns. Subtract Ns present in the 1stHalf/2ndHalf characters. Now that we have all the Ns only in the middle spacing, we can divide by two for the two different sites.
for i, line in enumerate(methylationmotif):
    if methylationmotif[i].count('/') == 0:
        ns = methylationmotif[i].count('N') - first[i].count('N')
    else:
        ns = methylationmotif[i].count('N') - first[i].count('N') - second[i].count('N')
    ns = ns/2
    onehot_list[i].append(ns)
    
torch.save(torch.FloatTensor(onehot_list), "data/target_motif1stHalf2ndHalf_padmiddle_numN.pt")
