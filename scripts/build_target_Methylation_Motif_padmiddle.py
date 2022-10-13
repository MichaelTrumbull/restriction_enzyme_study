'''
This script will use raw_data/metadata_600.csv to build the target for our training.
There are multiple ways to build. This script will build according to the 'Methylation Motif' column.
- NAME: Same situation as padlast however the left and right will be padded before pushing together. 
    This will keep the start of 'right' at the same spot.
'''

import pandas as pd
import torch
# label pseudo one hot values for each residue
bas4_dict = {
    'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
    'R': [1,0,1,0], 'Y': [0,1,0,1], 'K': [0,0,1,1], 'M': [1,1,0,0], 'S': [0,1,1,0], 'W': [1,0,0,1],
    'B': [0,1,1,1], 'D': [1,0,1,1], 'H': [1,1,0,1], 'V': [1,1,1,0], 'N': [1,1,1,1], 
    '0': [0,0,0,0]
}

df = pd.read_csv("raw_data/metadata_600.csv") #remove ../
seqs = df['Methylation Motif'].tolist()
split_seqs = [item.split('/') for item in seqs]

def onehot(seqs_list):
    onehot_seqs = []
    for seq in seqs_list:
        onehot_seq = []
        for char in seq:
            onehot_seq = onehot_seq + bas4_dict[char]
        onehot_seqs.append(onehot_seq)
    return onehot_seqs
def pad(onehot_seqs):
    max_len = len(max(onehot_seqs, key=len))
    padded_onehot_seqs = []
    for line in onehot_seqs:
        padded_onehot_seqs.append(line + [0]*(max_len - len(line)))
    return padded_onehot_seqs

left = []
right = []
for line in split_seqs:
    left.append( line[0] )
    try: right.append( line[1] )
    except: right.append('0')
left = pad( onehot( left ) )
right = pad( onehot( right ) )

padded_onehot_seqs = []
for l,r in zip(left,right):
    padded_onehot_seqs.append(l+r)

torch.save(torch.FloatTensor(padded_onehot_seqs), "data/target_Methylation_Motif_padlast.pt")
