'''
This script will use raw_data/metadata_600.csv to build the target for our training.
This build only takes the DNA seq LEFT of the slash. Because right of the slash is just 
the other half of the DNA and is probably redundant data.
'''

import pandas as pd
import torch
# label pseudo one hot values for each residue
#bas4_dict = {
#    'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
#    'R': [1,0,1,0], 'Y': [0,1,0,1], 'K': [0,0,1,1], 'M': [1,1,0,0], 'S': [0,1,1,0], 'W': [1,0,0,1],
#    'B': [0,1,1,1], 'D': [1,0,1,1], 'H': [1,1,0,1], 'V': [1,1,1,0], 'N': [1,1,1,1], 
#    '0': [0,0,0,0]
#}
bas15_dict = {
    'A': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'C': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], 'G': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], 'T': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'R': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], 'Y': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], 'K': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], 'M': [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], 'S': [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], 'W': [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'B': [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 'D': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], 'H': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], 'V': [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], 'N': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], 
    '0': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
}

df = pd.read_csv("raw_data/metadata_600.csv") 
seqs = df['Methylation Motif'].tolist()
left_seqs = [item.split('/')[0] for item in seqs]

onehot_seqs = []
for seq in left_seqs:
    onehot_seq = []
    for char in seq:
        onehot_seq = onehot_seq + bas15_dict[char]
    onehot_seqs.append(onehot_seq)

max_len = len(max(onehot_seqs, key=len))
padded_onehot_seqs = []
for line in onehot_seqs:
    padded_onehot_seqs.append(line + [0]*(max_len - len(line)))

torch.save(torch.FloatTensor(padded_onehot_seqs), "data/target_methylation_motif_oneside_trueonehot_padend.pt")
# 255 len