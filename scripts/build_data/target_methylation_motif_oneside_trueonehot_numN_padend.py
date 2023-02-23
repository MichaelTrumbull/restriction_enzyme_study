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
############# new code: remove Ns and pad in the middle
# num Ns
num_N_list = [item.count('N') for item in left_seqs]
# split at 6th letter. Note: There are seqs such as 'CGNNAYNNNNNTCG' that have Ns in the left. 
# I am going to ignore this because the middle padding scheme can not account for this. Better reason to go with end padding scheme when dealing with counting Ns anyways
seq_start = [item[0:6].replace('N','') for item in left_seqs]

seq_end = [item[6:].replace('N','') for item in left_seqs]

left_seqs = [a + b for a,b in zip(seq_start, seq_end)]# now it is the same as before but we removed all the Ns and gave padding accordingly with 0s
#############
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
##### new code
[a.append(b) for a,b in zip(padded_onehot_seqs, num_N_list)] # put the number N at the end
######
torch.save(torch.FloatTensor(padded_onehot_seqs), "data/target_methylation_motif_oneside_trueonehot_numN_padend.pt")
# all 136 in len
