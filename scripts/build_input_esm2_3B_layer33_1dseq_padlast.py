'''
- This script builds esm2(3B) input data used in training. 
- From metadata_600 we get the protiens we are using (600 of them) and the order to place them.
- Target data will be built from metadata_600 using the same protiens and ordering.
- Data will be a 2d torch tensor
    - ESM2 creates a 2d tensor per sequence
    - This tensor will be flattened so each sequence is a 1d tensor
    - These 1d tensors will be stacked to our final 2d tensor
        - Note: this data will be padded AFTER the m,r, and s tokins are combined.
- two saves:
    - tokenrep is a flattened vector of layer 33 from esm2(3B)
    - sequencerep is an average of tokenrep. much smaller.
'''

import torch
import pandas as pd
from Bio import SeqIO
import esm

M_PATH = "raw_data/Type_I_M_subunit_genes_Protein_g_clean_sorted.faa"
R_PATH = "raw_data/Type_I_R_subunit_genes_Protein_g_clean_sorted.faa"
S_PATH = "raw_data/Type_I_S_subunit_genes_Protein_g_clean_sorted.faa"
METADATA_PATH = "raw_data/metadata_600.csv"


mrs_dict = {
    'm': {rec.id : rec.seq for rec in SeqIO.parse(M_PATH, "fasta")},
    'r': {rec.id : rec.seq for rec in SeqIO.parse(R_PATH, "fasta")},
    's': {rec.id : rec.seq for rec in SeqIO.parse(S_PATH, "fasta")}
}
metadata = pd.read_csv(METADATA_PATH)

def pad_tensor_list(tensor_list):
    max_len = len(max(tensor_list, key=len))
    tensor_list_pad = []
    for i, line in enumerate(tensor_list):
        tensor_list_pad.append( torch.nn.functional.pad(tensor_list[i], ( 0, max_len-len(line)), "constant" ) )
    return tensor_list_pad

device = torch.device("cpu") #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # had to remove for memory issued on exxmini
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()
def get_flat_esm2_embedding( seq ):
    batch_labels, batch_strs, batch_tokens = batch_converter( [("protien_name", seq)] ) # in the example, a tuple was passed.
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representation = token_representations[0, 1 : len(seq) + 1].mean(0)
    #! what is sequence vs tokin representatiion
    return torch.flatten(token_representations), sequence_representation

list_encoded_seqs_tokenrep = []
list_encoded_seqs_seqrep = []
for _, row in metadata.iterrows():
    mt, ms = get_flat_esm2_embedding( mrs_dict['m'][row['MTase_Name']] ) 
    rt, rs = get_flat_esm2_embedding( mrs_dict['r'][row['R_gene_Name']] )
    st, ss = get_flat_esm2_embedding( mrs_dict['s'][row['S_gene_Name']] )
    list_encoded_seqs_tokenrep.append( torch.cat((mt, rt, st)) )
    list_encoded_seqs_seqrep.append( torch.cat((ms, rs, ss)) )

list_encoded_seqs_tokenrep_pad = pad_tensor_list(list_encoded_seqs_tokenrep)
list_encoded_seqs_seqrep_pad = pad_tensor_list(list_encoded_seqs_seqrep)

data_tokenrep = torch.stack(list_encoded_seqs_tokenrep_pad)
data_seqrep = torch.stack(list_encoded_seqs_seqrep_pad)

torch.save(data_tokenrep, 'data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt')
torch.save(data_seqrep, 'data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt')

