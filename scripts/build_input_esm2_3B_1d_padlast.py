'''
- This script builds esm2(3B) input data used in training. 
- From metadata_600 we get the protiens we are using (600 of them) and the order to place them.
- Target data will be built from metadata_600 using the same protiens and ordering.
- Data will be a 2d torch tensor
    - ESM2 creates a 2d tensor per sequence
    - This tensor will be flattened so each sequence is a 1d tensor
    - These 1d tensors will be stacked to our final 2d tensor
        - Note: this data will be padded AFTER the m,r, and s tokins are combined.

'''

from webbrowser import get
import torch
import pandas as pd
from Bio import SeqIO
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
'''
M_PATH = "raw_data/Type_I_M_subunit_genes_Protein_g_clean_sorted.faa"
R_PATH = "raw_data/Type_I_R_subunit_genes_Protein_g_clean_sorted.faa"
S_PATH = "raw_data/Type_I_S_subunit_genes_Protein_g_clean_sorted.faa"
METADATA_PATH = "raw_data/metadata_600.csv"
'''
#tmp
M_PATH = "test_data/Type_I_M_subunit_genes_Protein_g_clean_sorted.faa"
R_PATH = "test_data/Type_I_R_subunit_genes_Protein_g_clean_sorted.faa"
S_PATH = "test_data/Type_I_S_subunit_genes_Protein_g_clean_sorted.faa"
METADATA_PATH = "test_data/metadata_600.csv"
#

def get_flat_esm2_embedding( seq ):
    return torch.tensor([1,2,3]) #tmp

mrs_dict = {
    'm': {rec.id : rec.seq for rec in SeqIO.parse(M_PATH, "fasta")},
    'r': {rec.id : rec.seq for rec in SeqIO.parse(R_PATH, "fasta")},
    's': {rec.id : rec.seq for rec in SeqIO.parse(S_PATH, "fasta")}
}

metadata = pd.read_csv(METADATA_PATH)

list_encoded_seqs = []
for _, row in metadata.iterrows():
    print(row['MTase_Name'], mrs_dict['m'][row['MTase_Name']][0:10] )
    m = get_flat_esm2_embedding( mrs_dict['m'][row['MTase_Name']] )
    print(row['R_gene_Name'], mrs_dict['r'][row['R_gene_Name']][0:10] )
    r = get_flat_esm2_embedding( mrs_dict['r'][row['R_gene_Name']] )
    print(row['S_gene_Name'], mrs_dict['s'][row['S_gene_Name']][0:10] )
    s = get_flat_esm2_embedding( mrs_dict['s'][row['S_gene_Name']] )
    list_encoded_seqs.append( torch.cat((m, r, s)) )
print(list_encoded_seqs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
