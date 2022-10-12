'''
This script will take go through each fasta file and split any sequence longer than 1022
into len 1022 and len what-is-left-over.
'''

from Bio import SeqIO

def split(filename, loc):
    hold1 = []
    hold2 = []
    for seq_record in SeqIO.parse(filename, "fasta"):
        if len(seq_record.seq)>1022:
            # split and save into two difference locations
            #SeqIO.write(seq_record[:1022], "temp_data/"+loc+"1.faa", "fasta")
            #SeqIO.write(seq_record[1022:], "temp_data/"+loc+"2.faa", "fasta")
            hold1.append(seq_record[:1022])
            hold2.append(seq_record[1022:])
        else:
            # save original into first location
            #SeqIO.write(seq_record, "temp_data/"+loc+"1.faa", "fasta")
            hold1.append(seq_record)
    SeqIO.write(hold1, "temp_data/"+loc+"1.faa", "fasta")
    SeqIO.write(hold2, "temp_data/"+loc+"2.faa", "fasta")

filenames = ["raw_data/Type_I_M_subunit_genes_Protein_g_clean_sorted.faa", 
    "raw_data/Type_I_R_subunit_genes_Protein_g_clean_sorted.faa", 
    "raw_data/Type_I_S_subunit_genes_Protein_g_clean_sorted.faa"]
ordering = ["m", "r", "s"]
for filename, order in zip(filenames, ordering):
    split(filename, order)