import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import os

OUTPUT_DIR = "data/"
INPUT_DIR = "raw_data/"

model, alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
model.eval()

if torch.cuda.is_available(): model = model.cuda()

def run_bulk_esm1b():






fasta_files = []
for filename in os.listdir(INPUT_DIR):
    if filename[-4:-1] == ".fa": fasta_files.append(INPUT_DIR+filename)
print(fasta_files)
