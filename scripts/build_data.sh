# setting env 
source lib/conda/bin/activate
conda activate metal_motifs_env

# make temp_data directory
#mkdir temp_data *needs to be in root which has issues with 'nohup'

# run splitting script to split each sequence in fasta files into two (sequences are too long to fit in esm1b)
# see https://github.com/brianhie/evolocity/issues/2 for recomendation to split sequence.
echo splitting data
python scripts/splitMRS.py

# run esm script to generate esm1b inferences from bulk fasta files
echo running esm1b
python ../esm/scripts/extract.py esm1b_t33_650M_UR50S temp_data/m1.faa \
  temp_data/m1split --repr_layers 0 32 33 --include mean per_tok --nogpu
python ../esm/scripts/extract.py esm1b_t33_650M_UR50S temp_data/m2.faa \
  temp_data/m2split --repr_layers 0 32 33 --include mean per_tok --nogpu
echo finished m

python ../esm/scripts/extract.py esm1b_t33_650M_UR50S temp_data/r1.faa \
  temp_data/r1split --repr_layers 0 32 33 --include mean per_tok --nogpu
python ../esm/scripts/extract.py esm1b_t33_650M_UR50S temp_data/r2.faa \
  temp_data/r2split --repr_layers 0 32 33 --include mean per_tok --nogpu
echo finished r

python ../esm/scripts/extract.py esm1b_t33_650M_UR50S temp_data/s1.faa \
  temp_data/s1split --repr_layers 0 32 33 --include mean per_tok --nogpu
python ../esm/scripts/extract.py esm1b_t33_650M_UR50S temp_data/s2.faa \
  temp_data/s2split --repr_layers 0 32 33 --include mean per_tok --nogpu
echo finished s

# build input data tensor by taking the individual .pt files for each sequence, 
# pushing them together, padding, and flattening them.
echo building input data
python scripts/build_esm1b_pad_flat.py

# build target data using a psuedo one hot method
echo building target data
python scripts/build_target-motifs_numn.py

# remove temp_data directory
#rm -r temp_data