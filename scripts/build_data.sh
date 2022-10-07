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
python ../esm-modified_script/scripts/extract.py esm1b_t33_650M_UR50S temp_data/m1.faa \
  temp_data/m1split --repr_layers 0 32 33 --include mean per_tok --nogpu
python ../esm-modified_script/scripts/extract.py esm1b_t33_650M_UR50S temp_data/m2.faa \
  temp_data/m2split --repr_layers 0 32 33 --include mean per_tok --nogpu
echo finished m

python ../esm-modified_script/scripts/extract.py esm1b_t33_650M_UR50S temp_data/r1.faa \
  temp_data/r1split --repr_layers 0 32 33 --include mean per_tok --nogpu
python ../esm-modified_script/scripts/extract.py esm1b_t33_650M_UR50S temp_data/r2.faa \
  temp_data/r2split --repr_layers 0 32 33 --include mean per_tok --nogpu
echo finished r

python ../esm-modified_script/scripts/extract.py esm1b_t33_650M_UR50S temp_data/s1.faa \
  temp_data/s1split --repr_layers 0 32 33 --include mean per_tok --nogpu
python ../esm-modified_script/scripts/extract.py esm1b_t33_650M_UR50S temp_data/s2.faa \
  temp_data/s2split --repr_layers 0 32 33 --include mean per_tok --nogpu
echo finished s

# build input and target data
echo building target data
python scripts/build_input_and_target.py

# remove temp_data directory
#rm -r temp_data
