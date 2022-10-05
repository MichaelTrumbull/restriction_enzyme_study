source /data/mjt2211/restriction_enzyme_study/lib/conda/bin/activate
conda activate metal_motifs_env
which python

python ../esm/scripts/extract.py esm1b_t33_650M_UR50S test-esm/raw_data/testing.faa test-esm/data/m1split --repr_layers 0 32 33 --include mean per_tok --nogpu
