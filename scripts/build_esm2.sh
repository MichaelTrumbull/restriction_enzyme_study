source lib/conda/bin/activate
conda activate metal_motifs_env

CUDA_VISIBLE_DEVICES=0 python scripts/build_input_esm2_3B_layer33_1dseq_padlast.py