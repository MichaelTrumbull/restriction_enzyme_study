source lib/conda/bin/activate
conda activate metal_motifs_env

CUDA_VISIBLE_DEVICES=0 python model/test_cuda_error.py
