source lib/conda/bin/activate
conda activate metal_motifs_env

CUDA_VISIBLE_DEVICES=1 python model/train.py --batch 15 --epochs 1 --hid 0
