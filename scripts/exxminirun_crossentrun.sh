source lib/conda/bin/activate
conda activate methyl_motifs_venv

# hid 5 con 512 seemed to train the best based on earlier experiments

CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 100 --hid 5 --lf crossent_1 --lrval 0.0001 --connections 512 --group crossent_test_1and15 --target_path data/target_methylation_motif_oneside_pseudoonehot_padend.pt --run_name pseudoonehot_crossent
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 100 --hid 5 --lf crossent_15 --lrval 0.0001 --connections 512 --group crossent_test_1and15 --target_path data/target_methylation_motif_oneside_trueonehot_padend.pt --run_name trueonehot_crossent

echo "BATCH SCRIPT FINISHED"
