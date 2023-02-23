source lib/conda/bin/activate
conda activate methyl_motifs_venv

# hid 5 con 512 seemed to train the best based on earlier experiments

CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 200 --hid 5 --lf mse --lrval 0.0001 --connections 512 --group six_target_options --target_path data/target_methylation_motif_oneside_pseudoonehot_numN_padend.pt --run_name pseudoonehot_numN_padend
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 200 --hid 5 --lf mse --lrval 0.0001 --connections 512 --group six_target_options --target_path data/target_methylation_motif_oneside_pseudoonehot_numN_padmiddle.pt --run_name pseudoonehot_numN_padmiddle
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 200 --hid 5 --lf mse --lrval 0.0001 --connections 512 --group six_target_options --target_path data/target_methylation_motif_oneside_pseudoonehot_padend.pt --run_name pseudoonehot_padend
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 200 --hid 5 --lf mse --lrval 0.0001 --connections 512 --group six_target_options --target_path data/target_methylation_motif_oneside_trueonehot_numN_padend.pt --run_name trueonehot_numN_padend
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 200 --hid 5 --lf mse --lrval 0.0001 --connections 512 --group six_target_options --target_path data/target_methylation_motif_oneside_trueonehot_numN_padmiddle.pt --run_name trueonehot_numN_padmiddle
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 200 --hid 5 --lf mse --lrval 0.0001 --connections 512 --group six_target_options --target_path data/target_methylation_motif_oneside_trueonehot_padend.pt --run_name trueonehot_padend

echo "BATCH SCRIPT FINISHED"
