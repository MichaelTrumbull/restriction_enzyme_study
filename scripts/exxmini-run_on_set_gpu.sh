source lib/conda/bin/activate
conda activate metal_motifs_env

export CUDA_VISIBLE_DEVICES=0
python model/trainmodel.py --batch 6
python model/trainmodel.py --batch 12
python model/trainmodel.py --batch 6 --hid 1
python model/trainmodel.py --batch 6 --hid 2
python model/trainmodel.py --batch 6 --target_path data/Motif_1stHalf_Motif_2ndHalf_numN.pt