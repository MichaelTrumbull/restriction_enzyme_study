source lib/conda/bin/activate
conda activate metal_motifs_env

export CUDA_VISIBLE_DEVICES=0
python model/trainmodel.py --batch 15
python model/trainmodel.py --batch 30
python model/trainmodel.py --batch 15 --hid 1 --crossent
python model/trainmodel.py --batch 30 --hid 1
python model/trainmodel.py --batch 15 --hid 2 --crossent
python model/trainmodel.py --batch 30 --hid 2
python model/trainmodel.py --batch 15 --crossent
python model/trainmodel.py --batch 15 --mse
python model/trainmodel.py --batch 30 --crossent
python model/trainmodel.py --batch 15 --mse_136
