source lib/conda/bin/activate
conda activate metal_motifs_env

export CUDA_VISIBLE_DEVICES=0
python model/trainmodel.py --batch 15 --epochs 29
python model/trainmodel.py --batch 30 --epochs 29
python model/trainmodel.py --batch 15 --hid 1 --crossent --epochs 29
python model/trainmodel.py --batch 30 --hid 1 --epochs 29
python model/trainmodel.py --batch 15 --hid 2 --crossent --epochs 29
python model/trainmodel.py --batch 30 --hid 2 --epochs 29
python model/trainmodel.py --batch 15 --crossent --epochs 29
python model/trainmodel.py --batch 15 --mse --epochs 29
python model/trainmodel.py --batch 30 --crossent --epochs 29
python model/trainmodel.py --batch 15 --mse_136 --epochs 29
