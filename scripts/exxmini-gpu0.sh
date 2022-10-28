source lib/conda/bin/activate
conda activate metal_motifs_env

echo Testing layers

echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 15 --epochs 99 --hid 0
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 1 --connections 128
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 2 --connections 128
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 3 --connections 128
echo 5
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 4 --connections 128
echo 6
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 5 --connections 128
echo 7
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 6 --connections 128
echo 8
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 7 --connections 128
echo 9
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 8 --connections 128
echo 12
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 1 --connections 64
echo 13
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 2 --connections 64
echo 14
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 3 --connections 64
echo 15
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 4 --connections 64
echo 16
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 5 --connections 64
echo 17
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 6 --connections 64
echo 18
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 7 --connections 64
echo 19
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 8 --connections 64

echo Testing loss funcs
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 2 --connections 128 --lf split_mse
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 3 --connections 128 --lf split_mse
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 4 --connections 128 --lf split_mse
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 5 --connections 128 --lf split_mse
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 2 --connections 128 --lf crossent
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 3 --connections 128 --lf crossent
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 4 --connections 128 --lf crossent
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 5 --connections 128 --lf crossent
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 2 --connections 128 --lf split_crossent
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 3 --connections 128 --lf split_crossent
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 4 --connections 128 --lf split_crossent
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 5 --connections 128 --lf split_crossent
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 2 --connections 64 --lf split_mse
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 3 --connections 64 --lf split_mse
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 4 --connections 64 --lf split_mse
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 99 --hid 5 --connections 64 --lf split_mse