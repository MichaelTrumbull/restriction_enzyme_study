source lib/conda/bin/activate
conda activate metal_motifs_env

echo SSSSSSSSSSSSSTTTTTTTTTTTTTTTTTAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRTTTTTTTTTTTTTTTTTTTTTTTTTTTT
echo MSE
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 15 --epochs 199 --hid 0
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 128
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 128
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 128
echo 5
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 128
echo 6
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 128
echo 7
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 128
echo 8
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 128
echo 9
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 128
echo 12
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 64
echo 13
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 64
echo 14
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 64
echo 15
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 64
echo 16
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 64
echo 17
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 64
echo 18
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 64
echo 19
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 64


echo SPLIT MSE
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 15 --epochs 199 --hid 0 --lf split_mse
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 128 --lf split_mse
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 128 --lf split_mse
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 128 --lf split_mse
echo 5
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 128 --lf split_mse
echo 6
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 128 --lf split_mse
echo 7
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 128 --lf split_mse
echo 8
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 128 --lf split_mse
echo 9
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 128 --lf split_mse
echo 12
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 64 --lf split_mse
echo 13
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 64 --lf split_mse
echo 14
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 64 --lf split_mse
echo 15
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 64 --lf split_mse
echo 16
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 64 --lf split_mse
echo 17
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 64 --lf split_mse
echo 18
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 64 --lf split_mse
echo 19
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 64 --lf split_mse


echo CROSSENT
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 15 --epochs 199 --hid 0 --lf crossent
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 128 --lf crossent
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 128 --lf crossent
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 128 --lf crossent
echo 5
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 128 --lf crossent
echo 6
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 128 --lf crossent
echo 7
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 128 --lf crossent
echo 8
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 128 --lf crossent
echo 9
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 128 --lf crossent
echo 12
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 64 --lf crossent
echo 13
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 64 --lf crossent
echo 14
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 64 --lf crossent
echo 15
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 64 --lf crossent
echo 16
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 64 --lf crossent
echo 17
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 64 --lf crossent
echo 18
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 64 --lf crossent
echo 19
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 64 --lf crossent

echo SPLIT CROSSENT
echo 1
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 15 --epochs 199 --hid 0 --lf split_crossent
echo 2
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 128 --lf split_crossent
echo 3
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 128 --lf split_crossent
echo 4
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 128 --lf split_crossent
echo 5
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 128 --lf split_crossent
echo 6
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 128 --lf split_crossent
echo 7
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 128 --lf split_crossent
echo 8
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 128 --lf split_crossent
echo 9
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 128 --lf split_crossent
echo 12
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 1 --connections 64 --lf split_crossent
echo 13
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 2 --connections 64 --lf split_crossent
echo 14
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 3 --connections 64 --lf split_crossent
echo 15
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 4 --connections 64 --lf split_crossent
echo 16
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 5 --connections 64 --lf split_crossent
echo 17
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 6 --connections 64 --lf split_crossent
echo 18
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 7 --connections 64 --lf split_crossent
echo 19
CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid 8 --connections 64 --lf split_crossent
