source lib/conda/bin/activate
conda activate metal_motifs_env

for i in 0 1 2 3 4 5 6 7 8 
do
 echo "hid $i"
 for j in mse split_mse crossent split_crossent
 do
  echo "lossfunc $j"
  CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 5 --epochs 199 --hid $i --lf $j
 done
done
