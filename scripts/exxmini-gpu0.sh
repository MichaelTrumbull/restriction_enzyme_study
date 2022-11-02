source lib/conda/bin/activate
conda activate metal_motifs_env

for i in 2 3 4 5
do
 echo "hid $i"
 for j in mse split_mse crossent split_crossent
 do
  echo "lossfunc $j"
  for k in 0.001 0.0001 0.00001
  do
   echo "lrval $k"
   CUDA_VISIBLE_DEVICES=1 python model/train.py --batch 5 --epochs 250 --hid $i --lf $j --lrval $k
  done
 done
done
