source lib/conda/bin/activate
conda activate metal_motifs_env

for i in 3 4 5 6
do
 echo "hid $i"
 for j in mse crossent
 do
  echo "lossfunc $j"
  for k in 0.0001
  do
   echo "lrval $k"
   CUDA_VISIBLE_DEVICES=1 python model/train.py --batch 5 --epochs 999 --hid $i --lf $j --lrval $k
  done
 done
done
echo "BATCH SCRIPT FINISHED"