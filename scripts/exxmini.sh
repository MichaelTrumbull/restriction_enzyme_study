source lib/conda/bin/activate
conda activate metal_motifs_env

for i in 3 4 5 6
do
 echo "hid $i"
 for j in mse crossent
 do
  echo "lossfunc $j"
  for k in 256 512
  do
   echo "con $k"
   CUDA_VISIBLE_DEVICES=1 python model/train.py --batch 3 --epochs 999 --hid $i --lf $j --lrval 0.0001 --connections $k
  done
 done
done
echo "BATCH SCRIPT FINISHED"