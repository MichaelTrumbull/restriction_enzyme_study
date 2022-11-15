source lib/conda/bin/activate
conda activate methyl_motifs_venv

for i in 3 4 5
do
 echo "hid $i"
 for j in mse
 do
  echo "lossfunc $j"
  for k in 256 512 1024
  do
   echo "con $k"
   CUDA_VISIBLE_DEVICES=0 python model/train.py --batch 3 --epochs 99 --hid $i --lf $j --lrval 0.0001 --connections $k
  done
 done
done
echo "BATCH SCRIPT FINISHED"