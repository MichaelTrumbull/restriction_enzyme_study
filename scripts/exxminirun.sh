source lib/conda/bin/activate
conda activate methyl_motifs_venv

# Tests: compare old targets vs new "oneside" target; test new split cross entropy (after diagnosing what is going on with cross entropy); test variables widely while looking as how validation does
for i in 5
do
 echo "hid $i"
 for j in mse
 do
  echo "lossfunc $j"
  for k in 512 1024 2048
  do
   echo "con $k"
   CUDA_VISIBLE_DEVICES=1 python model/train.py --batch 3 --epochs 999 --hid $i --lf $j --lrval 0.0001 --connections $k --group train-validation_second_attempt
  done
 done
done
echo "BATCH SCRIPT FINISHED"

