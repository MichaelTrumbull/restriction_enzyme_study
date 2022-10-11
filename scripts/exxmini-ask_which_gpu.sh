source lib/conda/bin/activate
conda activate metal_motifs_env

nvidia-smi # see what gpu is avail

echo Which GPU is available? Use integer value. Empty to cancel run
read gpu_number

if [[ "$gpu_number" =~ ^[0-3]+$ ]]
then
  echo Running trainmodel.py with CUDA:$gpu_number
  export CUDA_VISIBLE_DEVICES=$gpu_number
  python model/trainmodel.py --batch 6 --hid 1 --epochs 99 --connections 1024
else
  echo CANCELLED
fi
