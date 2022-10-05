source lib/conda/bin/activate
conda activate metal_motifs_env

nvidia-smi # see what gpu is avail

echo Which GPU is available? Use integer value. Empty to cancel run
read gpu_number

if ((gpu_number >= 0 && gpu_number <= 3))
then
  echo Running trainmodel.py with CUDA:$gpu_number
  #python model/trainmodel.py --flags
else
  echo Run cancelled
fi
