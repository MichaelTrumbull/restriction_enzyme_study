source lib/conda/bin/activate
conda activate metal_motifs_env

nvidia-smi # see what gpu is avail

echo Which GPU is available? Use integer value. Empty to cancel run
read gpu_number
echo Running trainmodel.py with CUDA:$gpu_number

if ((number >= 0 && number <= 3))
then
  #python model/trainmodel.py --flags
  echo finished
else
  echo Run cancelled
fi
