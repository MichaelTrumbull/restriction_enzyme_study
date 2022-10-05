source lib/conda/bin/activate
conda activate metal_motifs_env

nvidia-smi # see what gpu is avail

echo Which GPU is available?
echo Integer value:
read gpu_number
echo Running trainmodel.py with CUDA:$gpu_number

#python model/trainmodel.py --flags