# Restriction enzyme study
Can we model the metalation motifs of 600 protiens?
## Setup
***scripts might require elevated privileges***
The environment is nothing special. It needs to standard PyTorch, Pandas, and Numpy. For easy setup use:
```bash
scripts/install_conda_env.sh
```

Activate this environment with
```bash
scripts/activate_conda_env.sh
```

The raw sequences and their responses are located in raw_data. To prepare the input data use
```bash
scripts/build_esm1b_data.sh
```
*Note that this script is currently flagged to NOT use the GPU* This splits raw_data into esm1b-able chunks, runs esm1b on each sequence, then pieces it all together into a single input tensor.

To build the target data using pseudo-one-hot encoding (A -> [1,0,0,0], G -> [0,0,1,0], R -> [1,0,1,0]) and also the size of the gap between sequence locations.
```bash
scripts/build_target-motifs_numn.py
```

## Training
Multiple networks are availible for training on with auto cuda detection. The differences between networks are in depth and shape but all are simple neural nets.
### HPC
Slurm scripts are provided in `scripts/slurm_scripts`.
### Regular
To run the training model use:
```bash
model/trainmodel.py --asdf
```
## Next Steps
- Should try with MSA. 
- Use HMMER then esmb1 before putting it through network

## Other
For poor remote connection use 
```bash
nohup
```
so if a disconnection occurs it won't effect the shell script run.