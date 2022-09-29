# Restriction enzyme study
Can we model the metalation motifs of 600 protiens?
## Setup
The environment is nothing special. It needs to standard PyTorch, Pandas, and Numpy. For easy setup use:
```bash
scripts/install_conda_env.sh
```

The raw sequences and their responses are located in raw_data. Run this data through ESM-1b:
```bash
scripts/build_esm1b_data.sh
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
