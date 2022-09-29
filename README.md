# Restriction enzyme study
Can we model the metalation motifs of 600 protiens?
## Setup
The environment is nothing special. It needs to standard PyTorch, Pandas, and Numpy. For easy setup use:
```
scripts/install_conda_env.sh
```

The raw sequences and their responses are located in raw_data. Run this data through ESM-1b:
```
scripts/build_esm1b_data.sh
```

