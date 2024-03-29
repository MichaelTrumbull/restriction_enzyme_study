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
Install ESM scripts outside of ***restriction_enzyme_study*** to run inference on our raw data to set up for training. This code is modified from facebook/esm because of an error thrown by passing too many arguments in their premade scripts we are using. This code is minimally modified.
```bash
git clone https://github.com/MichaelTrumbull/esm-modified_script.git
```
To build data for training from the raw_data run
```bash
scripts/build_data.sh
```
This script will run 
- splitMRS.py (Splits the raw data into esm1b-able chunks. esm1b cannot use >1022 length seq)
- ../esm-modified_script/scripts/extract.py (runs esm1b inference on each sequence *Note that this script is currently flagged to NOT use the GPU*)
- build_esm1b_pad_flat.py (combines the tokened sequences into a sinlge input tensor)
- build_target-motifs_numn.py (builds the target data using pseudo-one-hot encoding (A -> [1,0,0,0], G -> [0,0,1,0], R -> [1,0,1,0]) and also the size of the gap between sequence locations)

### Test data was built correctly
Run this script to print out the input and target data tensor shapes to check the build ran correctly.
```bash
scripts/test_data.py
```

## Training
Multiple networks are availible for training on with auto cuda detection. The differences between networks are in depth and shape but all are simple neural nets.
### Data
Training input and target data are built from the raw_data files. Typically, the input data uses Facebook's ESM protien language models and target data is one-hot encoded (sort of). The following variations are:
- input_esm2_3B_layer33_1dseq_padlast_tokenrepresentation
    - ESM2 (3B) inference is used on each sequence. Layer 33 is extracted from the model's activation. This layer is flattened to be a 1d tensor representing each seq. 
- 
### HPC
Slurm scripts are provided in `scripts/slurm_scripts`.
Check which GPUs are available: `nvidia-smi`
### Regular
To run the training model use:
```bash
model/trainmodel.py --asdf
```
## Next Steps
- Should try with MSA. 
- Use HMMER then esmb1 before putting it through network
Build modifications:
- the target data represents the number of spaces between specified residues as a number at the end. What if I don't do that and just encode each N as [1,1,1,1]. 


## Other
For poor remote connection use 
```bash
nohup cmd &
```
so if a disconnection occurs it won't effect the shell script run.