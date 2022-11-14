# install miniconda into local \lib. 
# run using sudo command ?
# need to mkdir runs lib data temp_data
wget -P /temp_data "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash temp_data/Miniconda3-latest-Linux-x86_64.sh -b -p lib/conda
rm temp_data/Miniconda3-latest-Linux-x86_64.sh

# set up conda env
export PATH=lib/conda/bin:$PATH
conda env create -f scripts/environment.yml
