source lib/conda/bin/activate
conda activate methyl_motifs_venv

python scripts/build_data/Methylation_Motif_oneside.py
python scripts/build_data/Methylation_Motif_padlast.py
python scripts/build_data/Methylation_Motif_padmiddle.py
python scripts/build_data/motif1stHalf2ndHalf_padmiddle_numN.py

python scripts/build_data/esm2_3B_avg.py
python scripts/build_data/esm2_15B_avg.py