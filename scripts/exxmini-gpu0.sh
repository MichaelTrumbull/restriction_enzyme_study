source lib/conda/bin/activate
conda activate metal_motifs_env

CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padmiddle.pt
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padmiddle.pt

CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padmiddle.pt
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padmiddle.pt


CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padlast.pt
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padlast.pt

CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padlast.pt
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padlast.pt

CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt

CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt
