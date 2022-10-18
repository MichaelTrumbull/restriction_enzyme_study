source lib/conda/bin/activate
conda activate metal_motifs_env

echo 1
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padmiddle.pt
echo 2
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padmiddle.pt


echo 3 - reduced batch size
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 5 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padmiddle.pt
echo not running 4 - batch cant go smaller
#CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padmiddle.pt




echo 5
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padlast.pt
echo 6
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_Methylation_Motif_padlast.pt


echo 7 - reduced batch size
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 5 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padlast.pt
echo not running 8 - batch cant go smaller
#CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_Methylation_Motif_padlast.pt


echo 9
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 15 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt
echo 10
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_sequencerep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt


echo 11 - reduce batch size
CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 5 --epochs 19 --hid 0 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt
echo not running 12 - batch cant go smaller
#CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py --batch 3 --epochs 19 --hid 1 --input_path data/input_esm2_3B_layer33_1dseq_padlast_tokenrep.pt --target_path data/target_motif1stHalf2ndHalf_padmiddle_numN.pt


# testing github.dev