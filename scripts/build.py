'''
Script to build torch tensors for: Type I RMS metalation motifs
Before starting: 
    Use ESM-b1 language model to convert 'm', 'r', and 's' FASTA files to richer data tensor
    ESM-b1: https://github.com/facebookresearch/esm *Should use the script provided for conversion
    Need metalation motifs in csv form


'''

BUILD_ESMB1_PAD_FLAT = False # each esmb1 tensor is flattened and padded to get a 2d tensor
BUILD_ESMB1_PAD_NOFLAT = False # each esmb1 tensor is padded to create a 3d tensor
BUILD_MOTIFS_4BASE_NUMN = False # target: speudo one hot (four bits {ACGT} mutually exclusive, but could target A and C at the same time{not orthonormal})

BUILD_PARTIAL = False # when any of the builds run it will also save a tensor with len=2 (instead of len=600) for trial runs on personal computer

INPUT_DIRS = [ "m1split/", "m2split/", "r1split/", "r2split/", "s1split/", "s2split/" ]
TARGET = "metadata_600_MODIFIED.csv"

import os
if not os.path.exists("data/"): os.mkdir("data/") # we will save the outputs to this directory

if BUILD_ESMB1_PAD_FLAT:
    print('RUNNING BUILD_ESMB1_PAD_FLAT')
    '''
    - Data came from Dr. Gang Fang
    - Data was 3 files of input data (split into M R and S) and 1 target file (only 600 seq long containing metalation motifs)
    - M R and S sequence lengths were too big (>1024) for ESM-b1 protien language model processing so these files were split into 2
    - prebuilt script to translate seq's into language model tensors
        - python scripts/extract.py esm1b_t33_650M_UR50S filename.faa newsavelocation/ --repr_layers 0 32 33 --include mean per_tok
    '''
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np

    # .pt files built using premade script: python scripts/extract.py esm1b_t33_650M_UR50S filename.faa newsavelocation/ --repr_layers 0 32 33 --include mean per_tok
    # .pt files were generated for each enzyme for each file
    m1s = []
    m2s = []
    r1s = []
    r2s = []
    s1s = []
    s2s = []
    directory = INPUT_DIRS
    for filename in os.listdir(directory[0]):
        f = os.path.join(directory[0], filename)
        if f[-3:len(f)] == ".pt":
            m1s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[1]):
        f = os.path.join(directory[1], filename)
        if f[-3:len(f)] == ".pt":
            m2s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[2]):
        f = os.path.join(directory[2], filename)
        if f[-3:len(f)] == ".pt":
            r1s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[3]):
        f = os.path.join(directory[3], filename)
        if f[-3:len(f)] == ".pt":
            r2s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[4]):
        f = os.path.join(directory[4], filename)
        if f[-3:len(f)] == ".pt":
            s1s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[5]):
        f = os.path.join(directory[5], filename)
        if f[-3:len(f)] == ".pt":
            s2s.append(torch.load(f)["representations"][33])

    print("Checking torch.cat on [0], [1], [2]")
    i = 0
    print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())
    i = 1
    print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())
    i = 2
    print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())

    print("make data_list by: data_list.append(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) )) ")
    data_list = []
    for i in range(len(m1s)):
        data_list.append(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ))
    
    #### flatten
    data_list_flat = []
    for line in data_list:
        data_list_flat.append(torch.flatten(line))
    print("data_list_flat[0].size()", data_list_flat[0].size())

    #### get padding size
    len_list = []
    for list in data_list_flat:
        len_list.append(len(list))
    max_len_val = max(len_list)
    print("max_len_val", max_len_val)
    
    #### PAD

    data_list_flat_padded = []
    for i, line in enumerate(data_list_flat):
        data_list_flat_padded.append( F.pad(data_list_flat[i], ( max_len_val-len(line), 0), "constant" ) )
    print("data_list_flat_padded[0].size()", data_list_flat_padded[0].size())

    #### build list of tensors into single torch tensor
    print("torch stacking the list of tensors into single tensor")
    data = torch.stack(data_list_flat_padded)

    print("data.size()", data.size())

    print('saving')
    torch.save(data, "data/msr-esmb1-flat.pt")
    if BUILD_PARTIAL: torch.save(data[0:2], "data/PARTIALDATAmsr-esmb1-flat.pt")



if BUILD_ESMB1_PAD_NOFLAT:
    '''
    - Data came from Dr. Gang Fang
    - Data was 3 files of input data (split into M R and S) and 1 output file (only 600 seq long containing metalation motifs)
    - M R and S sequence lengths were too big (>1024) for ESM-b1 protien language model processing
    - Therefore these files were split into 2
    - prebuilt script to translate seq's into language model tensors
        - python scripts/extract.py esm1b_t33_650M_UR50S filename.faa newsavelocation/ --repr_layers 0 32 33 --include mean per_tok
    '''
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np

    # .pt files built using premade script: python scripts/extract.py esm1b_t33_650M_UR50S filename.faa newsavelocation/ --repr_layers 0 32 33 --include mean per_tok
    # .pt files were generated for each enzyme for each file
    m1s = []
    m2s = []
    r1s = []
    r2s = []
    s1s = []
    s2s = []
    directory = INPUT_DIRS
    for filename in os.listdir(directory[0]):
        f = os.path.join(directory[0], filename)
        if f[-3:len(f)] == ".pt":
            m1s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[1]):
        f = os.path.join(directory[1], filename)
        if f[-3:len(f)] == ".pt":
            m2s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[2]):
        f = os.path.join(directory[2], filename)
        if f[-3:len(f)] == ".pt":
            r1s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[3]):
        f = os.path.join(directory[3], filename)
        if f[-3:len(f)] == ".pt":
            r2s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[4]):
        f = os.path.join(directory[4], filename)
        if f[-3:len(f)] == ".pt":
            s1s.append(torch.load(f)["representations"][33])
    for filename in os.listdir(directory[5]):
        f = os.path.join(directory[5], filename)
        if f[-3:len(f)] == ".pt":
            s2s.append(torch.load(f)["representations"][33])


    print("Checking torch.cat on [0], [1], [2]")
    i = 0
    print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())
    i = 1
    print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())
    i = 2
    print(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ).size())

    print("make data_list by: data_list.append(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) )) ")
    data_list = []
    for i in range(len(m1s)):
        data_list.append(torch.cat( ( m1s[i], m2s[i], r1s[i], r2s[i], s1s[i], s2s[i] ) ))

    #### get padding size
    hold = []
    for list in data_list:
        hold.append(len(list))
    max_len = max(hold)
    print("max len in data list (for padding): ", max_len)
    print("data_list[0].size() :", data_list[0].size())


    data_list_padded = []
    for i, line in enumerate(data_list):
        data_list_padded.append( F.pad(data_list[i], ( 0, 0, 0, max_len-len(line)), "constant" ) )
    print("data_list_padded[0].size() :", data_list_padded[0].size())

    print("torch stacking the list of tensors into single tensor")
    data = torch.stack(data_list_padded) ### !!!! NEED TO PAD (ALL SAME SIZE) FOR STACKING TO WORK


    print("data.size()", data.size())

    print('saving')
    torch.save(data, "data/msr-esmb1.pt")

    if BUILD_PARTIAL: torch.save(data[0:2], "data/PARTIALDATAmsr-esmb1.pt")


if BUILD_MOTIFS_4BASE_NUMN:
    import numpy as np
    import torch
    import pandas as pd
    ###### import data as pandas dataframe
    MOTIFS = TARGET
    df = pd.read_csv(MOTIFS)
    ###### set up directory of how each symbol connects to ACGT
    # http://www.hgmd.cf.ac.uk/docs/nuc_lett.html
    bas4_dict = {
        'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1],
        'R': [1,0,1,0], 'Y': [0,1,0,1], 'K': [0,0,1,1], 'M': [1,1,0,0], 'S': [0,1,1,0], 'W': [1,0,0,1],
        'B': [0,1,1,1], 'D': [1,0,1,1], 'H': [1,1,0,1], 'V': [1,1,1,0], 'N': [1,1,1,1]
    }
    # mapping 'N': [1,1,1,1] because it should accept any base target. Could be [0,0,0,0] because this is just padding...
    def str_to_base4code(string):
        coded = []
        for char in string:
            coded = coded + bas4_dict[char]
        return coded
    ###### check the data structure
    motifs_pd = df
    print(motifs_pd.head())
    ###### functions for building list of encoded strings and padding the lists
    def one_hot(data_1d):
        data_1d_encoded = []
        for string in data_1d:
            data_1d_encoded.append(str_to_base4code(string))
        return data_1d_encoded
    def pad(data_1d):
        max_length = len(max(data_1d, key=len))
        padded_data = []
        for i, line in enumerate(data_1d):
            padded_data.append(line + [0]*(max_length - len(line)))
        return padded_data
    ###### transitioning from pandas dataframe to list of the feature we want
    # 600: Do they want to keep their own testing data away from us for when we give them the model?
    motif11 = []#first half, left of /
    motif12 = []#first half, right of /
    for line in motifs_pd['Motif_1stHalf'].to_list():
        motif11.append(line.rsplit('/')[0])
        try:
            motif12.append(line.rsplit('/')[1])
        except:
            motif12.append('')
            print(line, " has no slash 1")
    motif21 = []
    motif22 = []

    for line in motifs_pd['Motif_2ndHalf'].to_list():
        motif21.append(line.rsplit('/')[0])
        try: # sometimes there is no / present
            motif22.append(line.rsplit('/')[1])
        except:
            motif22.append('')
            print(line, " has no slash 2")
    ###### encoding and then padding each list sequences
    motif11 = one_hot(motif11)
    motif11 = pad(motif11)
    motif12 = one_hot(motif12)
    motif12 = pad(motif12)
    motif21 = one_hot(motif21)
    motif21 = pad(motif21)
    motif22 = one_hot(motif22)
    motif22 = pad(motif22)
    ###### one hot encode the number of Ns for the spacing
    # compare 'N's on left / right half. double check they are even
    same_on_both_sides = True
    for line in motifs_pd['Methylation Motif']:
        try:
            a,b = line.split("/")
            if int(a.count('N')) != int(b.count('N')):
                same_on_both_sides = False
        except:
            pass
    print("Ns are same on both sides? ", same_on_both_sides)

    # Get number of Ns into a one hot
    num_ns = []
    for line in motifs_pd['Methylation Motif']:
        num_ns.append(int(line.count('N')/2))

    num_ns_onehot = []
    for n in num_ns:
        num_ns_onehot.append([int(i) for i in np.eye(9)[n-2].tolist()])
    print(len(num_ns))
    ######
    # bring together one hot version of motif 1st half part 1, motif 1st half part 2, motif 2nd half part 1, motif 2nd half part 2, and number of spaces (n) 
    motifs_onehot = []
    for a, b, c, d, n in zip(motif11, motif12, motif21, motif22, num_ns_onehot):
        motifs_onehot.append(a + b + c + d + n)

    torch.save(torch.FloatTensor(motifs_onehot), "data/motifs-base4-numN.pt")
    print(len(motifs_onehot[0]))
    ######
    if BUILD_PARTIAL: torch.save(torch.FloatTensor(motifs_onehot)[0:2], "data/PARTIALDATAmotifs-base4-numN.pt")
    ######
    hold = torch.load("data/motifs-base4-numN.pt")
    print("data.size()", hold.size())
        
