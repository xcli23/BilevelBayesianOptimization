from collections import defaultdict
from attack_util import *

def get_amino_info():
    name2alph = {}
    alph2name = {}  
    with open('attack_codes/amino', "r") as f:
        lines = f.readlines()
        for line in lines:
            name, alphabet = line[:-1].split(" ")
            name2alph[name] = alphabet
            alph2name[alphabet] = name
    return name2alph, alph2name

def get_dist_mat():
    with open('attack_codes/dist', "r") as f:
        lines = f.readlines()
        first_line = lines[0]
        row_name = first_line[1:-1].split("\t")
        column_name = []
        dist_mat = defaultdict(dict)

        for line in lines[1:]:
            line_split = line[:-1].split("\t")
            column_name.append(line_split[0])
            for i, num in enumerate(line_split[1:-1]):
                col = line_split[0]
                row = row_name[i]
                if row == col:
                    dist_mat[row][col] = 0.0
                elif num == '.':
                    dist_mat[row][col] = float('inf')
                else:
                    dist_mat[row][col] = int(num)/1000
    return dist_mat

def get_synonym(vocab, WORKING_FOLDER):

    syndict = defaultdict(list)
    class_num = num_category[str(WORKING_FOLDER)]

    for v,k in vocab.itos.items():
        my_list = [i for i in range(class_num)]
        my_list.remove(k)
        my_list.insert(0,k)
        syndict[v] = my_list

    return syndict

if __name__ == '__main__':
    from fastai.text import Vocab
    import numpy as np
    
    tok_itos = np.load('protein_codes/datasets/clas_ec/clas_ec_ec50_level1/tok_itos.npy',allow_pickle=True)
    itos={i:x for i,x in enumerate(tok_itos)}
    vocab=Vocab(itos)

    syndict= get_synonym(vocab)
    for key, l_ in syndict.items():
        print(key, l_)
   