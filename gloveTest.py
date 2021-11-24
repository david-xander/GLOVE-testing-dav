import numpy as np
from scipy import spatial
import os
import sys


def find_closest_embeddings(glove_emb_dict, embedding):
    return sorted(glove_emb_dict.keys(), key=lambda token: spatial.distance.euclidean(glove_emb_dict[token], embedding))    

def find_closest_concepts(glove_emb_dict, concepts, num):
        embeddings = glove_emb_dict[concepts[0]]
        for t in concepts:
            if t in glove_emb_dict:
                embeddings = embeddings + glove_emb_dict[t]
        closest_terms = find_closest_embeddings(glove_emb_dict, embeddings)[:num]
        return closest_terms

def load_glove(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            token = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[token] = vector
    return embeddings_dict

def execute_from_command_line(argv=None):
    root_folder = "."
    data_folder_name = "glove.6B"
    DATA_PATH = os.path.abspath(os.path.join(root_folder, data_folder_name))
    glove_filename='glove.6B.300d.txt'    
    glove_path = os.path.abspath(os.path.join(DATA_PATH, glove_filename))

    argv = argv or sys.argv[:]

    general_embeddings_dict = load_glove(glove_path)

    result = find_closest_concepts(general_embeddings_dict, argv[1:], 10)

    for item in result:
        print(item)

if __name__ == "__main__":
    execute_from_command_line()

