import numpy as np, json
from time import time
import spacy
from tqdm import tqdm


def load_id_ques_map(fname):
    with open(fname) as fh:
        id_ques_map = json.load(fh)
    id_ques_map = {int(k): str(v) for k, v in id_ques_map.items()}
    id_ques_map[0] = ' '
    print(f'Loaded {max(id_ques_map.keys())} ids to questions from {fname}')
    return id_ques_map


def main():
    ip_map_fname = 'data/id_ques_map.json'
    op_save_fname = 'data/spacy_w2v_avg_embeddings.npy'

    nlp = spacy.load("en_core_web_sm")

    id_ques_map = load_id_ques_map(ip_map_fname)

    sents = [str(id_ques_map[i])for i in range(max(id_ques_map)+1)]#[:100]

    doc_w2v_avg_embs = []
    for s in tqdm(sents):
        doc = nlp(s)
        doc_vec = doc.vector
        doc_w2v_avg_embs.append(doc_vec)

    doc_w2v_avg_embs = np.array(doc_w2v_avg_embs)
    print(f'Shape of spacy vectors: {doc_w2v_avg_embs.shape}')

    np.save(op_save_fname, doc_w2v_avg_embs)
    print(f'File saved: {op_save_fname}')


if __name__ == '__main__':
    main()