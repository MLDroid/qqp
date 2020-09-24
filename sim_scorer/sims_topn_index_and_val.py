import numpy as np, json, ast
from time import time
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_id_ques_map(fname):
    with open(fname) as fh:
        id_ques_map = json.load(fh)
    id_ques_map = {int(k): str(v) for k, v in id_ques_map.items()}
    id_ques_map[0] = ''
    print(f'Loaded {max(id_ques_map.keys())} ids to questions from {fname}')
    return id_ques_map


def load_emb(emb_fname):
    sent_embs = np.load(emb_fname)
    print(f'Loaded embeddings of shape: {sent_embs.shape} from {emb_fname}')
    return sent_embs


def main():
    batch_size = 20000
    max_samples = None
    top_n_sim_sents = 100

    emb_fname = '../data/distilbert-base-nli-stsb-mean-tokens_sent_embeddings.npy'

    sent_embs = load_emb(emb_fname)
    sent_embs = sent_embs[:max_samples]
    sent_embs = sent_embs.astype(np.float32) #half precision
    print(f'After reduction, the shape of embedding matrix: {sent_embs.shape}')

    inds = list(range(sent_embs.shape[0]))
    chunked_inds = list(chunks(inds, batch_size))

    t00 = time()
    most_sim_inds = []
    sim_values = []
    for batch_num, batch_inds in enumerate(tqdm(chunked_inds), start=1):
        t0 = time()
        batch_sims = cosine_similarity(sent_embs[batch_inds], sent_embs).astype(np.float32)
        topinds = np.flip(batch_sims.argsort(),1)[:,:top_n_sim_sents]
        sims = np.take_along_axis(batch_sims,topinds,1)
        most_sim_inds.append(topinds)
        sim_values.append(sims)
        batch_time = time()-t0
        print(f'Done with batch: {batch_num} in {batch_time:.2f} sec.')
    our_total_time = time() - t00
    print(f'All batches done in : {our_total_time:.2f} sec.')

    most_sim_inds = np.vstack(most_sim_inds)
    sim_values = np.vstack(sim_values)

    np.save('most_sim_inds.npy',most_sim_inds)
    np.save('sim_values.npy',sim_values)


if __name__ == '__main__':
    main()