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


def compute_diff(sent_embs, our_total_time, sim_mat):
    t0 = time()
    actual_cs = cosine_similarity(sent_embs)
    actual_cs_time = time() - t0
    print(f'Actual cosine similarity computation took {actual_cs_time:.2f} sec.')
    diff_time = actual_cs_time - our_total_time
    print(f'Our improvement: {diff_time:.2f} sec.')

    diff_sum = sum(abs(sim_mat - actual_cs)).sum()
    print(f'Sum of the difference: {diff_sum}')


def main():
    batch_size = 20000
    max_samples = None
    top_n_sim_sents = 100

    emb_fname = '../data/distilbert-base-nli-stsb-mean-tokens_sent_embeddings.npy'

    sent_embs = load_emb(emb_fname)
    sent_embs = sent_embs[:max_samples]
    print(f'After reduction, the shape of embedding matrix: {sent_embs.shape}')

    inds = list(range(sent_embs.shape[0]))
    chunked_inds = list(chunks(inds, batch_size))
    num_batches = sent_embs.shape[0]//batch_size
    print(f'Total number of batches (floor): {num_batches}')

    for batch_num, batch_inds in enumerate(tqdm(chunked_inds), start=1):
        batch_sims = cosine_similarity(sent_embs[batch_inds], sent_embs)
        batch_sims = batch_sims.astype(np.float32)

        topinds = np.flip(batch_sims.argsort(), 1)[:, :top_n_sim_sents]
        sims = np.take_along_axis(batch_sims, topinds, 1)

        # type cast
        topinds = topinds.astype(np.int32)
        sims = sims.astype(np.float32)

        # save indexes that are similar to a given sent id
        fname_1 = f'/vol1/most_sim_inds_{batch_num}.npy'
        np.save(fname_1, topinds)

        # save sim scores - to be used for threshold
        fname_2 = f'/vol1/sim_values_{batch_num}.npy'
        np.save(fname_2, sims)


if __name__ == '__main__':
    main()