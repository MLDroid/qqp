import numpy as np, json
from time import time
from sklearn.metrics.pairwise import cosine_similarity


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
    batch_size = 5000
    max_samples = None

    emb_fname = '../data/distilbert-base-nli-stsb-mean-tokens_sent_embeddings.npy'
    sim_mat_save_fname = '/vol1/distilbert_similarities_.npy'

    sent_embs = load_emb(emb_fname)
    sent_embs = sent_embs[:max_samples]
    print(f'After reduction, the shape of embedding matrix: {sent_embs.shape}')

    inds = list(range(sent_embs.shape[0]))
    chunked_inds = chunks(inds, batch_size)
    num_batches = sent_embs.shape[0]//batch_size
    print(f'Total number of batches (floor): {num_batches}')

    t00 = time()
    sims_list = []
    for batch_num, batch_inds in enumerate(chunked_inds, start=1):
        t0 = time()
        batch_sims = cosine_similarity(sent_embs[batch_inds], sent_embs)
        batch_sims = batch_sims.astype(np.float32)
        sims_list.append(batch_sims)
        batch_time = time()-t0
        print(f'Done with batch: {batch_num} in {batch_time:.2f} sec.')
        if batch_num == 1:
            approx_total_time = num_batches * batch_time
            print(f'***Approx total time required: {approx_total_time:.2f} sec***')
    our_total_time = time() - t00
    print(f'All batches done in : {our_total_time:.2f} sec.')


    sim_mat = np.array(sims_list, dtype=np.float32)
    print(f'Original shape of batched sim matrix: {sim_mat.shape}')
    sim_mat = sim_mat.reshape(-1, sent_embs.shape[0])
    print(f'After reshaping,  shape of batched sim matrix: {sim_mat.shape}')


    t0 = time()
    np.save(sim_mat_save_fname, sim_mat)
    save_time = time()-t0
    print(f"Similarity matrix is saved in file: {sim_mat_save_fname}, "
          f"time taken to save: {save_time:.2f}")

    compute_diff(sent_embs, our_total_time, sim_mat)


if __name__ == '__main__':
    main()