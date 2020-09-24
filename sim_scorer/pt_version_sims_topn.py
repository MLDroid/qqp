import numpy as np
from time import time
from tqdm import tqdm
import torch

from sim_scorer import sims_topn_index_and_val


# def sim_matrix(a, b, eps=1e-8):
#     """
#     added eps for numerical stability
#     """
#     a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
#     return sim_mt

def sim_matrix(a, b):
    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt

def normailze_sent_emb(sent_embs, eps = 1e-8):
    sent_embs_norm = sent_embs.norm(dim=1)[:, None]
    sent_embs = sent_embs / torch.max(sent_embs_norm, eps * torch.ones_like(sent_embs_norm))
    return sent_embs

def main():
    batch_size = 3000
    max_samples = None
    top_n_sim_sents = 100


    emb_fname = '../data/distilbert-base-nli-stsb-mean-tokens_sent_embeddings.npy'

    sent_embs = sims_topn_index_and_val.load_emb(emb_fname)
    sent_embs = sent_embs[:max_samples]
    sent_embs = torch.tensor(sent_embs, dtype=torch.float32).cuda()
    print(f'After reduction, the shape of embedding matrix: {sent_embs.shape}')

    sent_embs = normailze_sent_emb(sent_embs)

    inds = list(range(sent_embs.shape[0]))
    chunked_inds = list(sims_topn_index_and_val.chunks(inds, batch_size))

    t00 = time()
    # most_sim_inds = []
    # sim_values = []
    for batch_num, batch_inds in enumerate(tqdm(chunked_inds), start=1):
        t0 = time()
        left_mat = sent_embs[batch_inds]
        batch_sims = sim_matrix(left_mat, sent_embs)
        batch_sims = batch_sims.cpu().numpy().astype(np.float32)

        #compare similarities
        # diff = cosine_similarity(left_mat.cpu().numpy(), sent_embs.cpu().numpy()) - batch_sims
        # diff = sum(abs(diff)).sum()
        # print(f'Diff between pt and scipy: {diff}')

        #get indexes and similarity scores
        topinds = np.flip(batch_sims.argsort(),1)[:,:top_n_sim_sents]
        sims = np.take_along_axis(batch_sims,topinds,1)

        #type cast
        topinds = topinds.astype(np.int32)
        sims = sims.astype(np.float32)

        #save indexes that are similar to a given sent id
        fname_1 = f'most_sim_inds_{batch_num}.npy'
        np.save(fname_1, topinds)

        #save sim scores - to be used for threshold
        fname_2 = f'sim_values_{batch_num}.npy'
        np.save(fname_2, sims)

        # most_sim_inds.append(topinds)
        # sim_values.append(sims)
        batch_time = time()-t0
        print(f'Done with batch: {batch_num} in {batch_time:.2f} sec.')
    our_total_time = time() - t00
    print(f'All batches done in : {our_total_time:.2f} sec.')

    # most_sim_inds = np.vstack(most_sim_inds)
    # sim_values = np.vstack(sim_values)
    #
    # np.save('most_sim_inds.npy',most_sim_inds)
    # np.save('sim_values.npy',sim_values)


if __name__ == '__main__':
    main()