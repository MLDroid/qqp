import torch
import numpy as np, sys, time, ast
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
import dataset
from vanilla_transformer import load_Xy, safe_calc_auc


def test(net, test_loader, device='cpu', pred_save_fname=False):
    net.eval()
    with torch.no_grad():
        for batch_num, (seq, attn_masks, labels) in enumerate(tqdm(test_loader), start=1):
            seq, attn_masks = seq.cuda(device), attn_masks.cuda(device)
            logits = net(seq, attn_masks)
            preds = logits.argmax(1).cpu().numpy()
            labels = labels.cpu().numpy()
            if batch_num == 1:
                all_trues = labels
                all_preds = preds
            else:
                all_trues = np.hstack([all_trues, labels])
                all_preds = np.hstack([all_preds, preds])

        auc = safe_calc_auc(all_trues, all_preds)
        f1 = f1_score(all_trues, all_preds)
        acc = accuracy_score(all_trues, all_preds)
        p = precision_score(all_trues, all_preds)
        r = recall_score(all_trues, all_preds)
        if pred_save_fname:
            with open(pred_save_fname,'w') as fh:
                for i in all_preds:
                    print(i,file=fh)
            print(f'Predictions are saved to file: {pred_save_fname}')

    return acc, auc, p, r, f1

def main():
    #cmd line args
    print(config.MODEL_NAME)
    TRAINED_MODEL_FNAME = sys.argv[2]
    config.BATCH_SIZE = int(sys.argv[3]) #1, 10, 100
    is_cpu = bool(int(sys.argv[4])) #0 or 1

    # TRAINED_MODEL_FNAME = "saved_model_op/DISTILBERT-BASE-UNCASED_model_e_10.pt"
    # config.BATCH_SIZE = 10  # 1, 10, 100
    # is_cpu = 0  # 0 or 1

    #Setting up the device to be used for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_cpu:
        device = torch.device('cpu')
    print(f'****Device used for this inference: {device}***')

    #Load the validation set
    _, X_valid, _, y_valid = load_Xy(1000)
    valid_set = dataset.dataset(X_valid, y_valid, max_len=config.MAX_SEQ_LEN)

    # Validation set should NOT be shuffled
    test_loader = DataLoader(valid_set,
                             shuffle=False,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_CPU_WORKERS)

    #Loading BERT model
    bert_model = torch.load(TRAINED_MODEL_FNAME)
    try:
        bert_model = bert_model.module
    except:
        print('Unable to locate .module form the model loaded, this model might not be from nn.DataParallel')

    # Switching to GPU or CPU based on the cmd line arg
    bert_model.cuda()
    if is_cpu:
        bert_model.to('cpu')
    print(f'Loaded trained model: {bert_model} \nfrom file: {TRAINED_MODEL_FNAME}')


    # # Multi GPU setting - this is NOT relevant now
    # if config.MULTIGPU:
    #     device_ids = [0, 1, 2, 3] #huggingface allows parallelizing only upto 4 cards
    #     bert_model = nn.DataParallel(bert_model, device_ids = device_ids)
    #     print(f'Model parallelized on the following cards: ',device_ids)

    t0 = time.perf_counter()
    acc, auc, p, r, f1 = test(bert_model, test_loader, device=device, pred_save_fname=False)
    t = round(time.perf_counter() - t0, 2)
    print(f'Inference time: {t}, validation ACC: {acc:.4f} and AUC: {auc:.4f},'
          f'P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}')


if __name__ == '__main__':
    main()