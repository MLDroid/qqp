import json, ast
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

import config, dataset, model


def load_id_ques_map(fname):
    with open(fname) as fh:
        id_ques_map = json.load(fh)
    id_ques_map = {int(k): str(v) for k, v in id_ques_map.items()}
    id_ques_map[0] = ''
    print(f'Loaded {max(id_ques_map.keys())} ids to questions from {fname}')
    return id_ques_map


def load_Xy(max_samples=None):
    train_pairs = np.array([list(ast.literal_eval(l.strip())) for l in
                            open(config.X_train_fname).readlines()])[:max_samples]
    valid_pairs = np.array([list(ast.literal_eval(l.strip())) for l in
                            open(config.X_valid_fname).readlines()])[:max_samples]

    id_ques_map = load_id_ques_map(config.id_ques_map)
    X_train = [(id_ques_map[i], id_ques_map[j]) for i,j in train_pairs]
    X_valid = [(id_ques_map[i], id_ques_map[j]) for i,j in valid_pairs]

    y_train = np.loadtxt(config.y_train_fname).astype(int)[:max_samples]
    y_valid = np.loadtxt(config.y_valid_fname).astype(int)[:max_samples]

    return X_train, X_valid, y_train, y_valid


def get_metrics_from_logits(logits, labels):
    all_preds = logits.argmax(1).cpu().numpy()
    all_trues = labels.cpu().numpy()
    auc = safe_calc_auc(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds)
    acc = accuracy_score(all_trues, all_preds)
    return acc, f1, auc


def safe_calc_auc(trues, preds):
    try:
        auc = roc_auc_score(trues, preds)
    except ValueError:
        auc = 0.5
    return auc


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


def train_model(net, criterion, optimizer, scheduler, train_loader, test_loader=None,
                print_every=100, n_epochs=10, device='cpu', save_model=True,
                save_every=5):

    for e in range(1, n_epochs+1):
        t0 = time.perf_counter()
        e_loss = []
        for batch_num, (seq_attnmask_labels) in enumerate(tqdm(train_loader), start=1):
            # Clear gradients
            optimizer.zero_grad()

            #get the 3 input args for this batch
            seq, attn_mask, labels = seq_attnmask_labels

            # Converting these to cuda tensors
            seq, attn_mask, labels = seq.cuda(device), attn_mask.cuda(device), labels.cuda(device)

            # Obtaining the logits from the model
            logits = net(seq, attn_mask)

            # Computing loss
            loss = criterion(logits, labels)
            e_loss.append(loss.item())

            # Backpropagating the gradients for losses on all classes
            loss.backward()

            # Optimization step
            optimizer.step()
            scheduler.step()

            if batch_num % print_every == 0:
                acc, f1, auc = get_metrics_from_logits(logits, labels)
                print(f"batch {batch_num} of epoch {e} complete. Loss : {loss.item()} "
                      f"ACC : {acc:.4f}, F1: {f1:.4f} and AUC:{auc:.4f}")

        t = time.perf_counter() - t0
        t = t/60 # mins
        e_loss = np.array(e_loss).mean()
        print(f'Done epoch: {e} in {t:.2f} mins, epoch loss: {e_loss}')
        if test_loader != None:
            if e%save_every == 0:
                pred_save_fname = config.PREDICTIONS_FNAME.replace('.txt', f'_{e}.txt')
            else:
                pred_save_fname = None
            acc, auc, p, r, f1 = test(net, test_loader, device=device, pred_save_fname=pred_save_fname)
            print(f'After epoch: {e}, validation ACC: {acc:.4f} and AUC: {auc:.4f},'
          f'P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}')

        if save_model:
            if e%save_every==0:
                save_fname = config.TRAINED_MODEL_FNAME_PREFIX + f'_e_{e}.pt'
                torch.save(net, save_fname)
                print(f'Saved model at: {save_fname} after epoch: {e}')



def compute_class_weight_balanced(y):
    n_samples = len(y)
    n_classes = 2 #for a binary classification
    w = n_samples / (n_classes * np.bincount(y))
    return w


def get_class_weigts(y_train, max_class_weight=50.):
    labels = y_train
    w = compute_class_weight_balanced(labels)
    w = w.clip(max = max_class_weight)
    return w


def main():
    max_samples = None
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    X_train, X_valid, y_train, y_valid = load_Xy(max_samples)

    #for dataset loader
    train_set = dataset.dataset(X_train, y_train, max_len=config.MAX_SEQ_LEN)
    valid_set = dataset.dataset(X_valid, y_valid, max_len=config.MAX_SEQ_LEN)

    #creating dataloader
    #Training set should be shuffled
    train_loader = DataLoader(train_set,
                              shuffle = True,
                              batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_CPU_WORKERS)
    #Validation set should NOT be shuffled
    test_loader = DataLoader(valid_set,
                             shuffle = False,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_CPU_WORKERS)

    #creating BERT model
    if config.TRAINED_MODEL_FNAME:
        bert_model = torch.load(config.TRAINED_MODEL_FNAME)
        print(f'Loaded trained model: {bert_model} from file: {config.TRAINED_MODEL_FNAME}')
    else:
        bert_model = model.bert_classifier(freeze_bert=config.BERT_LAYER_FREEZE)
        print(f"created NEW TRANSFORMER model for finetuning: {bert_model}")
    bert_model.cuda()

    # Multi GPU setting
    if config.MULTIGPU:
        device_ids = [0, 1, 2, 3] #huggingface allows parallelizing only upto 4 cards
        bert_model = nn.DataParallel(bert_model, device_ids = device_ids)
        print(f'Model parallelized on the following cards: ',device_ids)

    #loss function (with weights)
    class_weights = get_class_weigts(y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    param_optimizer = list(bert_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_set) / config.BATCH_SIZE * config.NUM_EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    train_model(bert_model, criterion, optimizer, scheduler, train_loader, test_loader,
                print_every=config.PRINT_EVERY, n_epochs=config.NUM_EPOCHS, device=device,
                save_model=True, save_every = config.SAVE_EVERY)

    acc, auc, p, r, f1 = test(bert_model, test_loader, device=device, pred_save_fname=True)
    print(f'After ALL epochs: validation ACC: {acc:.4f} and AUC: {auc:.4f},'
          f'P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}')


if __name__ == '__main__':
    main()