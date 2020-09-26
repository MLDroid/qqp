import psutil
import torch
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME =  'distilbert-base-uncased'#, 'bert-base-uncased', 'roberta-base', 'albert-base-v2'
MODEL_NAME =  'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'
MODEL_NAME = sys.argv[1]
BATCH_SIZE = int(sys.argv[2])
LR = 0.00001

data_folder = './data'
X_train_fname = os.path.join(data_folder, 'train_pairs.txt')
X_valid_fname = os.path.join(data_folder, 'valid_pairs.txt')
y_train_fname = os.path.join(data_folder, 'y_train.txt')
y_valid_fname = os.path.join(data_folder, 'y_valid.txt')
id_ques_map = os.path.join(data_folder, 'id_ques_map.json')


IS_LOWER = True if 'uncased' in MODEL_NAME else False

MAX_SEQ_LEN = 4096 if 'longformer' in MODEL_NAME else 512
NUM_EPOCHS = 20
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
SAVE_EVERY = 5

BERT_LAYER_FREEZE = False

MULTIGPU = True if torch.cuda.device_count() > 1 else False #when using xlarge vs 16x large AWS m/c

TRAINED_MODEL_FNAME_PREFIX = MODEL_NAME.replace('/','_').upper()+'_model'
TRAINED_MODEL_FNAME = None #MODEL_NAME.upper()+'_vanilla_model_e_.pt'

CONTEXT_VECTOR_SIZE = 1024 if 'large' in MODEL_NAME else 768
PREDICTIONS_FNAME = '_'.join([MODEL_NAME.replace('/','_').upper(),'preds.txt'])





