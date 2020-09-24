import torch.nn as nn
from transformers import AutoModel

import config


class bert_classifier(nn.Module):
    def __init__(self, freeze_bert=True):
        super(bert_classifier, self).__init__()
        # Instantiating BERT model object
        self.model_name = config.MODEL_NAME
        self.bert_layer = AutoModel.from_pretrained(self.model_name)
        print(f'Loaded model from AutoMdodel for model name: {self.model_name}')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.fc1 = nn.Linear(config.CONTEXT_VECTOR_SIZE, 100) #using a simple softmax classifier with one FC layer
        self.fc2 = nn.Linear(100, 2) #using a simple softmax classifier with one FC layer

        #Dropout
        self.dp20 = nn.Dropout(0.2)

        #non linear functions
        self.relu = nn.ReLU()


    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to DistilBERT model to obtain contextualized representations
        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)

        # Obtaining the representation of [CLS] head
        # cls_rep = cont_reps[0][:,0,:]
        op = cont_reps[0].mean(1)
        op = self.dp20(op)

        # Feeding op to the classifier layer
        op = self.relu(self.fc1(op))
        op = self.dp20(op)
        logits = self.fc2(op)

        return logits