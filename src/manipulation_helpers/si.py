import torch
import numpy as np
from torch import nn
from transformers import AutoModel
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score


class SILitModel(pl.LightningModule):

    def __init__(self, bert_model_name: str, optimizer, lr: float, freeze_bert: bool, 
                 loss_function, output_size: int, scheduler=None, random_refreeze: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.head = nn.Linear(self.bert.config.hidden_size, output_size)
        self.drop = nn.Dropout(0.2)
        self.lr = lr
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.random_refreeze = random_refreeze
        if freeze_bert: 
            self.refreeze_bert(random_refreeze)

    def refreeze_bert(self, random: bool = False):
            
        if random:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
            self.to('cpu')
            torch.cuda.empty_cache()
            to_freeze = np.random.choice([-1] + list(range(len(self.bert.encoder.layer))), 4, replace=False)
            for num_layer in to_freeze:
                if num_layer == -1:
                    for parameter in self.bert.embeddings.parameters():
                        parameter.requires_grad = True
                else:
                    for parameter in self.bert.encoder.layer[num_layer].parameters():
                        parameter.requires_grad = True
            self.to('cuda')
        else: 
            for parameter in self.bert.embeddings.parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[0].parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[1].parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[2].parameters():
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        dropped = self.drop(bert_output.last_hidden_state)
        output = self.head(dropped)

        return output
    
    def on_after_backward(self):
        self.refreeze_bert(self.random_refreeze)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        attention_mask = (y != -100)
        y_hat = self(x, attention_mask)
        y_hat = y_hat.transpose(1, 2)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch 
        attention_mask = (y != -100)
        y_hat = self(x, attention_mask)
        y_hat = y_hat.transpose(1, 2)
        loss = self.loss_function(y_hat, y)
 
        y_hat = y_hat.argmax(-2)
        
        mask = y != -100
        
        recall = (((y > 0) & (y_hat > 0)).sum(-1) / (y > 0).sum(-1)).sum() / (y > 0).any(-1).sum()
        precision = (((y > 0) & (y_hat > 0)).sum(-1) / (y_hat > 0).sum(-1)).sum() / (y_hat > 0).any(-1).sum()
        
        precision = torch.nan_to_num(precision)
        recall = torch.nan_to_num(recall)
        f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True, logger=True)
