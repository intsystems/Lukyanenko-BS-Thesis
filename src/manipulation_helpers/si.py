import torch
import numpy as np
from torch import nn
from transformers import AutoModel
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


class SILitModel(pl.LightningModule):

    def __init__(self, bert_model_name: str, optimizer, lr: float, freeze_bert: bool, 
                 loss_function, output_size: int, scheduler=None, freeze_all: bool = False, lora: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        if lora:
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1, bias="all"
            )
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()
        self.head = nn.Linear(self.bert.config.hidden_size, output_size)
        self.drop = nn.Dropout(0.2)
        self.lr = lr
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.freeze_all = freeze_all
        if freeze_bert: 
            self.freeze_bert()

    def freeze_bert(self):

        if self.freeze_all:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
        else:
            for parameter in self.bert.embeddings.parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[0].parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[1].parameters():
                parameter.requires_grad = False
            #for parameter in self.bert.encoder.layer[2].parameters():
            #    parameter.requires_grad = False
            #for parameter in self.bert.encoder.layer[3].parameters():
            #    parameter.requires_grad = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bert_output = self.bert.base_model.forward(input_ids, 
                                attention_mask=attention_mask, 
                                **kwargs)
        dropped = self.drop(bert_output.last_hidden_state)
        output = self.head(dropped)

        return output
    
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
