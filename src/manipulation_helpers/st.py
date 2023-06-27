import torch
from torch import nn
import numpy as np
from transformers import AutoModel
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


class STLitModel(pl.LightningModule):

    def __init__(self, bert_model_name: str, optimizer, lr: float, freeze_bert: bool, 
                 loss_function, output_size: int, scheduler=None, lora: bool = False, freeze_all: bool = False, mode: str = 'bi'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.drop = nn.Dropout(0.2)
        if mode == 'bi':
            self.head = nn.Linear(self.bert.config.hidden_size * 3, output_size)
        elif mode == 'cross':
            self.head = nn.Linear(self.bert.config.hidden_size * 2, output_size)
        self.lr = lr
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        if lora:
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1, bias="all"
            )
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()
        self.freeze_all = freeze_all
        if freeze_bert: 
            self.freeze_bert()
        self.mode = mode

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
            for parameter in self.bert.encoder.layer[2].parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[3].parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[4].parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[5].parameters():
                parameter.requires_grad = False
            for parameter in self.bert.encoder.layer[6].parameters():
                parameter.requires_grad = False
        

    def forward(self, full_text_ids, entity_ids, span_ids, attention_ft_mask, attention_ent_mask, attention_span_mask, **kwargs):
        if self.mode == 'bi':
            ft_output = self.bert.base_model.forward(full_text_ids, attention_mask=attention_ft_mask, **kwargs)
            ent_output = self.bert.base_model.forward(entity_ids, attention_mask=attention_ent_mask, **kwargs)
            spn_output = self.bert.base_model.forward(span_ids, attention_mask=attention_span_mask, **kwargs)
            cls_emb = ft_output.pooler_output
            ent_emb = ent_output.pooler_output
            span_emb = spn_output.pooler_output
            span_targeting_emb = torch.cat((cls_emb, ent_emb, span_emb), dim=-1)
        elif self.mode == 'cross':
            context_input = torch.concat([entity_ids, span_ids], axis=-1)
            attention_mask = torch.concat([attention_ent_mask, attention_span_mask], axis=-1)
 
            context_output = self.bert.base_model.forward(context_input, attention_mask=attention_mask, **kwargs)
            ft_output = self.bert.base_model.forward(full_text_ids, attention_mask=attention_ft_mask, **kwargs)

            context_emb = context_output.pooler_output
            ft_emb = ft_output.pooler_output
            
            span_targeting_emb = torch.cat((ft_emb, context_emb), dim=-1)
        
        span_targeting_emb = self.drop(span_targeting_emb)
        
        output = self.head(span_targeting_emb)

        return output

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        full_text_ids, entity_ids, span_ids, y = train_batch 
        attention_ft_mask, attention_ent_mask, attention_span_mask = (full_text_ids != 0), (entity_ids != 0), (span_ids != 0)
        
        y_hat = self(full_text_ids, entity_ids, span_ids, 
                     attention_ft_mask, attention_ent_mask, attention_span_mask)
        
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        full_text_ids, entity_ids, span_ids, y = batch 
        attention_ft_mask, attention_ent_mask, attention_span_mask = (full_text_ids != 0), (entity_ids != 0), (span_ids != 0)
        
        y_hat = self(full_text_ids, entity_ids, span_ids, 
                     attention_ft_mask, attention_ent_mask, attention_span_mask)
        
        loss = self.loss_function(y_hat, y)
 
        y_hat = y_hat.argmax(-1)
        #print(y_hat.cpu().flatten(), y.cpu().flatten())
        accuracy = (y.cpu().flatten() == y_hat.cpu().flatten()).sum() / y.flatten().shape[0]
        precision = precision_score(y.cpu().flatten(), y_hat.cpu().flatten())
        recall = recall_score(y.cpu().flatten(), y_hat.cpu().flatten())
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True, logger=True)