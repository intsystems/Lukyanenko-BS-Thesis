import torch
from torch import nn
from transformers import AutoModel
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score


class ManipulationTargetLightningModel(pl.LightningModule):

    def __init__(self, bert_model_name: str, optimizer, lr: float, freeze_bert: bool, loss_function):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.head = nn.Linear(self.bert.config.hidden_size, 2)
        self.lr = lr
        self.optimizer = optimizer
        self.loss_function = loss_function
        if freeze_bert: 
            self.freeze_bert()

    def freeze_bert(self):
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

    def forward(self, input_ids):
        bert_output = self.bert(input_ids)
        connection_output = self.head(bert_output.last_hidden_state)

        return connection_output

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        y_hat = self(x)
        y_hat = y_hat.transpose(1, 2)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        y_hat = y_hat.transpose(1, 2)
        loss = self.loss_function(y_hat, y)
        accuracy = accuracy_score(y.cpu().flatten(), y_hat.argmax(-2).cpu().flatten())
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, prog_bar=True, logger=True)
