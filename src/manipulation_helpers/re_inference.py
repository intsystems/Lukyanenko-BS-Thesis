from manipulation_helpers.data_preparation import ManipulationDataset, Markup
from manipulation_helpers.models import ComplexModel, MODEL_NAME
from skimage.restoration import denoise_bilateral
import numpy as np
from typing import List
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

PATH_TO_MODEL = "bilin_freeze_bert_RE_model_v1"

class ReModel:
    def __init__(
            self, 
            path_to_model: str = PATH_TO_MODEL, 
            model_name: str = MODEL_NAME):
        self.device = torch.device('cuda')
        self.model = ComplexModel(MODEL_NAME).to(self.device)
        self.model.load_state_dict(torch.load(PATH_TO_MODEL))
        self.model.eval()
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        self.device = torch.device('cuda')
        self.threshold = 0.5

    def preprocess(self, samples: List[str]):
        encodings = self.tokenizer(
            samples, 
            is_split_into_words=True, 
            return_offsets_mapping=True, 
            padding=True, 
            truncation=True, 
            max_length=512)
        dataloader = Dataloader(ManipulationDataset(encodings, inference=True), batch_size=1)
        return dataloader
    

    @staticmethod
    def postprocess(raw_matrix: List[List[float]]):
        raw_img = np.array(raw_matrix)
        denoised_img = denoise_bilateral(raw_img)
        return denoised_img
    

    @staticmethod
    def get_token_spans(processed_matrix, k=5):
        connections = []
        for i in range(len(processed_matrix)):
            connections.append((i, np.argwhere(processed_matrix[:, i]).reshape(-1)))
        topk = sorted(connections, key=lambda x: len(x[1]))[-k:]
        return topk
    

    def get_entity_and_manipulation_span(self, connection, input_id):
        entity = self.tokenizer.decode(input_id[connection[0]-5:connection[0]+5])
        manipulation = self.tokenizer.decode(np.array(input_id)[connection[1]])
        return entity, manipulation

    def inference(self, samples: List[str]):
        list_of_processed_predicted_matrix = []
        list_of_ner_man_spans_ids = []
        list_of_ner_man_spans = []

        dataloader = self.preprocess(samples)

        with torch.no_grad():
            i = 0
            for input_ids, attention_mask in dataloader:
                
                try:
                    sentence_len = (attention_mask[0] == 0).nonzero().squeeze()[0]
                except IndexError:
                    sentence_len = 512

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                connection_output = self.model(input_ids, attention_mask=attention_mask)

                inptid = input_ids[0].cpu().tolist()

                raw_matrix = connection_output[:, :sentence_len, :sentence_len][0].cpu().tolist()
                processed_matrix = self.postprocess(raw_matrix)

                list_of_processed_predicted_matrix.append(processed_matrix)
                top_k_length_manipulation_spans = self.get_token_spans(processed_matrix > self.threshold)
                list_of_ner_man_spans_ids.append(top_k_length_manipulation_spans)
                
                list_of_ner_man_spans.append([
                    self.get_entity_and_manipulation_span(ent_man, inptid) for ent_man in top_k_length_manipulation_spans
                    ])
        
        return list_of_processed_predicted_matrix, list_of_ner_man_spans_ids, list_of_ner_man_spans
