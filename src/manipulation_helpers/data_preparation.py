from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import json
import re
from itertools import product
from collections import defaultdict

import pandas as pd
import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from transformers.tokenization_utils_base import BatchEncoding
import torch
from torch.utils.data import Dataset
import numpy as np

nlp = spacy.load("ru_core_news_md")


class Markup(NamedTuple):
    """
    manipulation_class - название класса манипуляции
    manipulation_target - id сущности, на которую направлена манипуляция
    entity_id - id сущности
    """

    manipulation_class: str
    manipulation_target: int
    entity_id: int


def read_markup(markup_path):
    markup = pd.read_csv(markup_path, sep="\t")
    markup.columns = markup.columns.map(lambda x: x.lower().replace(":", "_"))
    markup = markup[~markup['output_result'].isnull()]
    markup = markup[~markup['input_entitiesdata'].isnull()]  ## new
    markup["output_result"] = markup["output_result"].apply(lambda x: json.loads(x) if not pd.isna(x) else [])
    markup["input_entitiesdata"] = markup["input_entitiesdata"].apply(
        lambda x: '[' + x + ']' if not pd.isna(x) else [])  ## new
    markup["input_entitiesdata"] = markup["input_entitiesdata"].apply(
        lambda x: x.replace('\\\\"', '').replace('\\', ''))  ## new
    markup["input_entitiesdata"] = markup["input_entitiesdata"].apply(lambda x: json.loads(x))  ## new
    markup = markup.reset_index().drop(['index'], axis='columns')
    return markup


def spans_to_offsets(doc: Doc, spans: List[Span], missing: str = "O") -> List[str]:
    spans = [x for x in spans if x is not None]
    spans = spacy.util.filter_spans(spans)
    spans = [(s.start_char, s.end_char, s.label_) for s in spans]
    markup_biluo = spacy.training.offsets_to_biluo_tags(doc, spans, missing=missing)
    return markup_biluo


def clear_biluo(bilio_tags: List[str], remove_prefixes: bool = False) -> List[str]:
    """
    Превращаем BILIO разметку в BIO или совсем удаляем префиксы, оставляем только названия классов
    """
    for i in range(len(bilio_tags)):
        if not remove_prefixes:
            bilio_tags[i] = (
                bilio_tags[i].replace("L-", "I-").replace("U-", "B-").replace("о-с", "о_с")
            )
        else:
            bilio_tags[i] = (
                bilio_tags[i]
                    .replace("U-", "")
                    .replace("B-", "")
                    .replace("L-", "")
                    .replace("I-", "")
            )
    return bilio_tags


def clear_text_from_markup(text: str) -> str:
    """
    Удаляем из текста разметку сущностей, которая использовалась в толоке
    *(1) Сущность*
    """
    text = re.sub(" \*\(\d+\)\s+([\w\s]+?)\* ", "\g<1>", text)
    text = re.sub("\s?\*\(\d+\)\s?", "", text)
    text = re.sub("(?:^| )\(?\d+\)", " ", text)
    text = text.replace("* ", "").replace(" *", "").strip("*")
    text = re.sub("\s+", " ", text).strip()
    return text


def markup_conll(
    text: str, markup: List[Any], entities: List[Any]
) -> Tuple[List[str], List[Markup]]:
    """
    Форматирование разметки из толоки.
    На выходе получаем список токенов с :
        1) разметкой фрагментов манипуляции
        2) id сущностей, на котороые эти фрагменты направлены
        3) id сущности
    """
    text = re.sub("\s+", " ", text)
    doc = nlp.make_doc(text)
    spans, not_found_entities = [], []
    manipulation_target_spans = []
    entities_spans = []

    for example in markup:
        example_text = clear_text_from_markup(example["text"])
        class_name = re.sub("\(.*\)", "", example["class_name"]).strip().replace(" ", "_")

        start_idx = text.find(example_text)
        if example_text and start_idx != -1:
            spans.append(
                doc.char_span(
                    start_idx, start_idx + len(example_text), class_name, alignment_mode="expand"
                )
            )

            if example["manipulation_target_id"] != "no-entity":
                manipulation_target_spans.append(
                    doc.char_span(
                        start_idx,
                        start_idx + len(example_text),
                        example["manipulation_target_id"],
                        alignment_mode="expand",
                    )
                )
        else:
            print("Не найдена сущность '{}'".format(example["text"]))
            not_found_entities.append(example)

    entities = [e for e in entities if e["from"] and e["to"]]
    for i, example in enumerate(entities):
        entities_spans.append(
            doc.char_span(example["from"], example["to"], str(i + 1), alignment_mode="expand")
        )

    if not len(spans):
        print("Нет сущностей")
    try:
        markup_biluo = spans_to_offsets(doc, spans)
        manipulation_target_biluo = spans_to_offsets(doc, manipulation_target_spans, missing="0")
        entities_biluo = spans_to_offsets(doc, entities_spans, missing="0")
    except Exception as e:
        # return entities_spans, None
        print(text[:30], spans)
        raise e

    markup_biluo = clear_biluo(markup_biluo)
    manipulation_target_biluo = clear_biluo(manipulation_target_biluo, remove_prefixes=True)
    entities_biluo = clear_biluo(entities_biluo, remove_prefixes=True)

    tokens_markup = zip(
        markup_biluo, list(map(int, manipulation_target_biluo)), list(map(int, entities_biluo))
    )
    tokens_markup = [Markup(*x) for x in tokens_markup]

    return list(map(str, doc)), tokens_markup


def encode_tags(
    tags: List[List[Markup]],
    encodings: BatchEncoding,
    target_column: str,
    tag2id: Optional[Dict[str, int]] = None,
):
    """
    Выравнивание разметки слов с токенами трансформера
    """
    assert target_column in {"manipulation_class", "entity_id", "manipulation_target", "bio_span", "eng"}

    if target_column == "manipulation_class":
        labels = [[tag2id[tag.manipulation_class] for tag in doc] for doc in tags]
    elif target_column == "entity_id":
        labels = [[x.entity_id for x in row] for row in tags]
    elif target_column == "manipulation_target":
        labels = [[x.manipulation_target for x in row] for row in tags]
    elif target_column == "bio_span":
        labels = [[tag2id[tag.manipulation_class[0]] for tag in doc] for doc in tags]
    elif target_column == "eng":
        labels = tags

    encoded_labels = []
    for i, doc_labels in enumerate(labels):
        word_ids = encodings.word_ids(i)
        aligned_labels = [-100 if idx is None else doc_labels[idx] for idx in word_ids]
        encoded_labels.append(aligned_labels)
    return encoded_labels


def get_connections(
    entities: List[List[Any]], manipulation_targets: List[List[Any]]
) -> Tuple[List[dict], List[List[int]]]:
    """
    Функция, которая принимает на вход последовательность меток для текста а на выход выдет список
        из структур, каждая из которых описывает попарное соответствие манипуляции и мишени а так же
        есть ли между ними связь
    """
    connections = []
    connections_ans = []
    for text_entities, text_mt in zip(entities, manipulation_targets):
        text_entities = [x for x in text_entities if x != -100]
        text_mt = [x for x in text_mt if x != -100]
        con = {
            "man_id_start": [],
            "man_id_end": [],
            "ent_id_start": [],
            "ent_id_end": [],
        }
        answer = []
        text_connections_man = {}
        text_connections_ent = {}

        curr_ent = 0
        for i, ent in enumerate(text_entities):
            if ent != 0 and ent != curr_ent:
                text_connections_ent[ent] = [i]
            elif ent != 0 and ent == curr_ent:
                text_connections_ent[ent].append(i)
            curr_ent = ent

        curr_man = 0
        for i, man in enumerate(text_mt):
            if man != 0 and man != curr_man:
                text_connections_man[man] = [i]
            elif man != 0 and man == curr_man:
                text_connections_man[man].append(i)
            curr_man = man

        # manipulation_target, entity_id
        for manipulation_target, entity_id in product(text_connections_man.keys(), text_connections_ent.keys()):
            con["man_id_start"].append(text_connections_man[manipulation_target][0])
            con["man_id_end"].append(text_connections_man[manipulation_target][-1])
            con["ent_id_start"].append(text_connections_ent[entity_id][0])
            con["ent_id_end"].append(text_connections_ent[entity_id][-1])
            if manipulation_target == entity_id:
                answer.append(1)
            else:
                answer.append(0)
        connections.append(con)
        connections_ans.append(answer)
    return connections, connections_ans


class ManipulationDataset(Dataset):
    """
    Класс датасета для задачи манипуляции.
        Параметр sample_size_of_connections отвечает за количество семплируемых connections
        (для того, чтобы унифицировать кол-во связей для обучения по батчам)
    """

    def __init__(self, encodings, labels=None, connections=None, connections_ans=None, inference=False):
        self.inference = inference
        if self.inference:
            self.encodings = encodings
        else:
            self.encodings = encodings
            self.labels = labels
            self.connections = connections
            self.connections_ans = connections_ans
            self.connections_matrices = self.generate_connection_matrices(
                connections, connections_ans, encodings["attention_mask"]
            )

    def __getitem__(self, idx):
        if self.inference:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            return input_ids, attention_mask
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            labels = torch.tensor(self.labels[idx])
            connections_matrix = torch.Tensor(self.connections_matrices[idx])
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            return input_ids, attention_mask, labels, connections_matrix

    def __len__(self):
        return len(self.encodings['input_ids'])

    @staticmethod
    def generate_connection_matrices(connections, connections_ans, attention_masks):
        connections_matrices = []
        for connection, connection_ans, attention_mask in zip(
            connections, connections_ans, attention_masks
        ):

            connection_ans = np.array(connection_ans)
            attention_mask = np.array(attention_mask)
            connection_ids = np.ones((4, len(connection_ans)), dtype=int)
            connection_ids[0] = np.array(connection['man_id_start'])
            connection_ids[1] = np.array(connection['man_id_end'])
            connection_ids[2] = np.array(connection['ent_id_start'])
            connection_ids[3] = np.array(connection['ent_id_end'])
            connection = connection_ids
            sentence_len = (attention_mask == 1).sum()
            connection_matrix = torch.zeros((len(attention_mask), len(attention_mask)))
            if len(connection_ans) == 0:
                connections_matrices.append(connection_matrix)
                continue
            true_connections = connection[:, connection_ans == 1]
            for i in range(true_connections.shape[1]):
                man_id_start = true_connections[0, i]
                man_id_end = true_connections[1, i]
                ent_id_start = true_connections[2, i]
                ent_id_end = true_connections[3, i]
                connection_matrix[
                    int(man_id_start.item()): int(man_id_end.item()) + 1,
                    int(ent_id_start.item()): int(ent_id_end.item()) + 1
                ] = torch.ones(int(man_id_end.item()) - int(man_id_start.item()) + 1,
                               int(ent_id_end.item()) - int(ent_id_start.item()) + 1)
            connections_matrices.append(connection_matrix)
        return connections_matrices


def get_spans(span):
    start_ids = np.where(span == 2)[0]
    for start_id in start_ids:
        spn = [0 for _ in range(start_id)] + [1]
        for i, token in enumerate(span[start_id + 1:]):
            if token == 0 or token == 2:
                spn += [0 for _ in range(512 - i - start_id - 1)]
                yield np.array(spn)
                break
            spn.append(1)

    
def create_span_targeting_data(encodings, entities, manipulation_targets, spans, encoded_cls_token, 
                               max_length_text: int = 512, max_length_entity: int = 64, max_length_span: int = 256):
    
    new_entities = []
    new_spans = []
    new_full_texts = []
    new_labels = []

    for encodings, entities, targets, span in zip(encodings['input_ids'], 
                                                  entities, 
                                                  manipulation_targets,
                                                  spans):
      
        encodings, entities, targets, span =\
                np.array(encodings), np.array(entities), np.array(targets), np.array(span)

        if max(entities) == 0: # (когда нет НЕРа - уходим)
            continue
        
        else:
            if max(targets) == 0: # (когда нет связанных фрагментов)
                if max(span) == 0: # (когда только НЕРы тоже уходим)
                    continue
                else: # (когда есть фрагменты манипуляции и есть НЕРы, но они не связаны -> рандомим)
                    
                    entity = np.random.randint(1, max(entities) + 1)
                    for span_ in get_spans(span):
                        new_full_texts.append(
                            [encoded_cls_token] + list(encodings)[:-1]
                        )
                        new_entities.append(
                            [encoded_cls_token] + \
                            list(encodings[entities == entity]) + \
                            [0 for _ in range(max_length_entity - (entities == entity).sum())]
                        )
                        new_spans.append(
                            [encoded_cls_token] + \
                            list(encodings[span_ == 1]) + \
                            [0 for _ in range(max_length_span - (span_ == 1).sum())]
                        )
                        new_labels.append(0)
                            
            else: # (когда есть связанные фрагменты)
                for entity in np.unique(targets):
                    if entity <= 0:
                        continue
                    new_full_texts.append(
                        [encoded_cls_token] + list(encodings)[:-1]
                    )
                    new_entities.append(
                        [encoded_cls_token] + \
                        list(encodings[entities == entity]) + \
                        [0 for _ in range(max_length_entity - (entities == entity).sum())]
                    )
                    new_spans.append(
                        [encoded_cls_token] + \
                        list(encodings[targets == entity])[:max_length_span] + \
                        [0 for _ in range(max_length_span - (targets == entity).sum())]
                    )
                    new_labels.append(1)

    print(f'% of positive class {sum(new_labels) / len(new_labels)}')
    return new_full_texts, new_entities, new_spans, new_labels


class SpanTargetingDataset(Dataset):
    """
    Класс датасета для задачи манипуляции.
        Параметр sample_size_of_connections отвечает за количество семплируемых connections
        (для того, чтобы унифицировать кол-во связей для обучения по батчам)
    """

    def __init__(self, full_texts, spans, entities, labels):
        
        self.full_texts = full_texts
        self.spans = spans
        self.entities = entities 
        self.labels = labels

    def __getitem__(self, idx):
        
        full_text_ids = torch.tensor(self.full_texts[idx])
        span_ids = torch.tensor(self.spans[idx])
        entity_ids = torch.tensor(self.entities[idx])
        label = torch.tensor(self.labels[idx])
        return full_text_ids, entity_ids, span_ids, label
        
    def __len__(self):
        return len(self.labels)
