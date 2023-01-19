from typing import Any, Dict, List, NamedTuple

from pymorphy2 import MorphAnalyzer
from razdel import tokenize

pymorphy = MorphAnalyzer()


class Substring(NamedTuple):
    start: int
    end: int
    text: str
    lemma: str


def preprocess_text(text: str) -> List[Substring]:
    result: List[Substring] = []
    for token in tokenize(text):
        p = pymorphy.parse(token.text)[0]
        result.append(Substring(token.start, token.stop, token.text, p.normal_form))
    return result


def find_entites_offsets(text: str, entities: List[str]) -> List[Dict[str, Any]]:
    index, entity_id = 0, 0
    current_entity = preprocess_text(entities[entity_id])
    token_start = None

    entity_spans = []
    preprocessed_text = preprocess_text(text)

    for token in preprocessed_text:
        if current_entity[index].lemma == token.lemma:
            index += 1
            if token_start is None:
                token_start = token.start

        if index >= len(current_entity):
            entity_spans.append(
                {"from": token_start, "to": token.end, "name": entities[entity_id]}
            )
            token_start = None
            entity_id += 1
            if entity_id >= len(entities):
                break
            current_entity = preprocess_text(entities[entity_id])
            index = 0
    return entity_spans
