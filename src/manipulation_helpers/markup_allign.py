import re
from typing import Any, List

import spacy


def allign_markup(text: str, markup: List[Any], nlp, remove_spaces: bool = True):
    text = re.sub("\s+", " ", text)
    doc = nlp(text)
    spans, not_found_entities = [], []

    for example in markup:
        example_text = re.sub(" \*\(\d+\)\s+([\w\s]+?)\* ", "\g<1>", example["text"])
        example_text = re.sub("\s?\*\(\d+\)\s?", "", example_text)
        example_text = re.sub("(?:^| )\(?\d+\)", " ", example_text)
        example_text = example_text.replace("* ", "").replace(" *", "").strip("*")
        example_text = re.sub("\s+", " ", example_text).strip()
        
        class_name = re.sub("\(.*\)", "", example["class_name"])
        if remove_spaces:
            class_name = class_name.strip().replace(" ", "_")
        start_idx = text.find(example_text)
        if example_text and start_idx != -1:
        # markup_offsets.append((start_idx, start_idx + len(example_text), class_name))
            spans.append(doc.char_span(start_idx, start_idx + len(example_text), class_name, alignment_mode="expand"))
        else:
            print("Не найдена сущность '{}'".format(example["text"]))
            not_found_entities.append(example)
            
    if not len(spans):
        print("Нет сущностей")
    spans = spacy.util.filter_spans(spans)
    spans = [x for x in spans if x is not None]

    return doc, spans
