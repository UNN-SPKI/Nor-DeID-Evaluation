#!/usr/bin/env python3
"""
eval.py calculates the standard NER metrics (precision, recall, F1) for an NER model
applied to a deidentification dataset.
"""
from typing import List, Tuple
import spacy
import spacy.training
import re

# The model is not used for NER
SPACY_MODEL = "nb_core_news_sm"

nlp = spacy.load(SPACY_MODEL)

# _ENCLOSED_IN_TAGS matches on expressions with XML-style tags (e.g. '<Age>23</Age>')
# putting the tag name in the first capturing group, and the contents in the second
# capturing group.
# NOTE: This will fail if you have nested annotations.
_ENCLOSED_IN_TAGS = re.compile(r'<([\w_]*)>([^<]*)<\/\1>')

def remove_tags(task: str) -> str:
    return _ENCLOSED_IN_TAGS.sub(r'\2', task)

def list_annotations(task: str) -> List[Tuple[int, int, str]]:
    annotations = []
    matches = _ENCLOSED_IN_TAGS.finditer(task)

    # We want to find the character spans as they will be
    # in the unannotated text. To achieve this, we keep a running
    # count of how many markup characters have found so far
    # in markup_offset:
    markup_offset = 0
    for match in matches:
        tag_name, contents = match.groups()
        total_markup_chars = (2*len(tag_name) + 4)
        tag_start = match.span()[0] - markup_offset
        tag_end = match.span()[1] - markup_offset - total_markup_chars - 1
        markup_offset += total_markup_chars
        annotations.append((tag_start, tag_end, tag_name))
    return annotations

def create_examples():
    examples = []
    with open('datasets/NorSynthClinical-PHI/reference_standard_annotated.txt', encoding='utf-8') as reference_file:
        for reference in reference_file:
            cleaned = reference.strip()
            no_tags = remove_tags(cleaned)
            doc = nlp.make_doc(no_tags)
            entities = list_annotations(cleaned)
            annotations = {'entities': entities}
            example = spacy.training.Example.from_dict(doc, annotations)
            examples.append(example)
    return examples

if __name__ == '__main__':
    examples = create_examples()