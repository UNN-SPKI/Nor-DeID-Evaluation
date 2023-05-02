#!/usr/bin/env python3
"""
eval.py calculates the standard NER metrics (precision, recall, F1) for an NER model
applied to a deidentification dataset.
"""
from typing import List, Tuple
import spacy
import re

# The model is not used for NER
SPACY_MODEL = "nb_core_news_sm"

nlp = spacy.load(SPACY_MODEL, disable=["parser"])

# _ENCLOSED_IN_TAGS matches on expressions with XML-style tags (e.g. '<Age>23</Age>')
# putting the tag name in the first capturing group, and the contents in the second
# capturing group.
# NOTE: This will fail if you have nested annotations.
_ENCLOSED_IN_TAGS = re.compile(r'<([\w_]*)>([^<]*)<\/\1>')

def remove_tags(task: str) -> str:
    return _ENCLOSED_IN_TAGS.sub(r'\2', task)

def list_annotations(task: str) -> List[Tuple[int, int, str]]:
    matches = _ENCLOSED_IN_TAGS.match(task)
    

def create_examples():
    examples = []
    with open('datasets/NorSynthClinical-PHI/reference_standard_annotated.txt') as reference_file:
        for reference in reference_file:
            cleaned = reference.strip()
            