"""
synthdeid.py
Implements the logic to load our synthetic dataset.
"""

import logging
import os
import pathlib
import spacy

def load_synthdeid(vocab, split = 'holdout') -> spacy.tokens.DocBin:
    logging.debug(f'Retrieving NorSynthClinical-PHI...')
    path = pathlib.Path('datasets/nor-deid-synthdata')
    if split == 'holdout':
        path = path / 'holdout.spacy'
    elif split == 'validation':
        path = path / 'validation.spacy'
    elif split == 'training':
        path = path / 'training.spacy'
    else:
        raise ValueError(f"Unknown split {split}")
    examples = spacy.tokens.DocBin().from_disk(path)
    map_categories = {
        'Health_Care_Unit': 'Location'
    }
    mapped_label = lambda l: map_categories[l] if l in map_categories else l
    fixed_docs = []
    for doc in examples.get_docs(vocab):
        fixed_labels = [spacy.tokens.span.Span(doc, s.start, s.end, mapped_label(s.label_)) for s in doc.ents]
        doc.set_ents(fixed_labels)
        fixed_docs.append(doc)
    return spacy.tokens.DocBin(docs=fixed_docs)