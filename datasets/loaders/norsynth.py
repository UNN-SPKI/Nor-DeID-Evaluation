"""
norsynth.py
Implements the logic to load the NorSynthClinical-PHI dataset
as a spaCy DocBin.
"""

import logging
import os
import spacy

def load_norsynth(vocab) -> spacy.tokens.DocBin:
    logging.debug(f'Converting CoNLL to SpaCy...')
    if not os.path.exists('tmp/norsynth/reference_standard_annotated.spacy'):
        os.makedirs('tmp/norsynth/', exist_ok=True)
        spacy.cli.convert(
            'datasets/NorSynthClinical-PHI/reference_standard_annotated.conll',
            'tmp/norsynth/',
            converter="conll",
            file_type="spacy")
    
    logging.debug(f'Retrieving NorSynthClinical-PHI...')
    examples = spacy.tokens.DocBin().from_disk('tmp/norsynth/reference_standard_annotated.spacy')
    map_categories = {
        'Date_Part': 'Date',
        'Date_Full': 'Date',
        'Health_Care_Unit': 'Location'
    }
    mapped_label = lambda l: map_categories[l] if l in map_categories else l
    fixed_docs = []
    for doc in examples.get_docs(vocab):
        fixed_labels = [spacy.tokens.span.Span(doc, s.start, s.end, mapped_label(s.label_)) for s in doc.ents]
        doc.set_ents(fixed_labels)
        fixed_docs.append(doc)
    return spacy.tokens.DocBin(docs=fixed_docs)