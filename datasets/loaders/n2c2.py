"""
n2c2.py
Implements the logic to load the 2006 and 2014 n2c2 challenge datasets
as spaCy DocBins.
"""

import pathlib
from typing import Literal, Union
import xml.etree.ElementTree as ET
import os

import spacy


def load_2006(nlp: spacy.language.Language, directory: str = 'datasets/n2c2/2006/', split: Literal['train', 'test'] = 'test') -> spacy.tokens.DocBin:
    if split == 'test':
        path = pathlib.Path(directory) / 'deid_surrogate_test_all_groundtruth_version2' / 'deid_surrogate_test_all_groundtruth_version2.xml'
        return load_2006_from_file(path, nlp)
    elif split == 'train': 
        path = pathlib.Path(directory) / 'deid_surrogate_test_all_groundtruth_version2' / 'deid_surrogate_test_all_groundtruth_version2.xml'
        return load_2006_from_file(path, nlp)

def load_2006_from_file(path: pathlib.Path, nlp: spacy.language.Language) -> spacy.tokens.DocBin:
    xml_tree = ET.parse(path)
    records = xml_tree.findall('RECORD')
    doc_bin = spacy.tokens.DocBin()
    for record in records:
        contents = ""
        spans = []
        text_elem = record.find('TEXT')
        for elem in text_elem.iter():
            cleaned_text = elem.text.replace('\n', ' ')
            if elem.tag == 'PHI':
                spans.append((len(contents), len(contents) + len(cleaned_text), elem.attrib['TYPE']))
            contents += cleaned_text
            contents += elem.tail.replace('\n', ' ')
        doc = nlp.make_doc(contents)
        # TODO: How do we handle the case where dates are only partly covered?
        ents = [doc.char_span(s[0], s[1], label=s[2], alignment_mode='expand') for s in spans]
        doc.set_ents(ents)
        doc_bin.add(doc)
        breakpoint()
    return doc_bin
