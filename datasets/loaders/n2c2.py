"""
n2c2.py
Implements the logic to load the 2006 and 2014 n2c2 challenge datasets
as spaCy DocBins.
"""

from typing import Literal, Union
import xml.etree.ElementTree as ET

import spacy


def load_2006(directory: str, nlp: spacy.language.Language, split: Literal['train', 'test'] = 'test') -> spacy.DocBin:
    pass

def load_2006_from_file(path: str, nlp: spacy.language.Language) -> spacy.tokens.DocBin:
    xml_tree = ET.parse(path)
    records = xml_tree.findall('RECORD')
    doc_bin = spacy.tokens.DocBin()
    for record in records:
        contents = ""
        entities = []
        text_elem = record.find('TEXT')
        for elem in text_elem.iter():
            if elem.tag == 'PHI':
                entities.append((len(contents), len(contents) + len(elem.text), elem.attrib['TYPE']))
            contents += elem.text
            contents += elem.tail
        doc = nlp.make_doc(contents)
        biluo_ents = spacy.training.offsets_to_biluo_tags(entities)
        doc.ents = biluo_ents
        doc_bin.add(doc)
    return doc_bin
