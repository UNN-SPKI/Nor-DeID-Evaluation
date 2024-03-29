"""
n2c2.py
Implements the logic to load the 2006 and 2014 n2c2 challenge datasets
as spaCy DocBins.
"""

import logging
import pathlib
from typing import Literal
import os
import linecache 

import xml.etree.ElementTree as ET

import spacy

# Version 2 of the 2006 training set has an error on line 25722
# which makes the XML malformed.
# See fix_2006_training_set.
# To fix it, we replace the PHI marker on line 25722
PATCH_LINE_INDEX = 25722
# with a closing tag:
PATCH_FIX = ('<PHI TYPE="DOCTOR">', '</PHI>')

def load_2006(nlp: spacy.language.Language, directory: str = 'datasets/n2c2/2006/', split: Literal['train', 'test'] = 'test') -> spacy.tokens.DocBin:
    """load_2006 returns the n2c2 2006 deidentification challenge dataset as a spaCy DocBin."""
    if split == 'test':
        path = pathlib.Path(directory) / 'deid_surrogate_test_all_groundtruth_version2' / 'deid_surrogate_test_all_groundtruth_version2.xml'
        return load_2006_from_file(path, nlp)
    elif split == 'train': 
        path = pathlib.Path(directory) / 'deid_surrogate_train_all_version2' / 'deid_surrogate_train_all_version2.xml'
        error_line = linecache.getline(str(path), PATCH_LINE_INDEX)
        if PATCH_FIX[0] in error_line:
            fixed_path = pathlib.Path(directory) / 'deid_surrogate_train_all_version2' / 'deid_surrogate_train_all_version2_fixed.xml.tmp'
            logging.debug(f'Malformed XML in training set, patching and saving fixed version to {fixed_path}')
            fix_2006_training_set(path, fixed_path)
            return load_2006_from_file(fixed_path, nlp)
        else:
            return load_2006_from_file(path, nlp)

def fix_2006_training_set(source: pathlib.Path, target: pathlib.Path):
    """fix_2006_training_set patches version 2 of the n2c2 2006 challenge training set
    to make the XML well-formed, saving the fixed version in target."""
    if os.path.exists(target):
        logging.debug("Patched version already exists.")
        return
    with open(target, 'w', encoding='utf8') as target_file:
        with open(source, 'r', encoding='utf8') as source_file:
            for i, line in enumerate(source_file.readlines()):
                if i == PATCH_LINE_INDEX - 1:
                    logging.debug(f"Patching line {PATCH_LINE_INDEX}...")
                    fixed_line = line.replace(PATCH_FIX[0], PATCH_FIX[1])
                    target_file.write(fixed_line)
                else:
                    target_file.write(line)


def load_2006_from_file(path: pathlib.Path, nlp: spacy.language.Language) -> spacy.tokens.DocBin:
    """load_2006 returns a file from the n2c2 2006 deidentification challenge dataset as a spaCy DocBin."""
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
    return doc_bin

def load_2014(nlp: spacy.language.Language, directory: str = 'datasets/n2c2/2014/', split: Literal['train', 'test'] = 'test') -> spacy.tokens.DocBin:
    if split == 'test':
        test_set_path = pathlib.Path(directory) / 'testing-PHI-Gold-fixed/'
        return _load_2014_from_folder(test_set_path, nlp)
    elif split == 'train':
        train_set1_path = pathlib.Path(directory) / 'training-PHI-Gold-Set1/'
        set1 = _load_2014_from_folder(train_set1_path, nlp)
        train_set2_path = pathlib.Path(directory) / 'training-PHI-Gold-Set2/'
        set2 = _load_2014_from_folder(train_set2_path, nlp)
        set1.merge(set2)
        return set1
    else:
        raise ValueError(f"Unknown dataset split '{split}' in n2c2 2014")

def _load_2014_from_folder(path: pathlib.Path, nlp: spacy.language.Language) -> spacy.tokens.DocBin:
    docs = spacy.tokens.DocBin()
    for doc_path in os.listdir(path):
        if not doc_path.endswith('.xml'):
            continue
        full_path = path / doc_path
        new_doc = _load_2014_from_document(full_path, nlp)
        docs.add(new_doc)
    return docs

def _load_2014_from_document(path: pathlib.Path, nlp: spacy.language.Language):
    xml_tree = ET.parse(path)
    root = xml_tree.getroot()
    text = root.find('TEXT').text
    doc = nlp.make_doc(text)
    tags = root.find('TAGS')
    spans = []
    for elem in tags.iter():
        if elem.tag == 'TAGS':
            continue
        # TODO: How do we handle the case where dates are only partly covered?
        span = doc.char_span(int(elem.attrib['start']), int(elem.attrib['end']), label=elem.attrib['TYPE'])
        if span is not None:
            spans.append(span)
    doc.set_ents(spans)
    return doc