#!/usr/bin/env python3

import time
import os
import logging
import json
from typing import Literal

from tap import Tap
import spacy
import spacy.scorer

import models.dummy
import models.davinci_edit
import models.gpt_chat
import models.hf_transformer
import models.hf_t5

import datasets.loaders.n2c2
import datasets.loaders.norsynth

logging.basicConfig(level=logging.DEBUG)

class ExperimentArguments(Tap):
    prompt_path: str
    """Path to the prompt template to use.""" 
    mode: Literal['replace', 'annotate']
    """Which setting to evaluate the method in."""
    dataset: str = 'norsynthclinical'
    """Which dataset loader to use. (see load_dataset in eval.py)"""
    model: str = 'dummy'
    """The model type to use (see load_model in eval.py)"""
    modelName: str = 'NbAiLab/nb-gpt-j-6B-alpaca'
    """The identifier/path to the model"""
    spacyPipeline: str = 'nb_core_news_sm'
    """The SpaCy Language to use for tokenization"""
    openAIKey: str = 'OPENAI_KEY_HERE'
    """OpenAI key for comparison models"""
    singleClass: bool = False
    """Whether to force all entities into a single 'PHI' class"""

def main(args: ExperimentArguments):
    with open(args.prompt_path, 'r', encoding="utf-8") as prompt_file:
        logging.debug(f'Using prompt {args.prompt_path}')
        prompt = prompt_file.read()

    logging.debug(f'Loading pipeline {args.spacyPipeline}')
    nlp = spacy.load(args.spacyPipeline, enable=['ner'])
    
    logging.debug(f'Loading model {args.model}')
    model = load_model(args.model, prompt, args)

    logging.debug(f'Loading dataset {args.dataset}')
    doc_bin = load_dataset(args.dataset, nlp)
    if args.singleClass:
        logging.debug('Setting all labels to \'PHI\'')
        doc_bin = _all_ents_to_label(doc_bin, nlp, 'PHI')

    if args.dataset in ['n2c2-2006', 'n2c2-2014'] and args.model in ['gpt-turbo-chat', 'davinci-edit']:
        raise ValueError("The N2C2 datasets cannot be shared with third parties.")
    
    logging.debug(f'Predicting...')
    answers = model.predict(doc_bin, nlp)

    print(f"Results for model {args.model} on dataset {args.dataset}:")
    scorer = spacy.scorer.Scorer(nlp)
    evaluation = scorer.score(answers)
    print(evaluation)

def load_model(model_name: str, prompt: str, args: ExperimentArguments):
    if model_name == 'dummy':
        return models.dummy.DummyModel()
    elif model_name == 'davinci-edit':
        return models.davinci_edit.DavinciEditModel(prompt, args.openAIKey)
    elif model_name == 'gpt-chat':
        return models.gpt_chat.GptChatModel(prompt, args.modelName, args.openAIKey)
    elif model_name == 'hf-transformer':
        return models.hf_transformer.HFTransformerModel(prompt, args.modelName)
    elif model_name == 'hf-t5':
        return models.hf_t5.HFT5Model(prompt, args.modelName)
    else:
        raise KeyError(f'Cannot find model {model_name}')

def load_docbin(dataset_path: str) -> spacy.tokens.DocBin:
    logging.debug(f'Loading dataset from path: {dataset_path}')
    return spacy.tokens.DocBin().from_disk(dataset_path)

def _all_ents_to_label(docs: spacy.tokens.DocBin, nlp: spacy.language.Language, label: str = 'PHI') -> spacy.tokens.DocBin:
    fixed_docs = []
    for doc in docs.get_docs(nlp.vocab):
        fixed_labels = [spacy.tokens.span.Span(doc, s.start, s.end, 'PHI') for s in doc.ents]
        doc.set_ents(fixed_labels)
        fixed_docs.append(doc)
    return spacy.tokens.DocBin(docs=fixed_docs)

def load_dataset(dataset_name: str, nlp: spacy.language.Language) -> spacy.tokens.DocBin:
    if dataset_name == 'norsynthclinical':
        return datasets.loaders.norsynth.load_norsynth(nlp.vocab)
    elif dataset_name == 'n2c2-2006':
        return datasets.loaders.n2c2.load_2006(nlp)
    elif dataset_name == 'n2c2-2014':
        return datasets.loaders.n2c2.load_2014(nlp)
    
    if os.path.exists(dataset_name) and dataset_name.endswith('.spacy'):
        return load_docbin(dataset_name)
    
    raise ValueError(f"Could not find dataset identifier and could not find a file at {dataset_name}")

if __name__ == '__main__':
    args = ExperimentArguments().parse_args()
    main(args)