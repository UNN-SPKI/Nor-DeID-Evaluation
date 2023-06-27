#!/usr/bin/env python3

import time
import os
import logging
import json

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
    dataset: str = 'norsynthclinical' # The identifier of the dataset to use (see load_dataset in eval.py)
    model: str = 'dummy' # The identifier of the model to use (see load_model in eval.py)
    modelName: str = 'NbAiLab/nb-gpt-j-6B-alpaca' # For models which expect a path, load the model here
    spacyPipeline: str = 'nb_core_news_sm' # The SpaCy Language to use for tokenization
    openAIKey: str = 'OPENAI_KEY_HERE' # OpenAI key for comparison models

def main(args: ExperimentArguments):
    logging.debug(f'Loading pipeline {args.spacyPipeline}')
    nlp = spacy.load(args.spacyPipeline, enable=['ner'])
    
    logging.debug(f'Loading model {args.model}')
    model = load_model(args.model, args)

    logging.debug(f'Loading dataset {args.dataset}')
    doc_bin = load_dataset(args.dataset, nlp)

    if args.dataset in ['n2c2-2006', 'n2c2-2014'] and args.model in ['gpt-turbo-chat', 'davinci-edit']:
        raise ValueError("The N2C2 datasets cannot be shared with third parties.")
    
    logging.debug(f'Predicting...')
    answers = model.predict(doc_bin, nlp)

    print(f"Results for model {args.model} on dataset {args.dataset}:")
    scorer = spacy.scorer.Scorer(nlp)
    evaluation = scorer.score(answers)
    print(evaluation)

def load_model(model_name: str, args: ExperimentArguments):
    if model_name == 'dummy':
        return models.dummy.DummyModel()
    elif model_name == 'davinci-edit':
        return models.davinci_edit.DavinciEditModel(args.openAIKey)
    elif model_name == 'gpt-turbo-chat':
        return models.gpt_chat.GptChatModel(args.openAIKey)
    elif model_name == 'hf-transformer':
        return models.hf_transformer.HFTransformerModel(args.modelName)
    elif model_name == 'hf-t5':
        return models.hf_t5.HFT5Model(args.modelName)
    else:
        raise KeyError(f'Cannot find model {model_name}')

def load_docbin(dataset_path: str) -> spacy.tokens.DocBin:
    logging.debug(f'Loading dataset from path: {dataset_path}')
    return spacy.tokens.DocBin().from_disk(dataset_path)

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