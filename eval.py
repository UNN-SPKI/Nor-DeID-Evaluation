#!/usr/bin/env python3

import os
import models.dummy
import models.davinci_edit
import spacy
import logging
from tap import Tap

logging.basicConfig(level=logging.DEBUG)

class ExperimentArguments(Tap):
    dataset: str = 'norsynthclinical' # The identifier of the dataset to use (see load_dataset in eval.py)
    model: str = 'dummy' # The identifier of the model to use (see load_model in eval.py)
    spacyPipeline: str = 'nb_core_news_sm' # The SpaCy Language to use for tokenization
    openAIKey: str = 'OPENAI_KEY_HERE' # OpenAI key for comparison models

def main(args: ExperimentArguments):
    logging.debug(f'Loading pipeline {args.spacyPipeline}')
    nlp = spacy.load(args.spacyPipeline)
    
    logging.debug(f'Loading model {args.model}')
    model = load_model(args.model, args)

    logging.debug(f'Loading dataset {args.dataset}')
    doc_bin = load_dataset(args.dataset)
    
    logging.debug(f'Predicting...')
    answers = model.predict(doc_bin, nlp)

    print(f"Results for model {args.model} on dataset {args.dataset}:")
    evaluation = nlp.evaluate(answers)
    print(nlp.evaluate(answers))

def load_model(model_name: str, args: ExperimentArguments):
    if model_name == 'dummy':
        return models.dummy.DummyModel()
    elif model_name == 'davinci-edit':
        return models.davinci_edit.DavinciEditModel(args.openAIKey)
    else:
        raise KeyError(f'Cannot find model {model_name}')

def load_dataset(dataset_name: str) -> spacy.tokens.DocBin:
    if dataset_name != 'norsynthclinical':
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    logging.debug(f'Converting CoNLL to SpaCy...')
    if not os.path.exists('tmp/norsynth/reference_standard_annotated.spacy'):
        os.makedirs('tmp/norsynth/', exist_ok=True)
        spacy.cli.convert(
            'datasets/NorSynthClinical-PHI/reference_standard_annotated.conll',
            'tmp/norsynth/',
            converter="conll",
            file_type="spacy")
    
    logging.debug(f'Retrieving NorSynthClinical-PHI...')
    return spacy.tokens.DocBin().from_disk('tmp/norsynth/reference_standard_annotated.spacy')

if __name__ == '__main__':
    args = ExperimentArguments().parse_args()
    main(args)