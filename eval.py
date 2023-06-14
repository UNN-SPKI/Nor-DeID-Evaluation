#!/usr/bin/env python3

import os
import spacy
import logging
from tap import Tap

import models.dummy
import models.davinci_edit
import models.gpt_chat
import models.hf_transformer

logging.basicConfig(level=logging.DEBUG)

class ExperimentArguments(Tap):
    dataset: str = 'norsynthclinical' # The identifier of the dataset to use (see load_dataset in eval.py)
    model: str = 'dummy' # The identifier of the model to use (see load_model in eval.py)
    modelName: str = 'NbAiLab/nb-gpt-j-6B-alpaca' # For models which expect a path, load the model here
    spacyPipeline: str = 'nb_core_news_sm' # The SpaCy Language to use for tokenization
    openAIKey: str = 'OPENAI_KEY_HERE' # OpenAI key for comparison models

def main(args: ExperimentArguments):
    logging.debug(f'Loading pipeline {args.spacyPipeline}')
    nlp = spacy.load(args.spacyPipeline)
    
    logging.debug(f'Loading model {args.model}')
    model = load_model(args.model, args)

    logging.debug(f'Loading dataset {args.dataset}')
    doc_bin = load_dataset(args.dataset, nlp.vocab)
    
    logging.debug(f'Predicting...')
    answers = model.predict(doc_bin, nlp)

    print(f"Results for model {args.model} on dataset {args.dataset}:")
    evaluation = nlp.evaluate(answers)
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
    else:
        raise KeyError(f'Cannot find model {model_name}')

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

def load_n2c2_2014() -> spacy.tokens.DocBin:
    return

def load_docbin(dataset_path: str) -> spacy.tokens.DocBin:
    logging.debug(f'Loading dataset from path: {dataset_path}')
    return spacy.tokens.DocBin().from_disk(dataset_path)

def load_dataset(dataset_name: str, vocab) -> spacy.tokens.DocBin:
    if dataset_name == 'norsynthclinical':
        return load_norsynth(vocab)
    elif dataset_name == 'n2c2-2014':
        return load_n2c2_2014()
    
    if os.path.exists(dataset_name) and dataset_name.endswith('.spacy'):
        return load_docbin(dataset_name)
    
    raise ValueError(f"Could not find dataset identifier and could not find a file at {dataset_name}")

if __name__ == '__main__':
    args = ExperimentArguments().parse_args()
    main(args)