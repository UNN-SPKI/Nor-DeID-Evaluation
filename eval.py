#!/usr/bin/env python3

import time
import os
import logging
import json
from typing import List, Literal

from tap import Tap
import spacy
import spacy.scorer

import datasets.loaders.n2c2
import datasets.loaders.norsynth
import datasets.loaders.synthdeid

import scoring.replacement

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
    output: str = None
    """Which file to write results to"""
    singleClass: bool = False
    """Whether all entities should be put in a single PHI class"""

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

    if args.dataset in ['n2c2-2006', 'n2c2-2014'] and args.model in ['gpt-turbo-chat', 'davinci-edit']:
        raise ValueError("The N2C2 datasets cannot be shared with third parties.")
    
    logging.debug(f'Predicting...')
    answers = model.predict(doc_bin, nlp, args.mode)

    print(f"Results for model {args.model} on dataset {args.dataset}:")
    if args.mode == 'annotate':
        scorer = spacy.scorer.Scorer(nlp)
        if args.singleClass:
            logging.debug("Putting all entities in the PHI class.")
            answers = [_split_example(a) for a in _all_answers_to_label(answers, nlp, 'PHI')]
        evaluation = scorer.score(answers)
        print(evaluation)
    elif args.mode == 'replace':
        scorer = scoring.replacement.Scorer(nlp)
        evaluation = scorer.score(doc_bin, answers)
        print(evaluation)
    else:
        logging.error(f"Unknown mode {args.mode}")
        return
    
    if args.output:
        with open(args.output, 'w', encoding="utf8") as outfile:
            json.dump(evaluation, outfile)

def load_model(model_name: str, prompt: str, args: ExperimentArguments):
    if model_name == 'dummy':
        import models.dummy
        return models.dummy.DummyModel()
    elif model_name == 'davinci-edit':
        import models.davinci_edit
        return models.davinci_edit.DavinciEditModel(prompt, args.openAIKey)
    elif model_name == 'gpt-chat':
        import models.gpt_chat
        return models.gpt_chat.GptChatModel(prompt, args.modelName, args.openAIKey)
    elif model_name == 'hf-transformer':
        import models.hf_transformer
        return models.hf_transformer.HFTransformerModel(prompt, args.modelName)
    elif model_name == 'hf-t5':
        import models.hf_t5
        return models.hf_t5.HFT5Model(prompt, args.modelName)
    elif model_name == 'replicate':
        import models.replicate
        return models.replicate.ReplicateChatModel(args.modelName, prompt)
    elif model_name == 'spacy':
        import models.spacy
        return models.spacy.SpacyModel()
    else:
        raise KeyError(f'Cannot find model {model_name}')

def load_docbin(dataset_path: str) -> spacy.tokens.DocBin:
    logging.debug(f'Loading dataset from path: {dataset_path}')
    return spacy.tokens.DocBin().from_disk(dataset_path)

def load_dataset(dataset_name: str, nlp: spacy.language.Language) -> spacy.tokens.DocBin:
    if dataset_name == 'norsynthclinical':
        return datasets.loaders.norsynth.load_norsynth(nlp.vocab)
    elif dataset_name == 'synthdeid':
        return datasets.loaders.synthdeid.load_synthdeid(nlp.vocab)
    elif dataset_name == 'n2c2-2006':
        return datasets.loaders.n2c2.load_2006(nlp)
    elif dataset_name == 'n2c2-2014':
        return datasets.loaders.n2c2.load_2014(nlp)
    
    if os.path.exists(dataset_name) and dataset_name.endswith('.spacy'):
        return load_docbin(dataset_name)
    
    raise ValueError(f"Could not find dataset identifier and could not find a file at {dataset_name}")

def _all_answers_to_label(answers: List[spacy.training.Example], nlp: spacy.language.Language, label: str = 'PHI') -> List[spacy.training.Example]:
    fixed_examples = []
    for answer in answers:
        
        fixed_labels = [spacy.tokens.span.Span(answer.reference, s.start, s.end, 'PHI') for s in answer.reference.ents]
        fixed_prediction = [spacy.tokens.span.Span(answer.predicted, s.start, s.end, 'PHI') for s in answer.predicted.ents]
        answer.reference.set_ents(fixed_labels)
        answer.predicted.set_ents(fixed_prediction)
        example = spacy.training.Example(answer.predicted, answer.reference)
        fixed_examples.append(example)
    return fixed_examples

def _split_entities(doc):
    """split_entities separates entities in a document into per-token entities."""
    new_doc = doc.copy()
    split_entities = []
    for ent in doc.ents:
        for token in range(ent.start, ent.end):
            new_ent = doc[token:token+1]
            new_ent.label = ent.label
            split_entities.append(new_ent)
    
    new_doc.set_ents(split_entities)
    return new_doc

def _split_example(ex):
    """split_example separates entities in a SpaCy example into individual token-level entities."""
    return spacy.training.Example(_split_entities(ex.predicted), _split_entities(ex.reference))

if __name__ == '__main__':
    args = ExperimentArguments().parse_args()
    main(args)