"""
gpt_chat is a wrapper around OpenAI's gpt-3.5-turbo model
and the redaction and pseudonymization task phrased for the model.
"""

import logging
import re
import time
from typing import List, Tuple

import spacy
import replicate

from joblib import Memory

from models.utilities.tags import list_annotations, remove_tags

CACHE_DIRECTORY = '.cache'
SYSTEM_PROMPT = """
You are assisting a healthcare professional. 
Respond in the original language of the notes, using only words from the notes and the tags specified. 
Do not use any tags which are not specified in the instruction. If a sentence is ambiguous, interpret it as containing sensitive information.
"""
EXPECTED_TAGS = ['First_Name', 'Last_Name', 'Location', 'Health_Care_Unit', 'Age', 'Phone_Number', 'Social_Security_Number', 'Date']

def get_chat_completion(model, system_prompt, prompt, source, temperature, rate_limit = None):
    if rate_limit:
        time.sleep(rate_limit)
    full_prompt = f"""[INST] {prompt} [/INST] 
Input: {source}
Output:"""

    response = replicate.run(model, input={"system_prompt": system_prompt, "prompt": full_prompt}, temperature=temperature)
    
    return ''.join(response)

def fix_orthography(answer: str) -> str:
    space_punctuation = re.sub('\s*([,.])\s+', r' \1 ', answer).rstrip()
    single_spaces = re.sub('\s+', ' ', space_punctuation)
    return single_spaces

class ReplicateChatModel:
    def __init__(self, model, prompt, rate_limit=2, retries=5):
        self._model = model
        self._prompt = prompt
        self._rate_limit = rate_limit
        self._retries = retries
        self._memory = Memory(CACHE_DIRECTORY)

    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language) -> List[spacy.training.Example]:
        examples = []
        for doc in doc_bin.get_docs(language.vocab):
            logging.debug(f"Task: {doc.text}")
            prediction = self.predict_task(doc.text).lstrip()
            logging.debug(f"Predicted: {prediction}")
            if remove_tags(prediction) != doc.text.rstrip():
                logging.warning("Misaligned text!")
                logging.warning(f"ORIGINAL: {doc.text}")
                logging.warning(f"RETURNED: {remove_tags(prediction)}")
            annotations = {'entities': list_annotations(prediction, EXPECTED_TAGS)}
            logging.debug(f"Annotations: {annotations}")

            example = spacy.training.Example.from_dict(doc, annotations)
            examples.append(example)
        return examples

    def predict_task(self, source: str) -> str:
        tries = 0
        temperature = 0.0
        while tries < self._retries:
            get_cached_completion = self._memory.cache(get_chat_completion)
            response = get_cached_completion(self._model, SYSTEM_PROMPT, self._prompt, source, temperature, self._rate_limit)
            if len(response) == 0:
                logging.error(
                    "Unexpected answer from Replicate - empty answer")
                temperature += 0.01
                tries += 1
                continue

            return fix_orthography(response)

        logging.error(f'Could not get an edit after {self._retries} tries.')
        return ''
