"""
davinci_edit is a wrapper around OpenAI's text-davinci-edit-001 model
and the redaction and pseudonymization task phrased for the model.
"""

import logging
import re
import time
from typing import List, Tuple

import requests
import spacy

from joblib import Memory

from models.utilities.alignment import fix_orthography
from models.utilities.tags import list_annotations, remove_tags

CACHE_DIRECTORY = '.cache'

USE_EXAMPLE = False
EXAMPLE = """Example:
Georg Nordmann er 47 år gammel og innlagt på Haukeland siden 3. april . Georgs kone Åshild ønsker at vi ringer henne på telefon 770 12345 når vi vet mer .

Result:
<First_Name>Georg</First_Name> <Last_Name>Nordmann</Last_Name> er <Age>47 år gammel</Age> og innlagt på <Location>Haukeland</Location> siden <Date>3. april</Date> . <First_Name>Georgs</First_Name> kone <First_Name>Åshild</First_Name> ønsker at vi ringer henne på telefon <Phone_Number>770 12345</Phone_Number> når vi vet mer ."""

ANNOTATION_PROMPT = """Anonymize the following clinical note with tags.
Enclose first names with <First_Name> tags. 
Enclose last names with <Last_Name> tags.
Enclose any strings that might be a location or address, such as "Åssiden 31" with <Location> tags. 
Enclose clinical and hospital names with <Location> tags. 
Enclose the patient's age and any texts that look like "X år gammel" with <Age> tags. 
Enclose phone numbers with <Phone_Number> tags.
Enclose 8 digit long numbers with <Phone_Number> tags. 
Enclose social security numbers with <Social_Security_Number> tags.
Enclose 11 digit long numbers with <Social_Security_Number> tags. 
Enclose dates and times with <Date> tags.
Do not use any tags which were not specified above.
""" + (EXAMPLE if USE_EXAMPLE else "")

EXPECTED_TAGS = ['First_Name', 'Last_Name', 'Location', 'Health_Care_Unit', 'Age', 'Phone_Number', 'Social_Security_Number', 'Date']

def get_completion(source, instruction, openAIAPIKey, temperature, rate_limit = None):
    if rate_limit:
        time.sleep(rate_limit)
    r = requests.post('https://api.openai.com/v1/edits',
        json={
            'model': 'text-davinci-edit-001',
            'input': source,
            'instruction': instruction,
            'temperature': temperature
        },
        headers={
            'Authorization': f'Bearer {openAIAPIKey}',
            'Content-Type': 'application/json'
        })
    if r.status_code != requests.codes.ok:
        logging.error(f"Got status code {r.status_code} from OpenAI.")
    response = r.json()
    return response



class DavinciEditModel:
    def __init__(self, openAIAPIKey, rate_limit = 2, retries = 5):
        self._openAIAPIKey = openAIAPIKey
        self._rate_limit = rate_limit
        self._retries = retries
        self._memory = Memory(CACHE_DIRECTORY)
    
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language) -> List[spacy.training.Example]:
        examples = []
        for doc in doc_bin.get_docs(language.vocab):
            logging.debug(f"Task: {doc.text}")
            prediction = self.predict_task(doc.text)
            logging.debug(f"Predicted: {prediction}")
            if remove_tags(prediction) != doc.text.rstrip():
                logging.warning("Misaligned text!")
                logging.warning(f"ORIGINAL: {doc.text}")
                logging.warning(f"RETURNED: {remove_tags(prediction)}")
            annotations = {'entities': list_annotations(prediction)}
            logging.debug(f"Annotations: {annotations}")

            example = spacy.training.Example.from_dict(doc, annotations)
            examples.append(example)
        return examples
    
    def predict_task(self, source: str) -> str:
        tries = 0
        temperature = 0.0 
        instruction = ANNOTATION_PROMPT
        while tries < self._retries:
            get_cached_completion = self._memory.cache(get_completion)
            response = get_cached_completion(source, instruction, self._openAIAPIKey, temperature, self._rate_limit)
            if 'choices' not in response:
                logging.error("Unexpected answer from OpenAI - could not find \'choices\'")
                temperature += 0.01
                tries += 1
                continue

            answer = response['choices'][0]['text']
            return fix_orthography(answer)
        
        logging.error(f'Could not get an edit after {self._retries} tries.')
        return ''