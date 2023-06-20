"""
gpt_chat is a wrapper around OpenAI's gpt-3.5-turbo model
and the redaction and pseudonymization task phrased for the model.
"""

import logging
import re
import time
from typing import List, Tuple

import requests
import spacy

from joblib import Memory

from models.utilities.tags import list_annotations, remove_tags

CACHE_DIRECTORY = '.cache'

SYSTEM_INSTRUCTION = """Anonymize the following clinical note with tags.
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
"""

EXPECTED_TAGS = ['First_Name', 'Last_Name', 'Location', 'Health_Care_Unit', 'Age', 'Phone_Number', 'Social_Security_Number', 'Date']

def get_chat_completion(source, openAIAPIKey, temperature, rate_limit = None):
    if rate_limit:
        time.sleep(rate_limit)
    prologue = [
        {'role': 'system', 'content': SYSTEM_INSTRUCTION},
        {'role': 'user', 'content': 'Input: Georg Nordmann er 47 år gammel og innlagt på Haukeland siden 3. april . Georgs kone Åshild ønsker at vi ringer henne på telefon 770 12345 når vi vet mer .: '},
        {'role': 'assistant', 'content': '<First_Name>Georg</First_Name> <Last_Name>Nordmann</Last_Name> er <Age>47 år gammel</Age> og innlagt på <Location>Haukeland</Location> siden <Date>3. april</Date> . <First_Name>Georgs</First_Name> kone <First_Name>Åshild</First_Name> ønsker at vi ringer henne på telefon <Phone_Number>770 12345</Phone_Number> når vi vet mer .'},
    ]
    messages = prologue + [{
        'role': 'user', 'content': 'Input: ' + source
    }]
    r = requests.post('https://api.openai.com/v1/chat/completions',
                      json={
                          'model': 'gpt-3.5-turbo',
                          'messages': messages,
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

def fix_orthography(answer: str) -> str:
    space_punctuation = re.sub('\s*([,.])\s+', r' \1 ', answer).rstrip()
    single_spaces = re.sub('\s+', ' ', space_punctuation)
    return single_spaces

class GptChatModel:
    def __init__(self, openAIAPIKey, rate_limit=2, retries=5):
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
            response = get_cached_completion(source, self._openAIAPIKey, temperature, self._rate_limit)
            if 'choices' not in response:
                logging.error(
                    "Unexpected answer from OpenAI - could not find \'choices\'")
                temperature += 0.01
                tries += 1
                continue

            answer = response['choices'][0]['message']['content']
            return fix_orthography(answer)

        logging.error(f'Could not get an edit after {self._retries} tries.')
        return ''
