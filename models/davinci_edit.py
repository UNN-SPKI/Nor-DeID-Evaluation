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

ANNOTATION_PROMPT = """
Annotate the following clinical note with XML-style tags.
Enclose any strings that might be a name or acronym or initials, patients' names, doctors' names, the names of the M.D. or Dr. with a pair of <NAME> tags. 
Enclose any pager names and medical staff names with <Name> tags. 
Enclose any strings that might be a location or address, such as \"Åssiden 31\" with <Location> tags. 
Enclose the patient's ages and any strings that look like \"X år gammel\" with <Age> tags. 
Enclose clinical and hospital names with <Health_Care_Unit> tags. 
Enclose phone numbers and 8 digit long numbers with <Phone> tags. 
Enclose social security numbers and 11 digit long numbers with <Social_Security_Number> tags. 
Enclose dates and times with <Date> tags.
"""

# _ENCLOSED_IN_TAGS matches on expressions with XML-style tags (e.g. '<Age>23</Age>')
# putting the tag name in the first capturing group, and the contents in the second
# capturing group.
# NOTE: This will fail if you have nested annotations.
_ENCLOSED_IN_TAGS = re.compile(r'<([\w_]*)>([^<]*)<\/\1>')

def remove_tags(task: str) -> str:
    return _ENCLOSED_IN_TAGS.sub(r'\2', task)

def list_annotations(annotated: str) -> List[Tuple[int, int, str]]:
    annotations = []
    matches = _ENCLOSED_IN_TAGS.finditer(annotated)

    # We want to find the character spans as they will be
    # in the unannotated text. To achieve this, we keep a running
    # count of how many markup characters have found so far
    # in markup_offset:
    markup_offset = 0
    for match in matches:
        tag_name, contents = match.groups()
        # The annotations consist of an opening tag and a closing tag
        # (e.g. <Age></Age>) - the tags themselves add 5 characters:
        total_markup_chars = (2*len(tag_name) + 5)
        tag_start = match.span()[0] - markup_offset
        tag_end = match.span()[1] - markup_offset - total_markup_chars
        markup_offset += total_markup_chars
        annotations.append((tag_start, tag_end, tag_name))
    return annotations

class DavinciEditModel:
    def __init__(self, openAIAPIKey, rate_limit = 2, retries = 5):
        self._openAIAPIKey = openAIAPIKey
        self._rate_limit = rate_limit
        self._retries = retries
    
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language) -> List[spacy.training.Example]:
        examples = []
        for doc in doc_bin.get_docs(language.vocab):
            logging.debug(f"Task: {doc.text}")
            prediction = self.predict_task(doc.text)
            logging.debug(f"Predicted: {prediction}")
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
            if self._rate_limit:
                time.sleep(self._rate_limit)
            r = requests.post('https://api.openai.com/v1/edits',
                json={
                    'model': 'text-davinci-edit-001',
                    'input': source,
                    'instruction': instruction,
                    'temperature': temperature
                },
                headers={
                    'Authorization': f'Bearer {self._openAIAPIKey}',
                    'Content-Type': 'application/json'
                })
            if r.status_code != requests.codes.ok:
                logging.error(f"Got status code {r.status_code} from OpenAI.")
            response = r.json()
            if 'choices' not in response:
                logging.error("Unexpected answer from OpenAI - could not find \'choices\'")
                temperature += 0.01
                tries += 1
                continue

            answer = response['choices'][0]['text']
            return answer
        
        logging.error(f'Could not get an edit after {self._retries} tries.')
        return ''