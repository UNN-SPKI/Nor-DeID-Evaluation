"""
hf_transformer is a wrapper around HuggingFace's Transformer library,
loading a specified model which implements AutoModelForCausalLM
(e.g. )
"""

import logging
import re
import time
from typing import List, Tuple

import spacy
import transformers
import accelerate

ANNOTATION_PROMPT = """
Annotate the following clinical notes with XML-style tags.
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

class HFTransformerModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        with accelerate.init_empty_weights():
           self.model = transformers.AutoModelForCausalLM.from_config(config)
        
        self.model.tie_weights()
        accelerate.load_checkpoint_and_dispatch(
            self.model, model_name, device_map="auto", no_split_module_classes=["GPTJBlock"]
        )
    
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language) -> List[spacy.training.Example]:
        examples = []
        for doc in doc_bin.get_docs(language.vocab):
            logging.debug(f"Task: {doc.text}")
            start = time.time()
            prediction = fix_orthography(self.predict_task(doc.text))
            inference_time = time.time() - start
            logging.debug(f"Finished in {inference_time} seconds.")

            logging.debug(f"Predicted: {prediction}")
            annotations = {'entities': list_annotations(prediction)}
            logging.debug(f"Annotations: {annotations}")

            example = spacy.training.Example.from_dict(doc, annotations)
            examples.append(example)
        return examples
    
    def predict_task(self, source: str) -> str:
        ANNOTATION_TASK = f"""Nedenfor er en instruksjon som beskriver en oppgave, sammen med et input som gir ytterligere kontekst. Skriv et svar som fullfører forespørselen på riktig måte.

### Instruksjon:
{ANNOTATION_PROMPT}

### Input:
{source}

### Respons:"""
        inputs = self.tokenizer(ANNOTATION_TASK, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=transformers.GenerationConfig(temperature=0.01, top_p=0.05, num_beams=1),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = []
        for seq in generation_output.sequences:
            output = self.tokenizer.decode(seq, skip_special_tokens=True)
            response.append(output.split("### Respons:")[-1].strip())
        return ''.join(response)
        