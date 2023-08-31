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

from models.utilities.alignment import fix_orthography
from models.utilities.tags import list_annotations

EXPECTED_TAGS = ['First_Name', 'Last_Name', 'Location', 'Health_Care_Unit', 'Age', 'Phone_Number', 'Social_Security_Number', 'Date']

class HFTransformerModel:
    def __init__(self, prompt: str, model_name: str):
        self.prompt = prompt
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
            annotations = {'entities': list_annotations(prediction, EXPECTED_TAGS)}
            logging.debug(f"Annotations: {annotations}")

            example = spacy.training.Example.from_dict(doc, annotations)
            examples.append(example)
        return examples
    
    def predict_task(self, source: str) -> str:
        ANNOTATION_TASK = f"""Nedenfor er en instruksjon som beskriver en oppgave, sammen med et input som gir ytterligere kontekst. Skriv et svar som fullfører forespørselen på riktig måte.

### Instruksjon:
{self.prompt}

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
        