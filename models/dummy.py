from typing import List, Union
import spacy

class DummyModel:
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language, mode: str) -> Union[List[spacy.training.Example], List[str]]:
        examples = []
        for doc in doc_bin.get_docs(language.vocab):
            if mode == 'replace':
                examples.append(str(doc))
            else:
                annotations = {'entities': []}
                example = spacy.training.Example.from_dict(doc, annotations)
                examples.append(example)
        return examples