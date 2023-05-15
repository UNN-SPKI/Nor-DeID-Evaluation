from typing import List
import spacy

class DummyModel:
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language) -> List[spacy.training.Example]:
        examples = []
        for doc in doc_bin.get_docs(language.vocab):
            annotations = {'entities': []}
            example = spacy.training.Example.from_dict(doc, annotations)
            examples.append(example)
        return examples