from typing import List
import spacy

class BrathenModel:
    """BrathenModel implements spaCy's default NER model with class-specific entity rules."""
    def __init__(self):
        # We don't want to change the spaCy pipeline passed to us, 
        # so just make a new one:
        self.language = spacy.load('nb_core_news_lg')
    
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language) -> List[spacy.training.Example]:
        examples = []
        for ref in doc_bin.get_docs(language.vocab):
            new_doc = ref.text
            results = self.language(new_doc)

            map_categories = {
                'PER': 'Name',
                'ORG': 'Location',
                'GPE': 'Location',
                'GPE_LOC': 'Location',
                'GPE_ORG': 'Location',
                'LOC': 'Location',
                'EVT': 'Other',
                'MISC': 'Other',
                'DRV': 'Other'
            }
            mapped_label = lambda l: map_categories[l] if l in map_categories else l
            fixed_labels = [spacy.tokens.span.Span(results, s.start, s.end, mapped_label(s.label_)) for s in results.ents]
            results.set_ents(fixed_labels)

            example = spacy.training.Example(predicted=results, reference=ref)
            examples.append(example)
        return examples