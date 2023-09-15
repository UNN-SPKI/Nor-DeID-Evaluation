import logging
from typing import Dict, List, Union
import re
import spacy

class SpacyModel:
    """SpacyModel implements a default spaCy EntityRecognizer and EntityRuler, allowing
    custom class."""
    def __init__(self):
        # We don't want to change the spaCy pipeline passed to us, 
        # so just make a new one:
        self.language = spacy.load('nb_core_news_lg')
        self.patterns = [
            (r"\d{11}", "Social_Security_Number"),
            (r"\d{6}[\s-]+\d{5}", "Social_Security_Number"),
            (r"\+\d{10}", "Phone_Number"),
            (r"00\d{10}", "Phone_Number"),
            (r"\d{8}", "Phone_Number"),
            (r"(\+\d{2})?\d{2}\s\d{2}\s\d{2}\s\d{2}", "Phone_Number"),
            (r"\d{1,2}\.\d{1,2}\.\d{2,4}", "Date"),
            (r"\d{1,2} \. \d{1,2} \. \d{2,4}", "Date"),
            (r"((19)|(20))\d{2}", "Date"),
            (r"\d{1,2}\s?\.\s?((januar)|(februar)|(mars)|(april)|(mai)|(juni)|(juli)|(august)|(september)|(oktober)|(november)|(desember))\s?\d{2,4}?", "Date")
        ]
    
    def predict(self, doc_bin: spacy.tokens.DocBin, language: spacy.Language, mode: str) -> Union[List[spacy.training.Example], List[str]]:
        examples = []
        for ref in doc_bin.get_docs(language.vocab):
            new_doc = ref.text
            results = self.language(new_doc)

            map_categories = {
                'PER': 'Ignore', # We handle first and last names separately
                'ORG': 'Location',
                'GPE': 'Location',
                'GPE_LOC': 'Location',
                'GPE_ORG': 'Location',
                'LOC': 'Location',
                'EVT': 'Ignore',
                'MISC': 'Ignore',
                'DRV': 'Ignore',
                'PROD': 'Ignore'
            }
                
            
            mapped_label = lambda l: map_categories[l] if l in map_categories else l
            spans = [spacy.tokens.span.Span(results, s.start, s.end, mapped_label(s.label_)) for s in results.ents if mapped_label(s.label_) != 'Ignore']

            for token in results:
                if token.ent_type_ == 'PER' and token.ent_iob_ == 'B':
                    spans.append(spacy.tokens.span.Span(results, token.i, token.i+1, 'First_Name'))
                elif token.ent_type_ == 'PER' and token.ent_iob_ == 'I':
                    spans.append(spacy.tokens.span.Span(results, token.i, token.i+1, 'Last_Name'))

            age_pattern = r"(\d+) år"
            age_matches = re.finditer(age_pattern, new_doc, re.IGNORECASE)
            for m in age_matches:
                span = results.char_span(m.start(), m.end() - 3, "Age", alignment_mode="expand")
                spans.append(span)

            for (pattern, label) in self.patterns:
                matches = list(re.finditer(pattern, ref.text, re.IGNORECASE))
                for match in matches:
                    span = results.char_span(match.start(), match.end(), label, alignment_mode="expand")
                    if span == None:
                        logging.warning(f"Matched {match} but could not create span")
                    else:
                        spans.append(span)
            
            fixed_spans = spacy.util.filter_spans(spans)
            results.set_ents(fixed_spans)

            if mode == 'annotate':
                example = spacy.training.Example(predicted=results, reference=ref)
            else:
                example = ""
                for tok in results:
                    if tok.ent_type_ == "":
                        example += tok.text_with_ws
                    else:
                        example += f"<{tok.ent_type_}> "
            examples.append(example)
        return examples