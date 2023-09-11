import spacy
from typing import List
import string2string.alignment

class Scorer:
    def __init__(self, nlp: spacy.Language):
        self.nlp = nlp
        return

    def score(self, doc_bin: spacy.tokens.DocBin, answers: List[str]) -> dict:
        for i, doc in enumerate(doc_bin.get_docs(self.nlp.vocab)):
            answer = answers[i]
            return align_answer(doc, answers[i])

def align_answer(source, response, gap_char='-'):
    nw = string2string.alignment.NeedlemanWunsch()
    source_split, response_split = str(source).split(), response.split() 
    source_aligned, response_aligned = nw.get_alignment(source_split, response_split)
    source_aligned_elems = [p.strip() for p in source_aligned.split(' | ')]
    response_aligned_elems = [p.strip() for p in response_aligned.split(' | ')]

    tp, tn, fp, fn = 0, 0, 0, 0
    insertions, removals, rewrites = 0, 0, 0
    _gap_chars = 0
    
    for i, src_token in enumerate(source_aligned_elems):
        resp_token = response_aligned_elems[i]
        if src_token == gap_char:
            # If we find a gap character in the source, there is a token in the response
            # which is either a spurious token introduced by the model, or a PHI marker
            insertions += 1
            _gap_chars += 1
            continue
        
        doc_token = source[i - _gap_chars]
        if doc_token.ent_type_ == "":
            if resp_token == gap_char:
                # This word was unnecessarily removed,
                # count it as a false positive
                removals += 1
                fp += 1
            elif resp_token.startswith('<') and resp_token.endswith('>'):
                # Non-PHI replaced with a PHI marker,
                # false positive
                fp += 1
            elif resp_token != src_token:
                # We count replacing the word with anything
                # as a false negative 
                rewrites += 1
                fn += 1
            else:
                # Non-PHI kept, true negative
                tn += 1
        else:
            if resp_token == gap_char or (resp_token.startswith('<') and resp_token.endswith('>')):
                # PHI removed or replaced with PHI marker,
                # true positive
                tp += 1
            elif resp_token == src_token:
                # PHI not replaced,
                # false negative
                fn += 1
            else:
                # We count replacing the text with anything other than a PHI marker
                # as a false negative
                rewrites += 1
                fn += 1
    
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, 
            "removals": removals, "rewrites": rewrites, "insertions": insertions}