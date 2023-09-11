import spacy
from typing import List
import string2string.alignment
import collections

class Scorer:
    def __init__(self, nlp: spacy.Language):
        self.nlp = nlp
        return

    def score(self, doc_bin: spacy.tokens.DocBin, answers: List[str]) -> dict:
        total_doc_length = 0
        ctr = collections.Counter()
        for i, doc in enumerate(doc_bin.get_docs(self.nlp.vocab)):
            answer = answers[i]
            results = align_answer(doc, answer)
            total_doc_length += len(doc)
            ctr.update(results)
        rates = {k: v / total_doc_length for (k, v) in ctr.items()}
        return rates

def align_answer(source, response, gap_char='~'):
    nw = string2string.alignment.NeedlemanWunsch(gap_char=gap_char)
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