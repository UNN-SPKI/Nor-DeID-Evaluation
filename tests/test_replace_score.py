import unittest

import re
import spacy
import string2string
import string2string.alignment

import models.utilities.tags
import scoring.replacement

import spacy.tokens

class ReplaceScoreTests(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("nb_core_news_sm")
        self.scorer = scoring.replacement.Scorer(self.nlp)
    
    def test_perfect_phi(self):
        source = self.nlp.make_doc("The quick brown fox jumps over the lazy dog")
        ents = [source.char_span(4, 9, "ADJ"), source.char_span(35, 39, "ADJ")] # "quick", "lazy"
        source.set_ents(ents)
        docbin = spacy.tokens.DocBin()
        docbin.add(source)
        answers = ["The <ADJ> brown fox jumps over the <ADJ> dog"]

        results = self.scorer.score(docbin, answers)
        
        self.assertAlmostEqual(results['precision'], 1.0)
        self.assertAlmostEqual(results['recall'], 1.0)
        self.assertAlmostEqual(results['f1'], 1.0)