import unittest

import re
import spacy
import string2string
import string2string.alignment

import models.utilities.tags
import scoring.replacement

class ReplaceAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("nb_core_news_sm")
    
    def test_no_phi(self):
        source = self.nlp.make_doc("The quick brown fox jumps over the lazy dog")
        target = "The quick brown fox jumps over the lazy dog"
        results = scoring.replacement.align_answer(source, target)

        self.assertEqual(results['tp'], 0)
        self.assertEqual(results['tn'], 9)
        self.assertEqual(results['fp'], 0)
        self.assertEqual(results['fn'], 0)

    def test_all_phi(self):
        source = self.nlp.make_doc("The quick brown fox jumps over the lazy dog")
        ents = [source.char_span(0, 43, "ADJ")]
        source.set_ents(ents)
        target = "<ADJ>"
        results = scoring.replacement.align_answer(source, target)

        self.assertEqual(results['tp'], 9)
        self.assertEqual(results['tn'], 0)
        self.assertEqual(results['fp'], 0)
        self.assertEqual(results['fn'], 0)
    
    def test_substitution(self):
        source = self.nlp.make_doc("The quick brown fox jumps over the lazy dog")
        ents = [source.char_span(4, 9, "ADJ"), source.char_span(35, 39, "ADJ")] # "quick", "lazy"
        source.set_ents(ents)
        target = "The <ADJ> brown fox jumps over the <ADJ> dog"

        results = scoring.replacement.align_answer(source, target)

        self.assertEqual(results['tp'], 2)
        self.assertEqual(results['tn'], 7)
        self.assertEqual(results['fp'], 0)
        self.assertEqual(results['fn'], 0)
    
    def test_removal(self):
        source = self.nlp.make_doc("The quick brown fox jumps over the lazy dog")
        target = "quick brown fox jumps over the lazy"

        results = scoring.replacement.align_answer(source, target)

        self.assertEqual(results['tp'], 0)
        self.assertEqual(results['tn'], 7)
        self.assertEqual(results['fp'], 2)
        self.assertEqual(results['fn'], 0)
        self.assertEqual(results['removals'], 2)
    
    def test_rewrites(self):
        source = self.nlp.make_doc("The quick brown fox jumps over the lazy dog")
        target = "Teh quick brown fox jumps over the lazy dgo"

        results = scoring.replacement.align_answer(source, target)

        self.assertEqual(results['tp'], 0)
        self.assertEqual(results['tn'], 7)
        self.assertEqual(results['fp'], 0)
        self.assertEqual(results['fn'], 2)
        self.assertEqual(results['rewrites'], 2)
    
    def test_insertions(self):
        source = self.nlp.make_doc("The quick brown fox jumps over the lazy dog")
        target = "The quick brown fox jumps over the lazy dog extra"

        results = scoring.replacement.align_answer(source, target)

        self.assertEqual(results['tp'], 0)
        self.assertEqual(results['tn'], 9)
        self.assertEqual(results['fp'], 0)
        self.assertEqual(results['fn'], 0)
        self.assertEqual(results['insertions'], 1)