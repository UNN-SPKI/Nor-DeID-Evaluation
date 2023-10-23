import unittest

import spacy
import models.spacy

class BaselineMatcherTests(unittest.TestCase):
    def setUp(self):
        self.model = models.spacy.SpacyModel()
        self.nlp = self.model.language

    def test_match_age(self):
        source = self.nlp.make_doc("Han er 29 år gammel")
        docbin = spacy.tokens.DocBin()
        docbin.add(source)

        answers = self.model.predict(docbin, self.nlp, 'annotate')
        prediction = answers[0].predicted

        self.assertEqual(len(prediction.ents), 1)
        self.assertEqual(prediction.ents[0].text, "29")
        self.assertEqual(prediction.ents[0].label_, "Age")
        
    def test_match_ssn(self):
        source = self.nlp.make_doc("Hans personnummer er 000000 00000")
        docbin = spacy.tokens.DocBin()
        docbin.add(source)

        answers = self.model.predict(docbin, self.nlp, 'annotate')
        prediction = answers[0].predicted

        self.assertEqual(len(prediction.ents), 1)
        self.assertEqual(prediction.ents[0].text, "000000 00000")
        self.assertEqual(prediction.ents[0].label_, "Social_Security_Number")
    
    def test_match_phone(self):
        source = self.nlp.make_doc("Hans telefonnummer er 77712345")
        docbin = spacy.tokens.DocBin()
        docbin.add(source)

        answers = self.model.predict(docbin, self.nlp, 'annotate')
        prediction = answers[0].predicted

        self.assertEqual(len(prediction.ents), 1)
        self.assertEqual(prediction.ents[0].text, "77712345")
        self.assertEqual(prediction.ents[0].label_, "Phone_Number")
    
    def test_match_date(self):
        source = self.nlp.make_doc("Jeg er født 1 . januar 1985")
        docbin = spacy.tokens.DocBin()
        docbin.add(source)

        answers = self.model.predict(docbin, self.nlp, 'annotate')
        prediction = answers[0].predicted
        
        self.assertEqual(len(prediction.ents), 1)
        self.assertEqual(prediction.ents[0].text, "1 . januar 1985")
        self.assertEqual(prediction.ents[0].label_, "Date")

    def test_replace(self):
        source = self.nlp.make_doc("Hans telefonnummer er 77712345 nå")
        docbin = spacy.tokens.DocBin()
        docbin.add(source)

        answers = self.model.predict(docbin, self.nlp, 'replace')
        
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0], "Hans telefonnummer er <Phone_Number> nå")
    
    def test_firstname_lastname(self):
        source = self.nlp.make_doc("Han heter Ola Olsen")
        docbin = spacy.tokens.DocBin()
        docbin.add(source)

        answers = self.model.predict(docbin, self.nlp, 'replace')
        
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0], "Han heter <First_Name> <Last_Name> ")

