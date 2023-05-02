import unittest

import eval

class EvaluationTests(unittest.TestCase):
    def test_norsynth_annotations(self):
        sample = "<First_Name>Frank</First_Name> er innlagt ved <Location>Testsykehuset</Location>"
        annotations = eval.list_annotations(sample)
        self.assertEqual(len(annotations), 2, "we expect to find two annotations")
        
        self.assertEqual(annotations[0], (0, 5, "First_Name"), "the annotation position should ignore the tags")
        self.assertEqual(annotations[1], (22, 35, "Location"))