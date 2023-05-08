import unittest

import eval

class EvaluationTests(unittest.TestCase):
    def test_norsynth_annotations(self):
        sample = "<First_Name>Frank</First_Name> er innlagt ved <Location>Testsykehuset</Location>"
        no_tags = "Frank er innlagt ved Testsykehuset"
        annotations = eval.list_annotations(sample)
        self.assertEqual(len(annotations), 2, "we expect to find two annotations")
        
        self.assertEqual(annotations[0], (0, 5, "First_Name"), "the annotation position should ignore the tags")
        self.assertEqual(annotations[1], (21, 34, "Location"))

    def test_norsynth_correct_offsets(self):
        """Verify that the offsets from list_annotations are correct."""
        sample = "<First_Name>Åge</First_Name> ved <Location>Testsykehuset</Location> kan nås på <Phone_Number>555 12345</Phone_Number>"
        
        annotations = eval.list_annotations(sample)
        cleaned = eval.remove_tags(sample)

        tag_0_start, tag_0_end, _ = annotations[0]
        self.assertEqual(cleaned[tag_0_start:tag_0_end], "Åge")

        tag_1_start, tag_1_end, _ = annotations[1]
        self.assertEqual(cleaned[tag_1_start:tag_1_end], "Testsykehuset")

        tag_2_start, tag_2_end, _ = annotations[2]
        self.assertEqual(cleaned[tag_2_start:tag_2_end], "555 12345")
