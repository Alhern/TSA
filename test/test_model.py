import model
import numpy as np
import unittest
from unittest.mock import patch
import pandas


filename = 'test_sentiment.csv'
test_df = pandas.read_csv(filename, encoding="latin1", error_bad_lines=False)


class TestModel(unittest.TestCase):

    def test_preprocess(self):
        test_case = "@grimes needs to release more songs in 2021"
        expected_result = ['need', 'to', 'release', 'more', 'song', 'in']
        self.assertEqual(model.preprocess(test_case), expected_result)

    def test_postprocess(self):
        test_data = model.postprocess(test_df)
        expected_result = ['is', 'upset', 'that', 'he', "can't", 'update', 'his', 'facebook', 'by', 'texting', 'it', '...', 'and', 'might', 'cry', 'a', 'a', 'result', 'school', 'today', 'also', '.', 'blah', '!']
        self.assertEqual(test_data.tokens[0], expected_result)

    @patch('model.MIN_COUNT', 1)
    def test_w2vmodel_builder(self):
        test_data = np.array(model.postprocess(test_df).tokens)
        w2vmodel = model.w2vmodel_builder(test_data)
        similar = w2vmodel.wv.most_similar('long')
        self.assertTrue(similar)


if __name__ == "__main__":
    unittest.main()
