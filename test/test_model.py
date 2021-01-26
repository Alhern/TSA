import model
import numpy as np
import unittest
from unittest.mock import patch
import tempfile
import os
import pandas
import io
import sys


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
        with patch('sys.stdout', new=io.StringIO()):
            test_data = np.array(model.postprocess(test_df).tokens)
            w2vmodel = model.w2vmodel_builder(test_data)
            similar = w2vmodel.wv.most_similar('long')
            self.assertTrue(similar)
            sys.stdout = sys.__stdout__

    @patch('model.MIN_COUNT', 1)
    def test_save_load_w2vmodel(self):
        test_tmp_model = os.path.join(tempfile.gettempdir(), 'test_word2vec_model.tmp')

        with patch('sys.stdout', new=io.StringIO()):
            test_data = np.array(model.postprocess(test_df).tokens)
            w2vmodel = model.w2vmodel_builder(test_data)
            test_vocab_len = len(w2vmodel.wv.vocab)
            model.save_w2vmodel(w2vmodel, test_tmp_model)

            loaded_model = model.load_w2vmodel(test_tmp_model)
            expected_vocab_len = len(loaded_model.wv.vocab)

            self.assertEqual(test_vocab_len, expected_vocab_len)
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    unittest.main()
