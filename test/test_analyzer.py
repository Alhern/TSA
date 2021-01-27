import analyzer
import utils
import tempfile
import sys
import os
import io
import unittest
from unittest.mock import patch

class TestAnalyzer(unittest.TestCase):

    def test_tokenize_tweets(self):
        with patch('sys.stdout', new=io.StringIO()):
            test_tmp_valid_tweets = os.path.join(tempfile.gettempdir(), 'test_valid_tweets.tmp')
            utils.valid_json("test_raw_tweets.json", test_tmp_valid_tweets)
            test_data = utils.read_json(test_tmp_valid_tweets)
            expected_result = ['finally', 'got', 'a', '#cyberpunk2077', 'crash', 'with', 'first', 'after', 'hour', 'still', 'not', 'good', 'to', 'see', 'a', 'game', 'crash', 'b']
            test_tokens = analyzer.tokenize_tweets(test_data)
            self.assertEqual(test_tokens[0], expected_result)
            sys.stdout = sys.__stdout__



if __name__ == '__main__':
    unittest.main()