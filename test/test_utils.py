import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import utils
import tempfile
import os
import sys
import io
import unittest
from unittest.mock import patch


class TestUtils(unittest.TestCase):

    def test_valid_read_json(self):
        with patch('sys.stdout', new=io.StringIO()):
            test_tmp_valid_tweets = os.path.join(tempfile.gettempdir(), 'test_valid_tweets.tmp')
            utils.valid_json("test_raw_tweets.json", test_tmp_valid_tweets)

            expected_result = {'text': 'Finally got a #Cyberpunk2077 crash with @GoogleStadia . First after 35 hours. Still not good to see a game crash, b… https://t.co/dhf0ewKxVX'}

            test_result = utils.read_json(test_tmp_valid_tweets)

            self.assertEqual(test_result[0], expected_result)
            sys.stdout = sys.__stdout__

    def test_load_modeljson(self):
        test_model = utils.load_modeljson("test_model_config.json", "test_model_weights.h5")
        test_layers_len = len(test_model.layers)
        self.assertEqual(test_layers_len, 3)
