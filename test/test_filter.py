import filter
import unittest
from unittest.mock import patch


@patch('filter.stoplist', filter.create_stoplist(punctuation=True, extra_punctuation=False, collection_w=False, stopword_list=False))

class TestFilter(unittest.TestCase):

    def test_create_stoplist(self):
        expected_result = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
        given_result = filter.create_stoplist(punctuation=True, extra_punctuation=False, collection_w=False, stopword_list=False)
        self.assertEqual(given_result, expected_result)

    def test_filter_stopwords(self):
        test_case = ['coucou', '!', 'un', ',', 'test', '.']
        expected_result = ['coucou', 'un', 'test']
        self.assertEqual(filter.filter_stopwords(test_case), expected_result)


if __name__ == "__main__":
    unittest.main()
