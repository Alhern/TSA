# TSA
Twitter Sentiment Analyzer using Gensim's Word2Vec + NLTK + Keras.

## Requirements
### Miner
⚠️To be able to use `miner.py` you must have a Twitter developer account. 

Your authentication details will need to be placed directly inside `miner.py` OR in a file named `creds.py` that will look like this:

    consumer_key = 'abcd'
    consumer_secret = 'efgh'
    access_token = '4567'
    access_token_secret = '1234'

### Analyzer
To use TSA you will need to open your virtual env and install the required files in `requirements.txt` with pip3:

    pip3 install -r requirements.txt

If you use Anaconda for example, you can follow these steps:

    conda create -n TSA python=3.6

    conda activate TSA

    pip3 install -r requirements.txt

## Usage
You can start the program by running `./analyzer.py`

You will then be greeted with a menu:

     
     ████████ ███████  █████  
        ██    ██      ██   ██ 
        ██    ███████ ███████ 
        ██         ██ ██   ██ 
        ██    ███████ ██   ██                          
    
    ---Welcome to TSA---
    What would you like to do?

	    1) Mine tweets
	    2) Validate a json file
	    3) Analyze a corpus of tweets
	    4) Find the most common words in a corpus of tweets
	    5) Predict a string
	    6) Quit

    Enter your choice (1-6): 

You will be prompted to enter a number in the range 1-6.

* `1) Mine tweets` will prompt you to enter your queries, `miner.py` will then look for tweets containing these terms and they will be put in a file called `tweets.json`. You will be able to interrupt the program by pressing Ctrl+C.


* `2) Validate a json file` will prompt you to enter a JSON file name. The file will then be reformatted into a valid JSON file and you will then be able to use it with the analyzer.


* `3) Analyze a corpus of tweets` will prompt you to enter a JSON file name and will ask you if you also want to visualize extra data. The file will then analyze the sentiments in your corpus, you will see a prediction percentage for positive and negative tweets. If you agreed to get extra data, you will also  get 4 extra charts: one showing the word embedding space in Sentiment140, one for Zipf's law, one for the word frequency distribution and one for bigrams.


* `4) Find the most common words in a corpus of tweets` will prompt you to enter the number of the most common words you want to see and a JSON file name for your corpus. You will then see the N most common words in your corpus.


* `5) Predict a string` will prompt you to enter a string you'd like to predict the sentiment of.


* `6) Quit` will quit the program.

## Credits
This program has been built with:

* `Python 3.6`
* `tweepy`
* `gensim`
* `tensorflow`
* `nltk`
* `scikit-learn`

⚠️Python 3.6 is recommended to run this program.