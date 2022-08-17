import string
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import emoji

###########################################
##########    STOPWORDS FILTER   ##########
###########################################

# GLOBAL VARIABLES:
# 1- collection_words
# 2- extra_punct
# 3- stopwords
# 4- stoplist

# FUNCTION:
# 1- filter_stopwords(tokens)

# ------------------------------------------

# collection_words sont les mots que l'on utilise pour dataminer nos tweets, ces mots de collection
# vont alors se retrouver dans tous ou la majorité des tweets.
# On va donc les enlever parce qu'ils n'apportent rien d'intéressant pour notre analyse,
# exemples :

collection_words_cp = ['#cyberpunk2077', 'cyberpunk', '2077', '#cyberpunk', '@cyberpunkgame']
collection_words_trump = ['impeached', 'impeachment']

# Mettez les collection words relatifs à votre query dans la liste suivante :

collection_words = []

# string.punctuation n'enlève pas toutes les ponctuations trouvées dans le dataset,
# on va alors les rajouter dans les extra :

extra_punct = ['…', '...', '’', '..', '️']

# Les stopwords de ntlk pour la langue anglaise.
# Les stopwords peuvent avoir un effet négatif ou positif suivant notre modèle et l'objectif recherché,
# je les rajoute donc ici en option, si on souhaite les utiliser on ajoutera simplement 'stopwords' à la 'stoplist'

stopwords = stopwords.words('english')

# Les emojis sont présents dans les tweets que l'on collecte,
# ils n'ont pas d'intérêt puisque Sentiment140 n'utilise pas d'emoji, on va donc les enlever :

emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)


# La stoplist va enfin être composée de plusieurs éléments indésirables, au choix :

def create_stoplist(punctuation=True, extra_punctuation=True, collection_w=True, stopword_list=False, emoji_list=True):
    stoplist = []
    if punctuation:
        stoplist += string.punctuation
    if extra_punctuation:
        stoplist += extra_punct
    if collection_w:
        stoplist += collection_words
    if stopword_list:
        stoplist += stopwords
    if emoji_list:
        stoplist += emojis
    return stoplist


# On l'initialise avec les éléments que l'on souhaite filtrer ou pas :

stoplist = create_stoplist(
    punctuation=True,
    extra_punctuation=True,
    collection_w=True,
    stopword_list=False,
    emoji_list=True)


# ------------------------------------------

# Les stopwords vont être filtrés de notre liste de tokens :

def filter_stopwords(tokens):
    return [t for t in tokens if not t in stoplist]
