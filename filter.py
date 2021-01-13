import string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


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

# collection_words sont les mots que l'on utilise pour dataminer nos tweets, ces mots de collection vont alors se retrouver dans tous ou la majorité des tweets. On va alors les enlever parce qu'ils n'apportent rien d'intéressant pour notre analyse :

collection_words = ['#cyberpunk2077', 'cyberpunk', '2077', '#cyberpunk','@cyberpunkgame']


# string.punctuation n'enlève pas toutes les ponctuations trouvé dans le dataset, on va alors les rajouter dans les extra :

extra_punct = ['…','...','’','..','️']


# Les stopwords de ntlk pour la langue anglaise. Les stopwords peuvent avoir un effet négatif ou positif suivant notre modèle et l'objectif recherché, je les rajoute donc ici en option, si on souhaite les utiliser on ajoutera simplement 'stopwords' à la 'stoplist'

stopwords = stopwords.words('english')


# La stoplist va enfin être composée de plusieurs éléments indésirables :

stoplist =  list(string.punctuation) + extra_punct + collection_words

# ------------------------------------------

# Les stopwords vont être filtré de notre liste de tokens (on en profite pour virer les chiffres) :

def filter_stopwords(tokens):
    return [t for t in tokens if not t in stoplist and not t.isdigit()]
