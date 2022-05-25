import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
pd.options.mode.chained_assignment = None
import numpy as np
from copy import deepcopy
import h5py
import re

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument

import nltk
from nltk.tokenize import TweetTokenizer
nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

from tfidf import tfidf_builder, save_tfidf, load_tfidf
from utils import save_modeljson, load_modeljson


###################################################
#                                                 #
#         IMPORTING THE TRAINING DATASET          #
#                                                 #
###################################################

# On va charger le dataset Sentiment140 se trouvant dans le dossier data
# /!\ Ce dataset de 239MB doit être téléchargé et placé dans le dossier data
# /!\ Son nom de fichier doit être training.1600000.processed.noemoticon.csv

# Sentiment140 peut être téléchargé soit via mon github, soit via un de ces liens :
# https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit
# http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

# Ce fichier csv contient 1,6M de tweets et 6 colonnes :
# 0- Polarité/Sentiment du tweet (0 = négatif, 4 = positif)
# 1- ID
# 2- Date
# 3- Query
# 4- Source/utilisateur
# 5- Le texte

# On va garder que les colonnes qui nous intéresse : polarité/sentiment et texte.
# Pour plus de clarté, le sentiment positif 4 = 1.


TRAINING_DATASET_PATH = "data/training.1600000.processed.noemoticon.csv"


def import_dataset():
    df = pd.read_csv(TRAINING_DATASET_PATH, encoding="latin1", error_bad_lines=False)
    df.columns = ['sentiment', 'id', 'time', 'query', 'source', 'text']
    df = df.drop(['id', 'time', 'query', 'source'], axis=1)
    df['sentiment'] = df['sentiment'].map({4: 1, 0: 0})
    return df


###################################################
#                                                 #
#               TOKENIZING TWEETS                 #
#                                                 #
###################################################

# On va transformer chaque tweet en tokens, on en profite pour virer ce qui nous intéresse pas
# en utilisant regex : les urls et les mentions

# Notez que je ne filtre pas les stopwords ici, j'ai remarqué que cela baissait les performances de mon modèle,
# ce qui est normal puisque si je prends les phrases "i am happy" et "i am not happy" et que je vire "not" qui
# fait partie des stopwords, les 2 phrases pourtant opposées prennent alors le même sens.

# Enfin notons que je garde aussi les hashtags parce qu'ils sont très souvent utilisés pour rajouter du sens
# et du contexte à un tweet, par exemple: "Donald did what? #idiot #demon #theworst" ou même "cats are the #greatest".

def preprocess(tweet):

    # Tokenize les tweets :
    tknzr = TweetTokenizer(preserve_case=False)
    tokens = tknzr.tokenize(tweet)

    # Vire les tokens qu'on ne veut pas garder :
    url = re.compile('https?://[A-Za-z0-9./]+')  # url
    mention = re.compile('@(\w+)')   # mentions
    tokens = [t for t in tokens if not url.search(t)]
    tokens = [t for t in tokens if not mention.search(t)]
    tokens = [t for t in tokens if not t.isdigit()]  # on vire les nombres

    # OPTION : Vire les lettres qui se répètent dans un mot, dans la limite de 2 lettres
    # (loveeee => lovee), on limite à 2 sinon on aura un souci avec les mots comme
    # "good" qui deviendrait "god" et prendrait un tout autre sens
    #tokens = [re.sub(r'(.)\1{2,}', r'\1\1', t) for t in tokens]

    # La lemmatization marche très bien avec Word2vec par rapport au stemming
    # Dans mon cas ça ne me permet pas d'améliorer mes performances de façon importante
    # mais ça me permet de réduire la taille de mon vocabulaire
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


# Transforme chaque tweet de notre corpus (ici 1,6M) en tokens avec la fonction précédente preprocess()

def postprocess(data, n=1600000):
    data = data.head(n).copy(deep=True)
    data['tokens'] = data['text'].progress_map(preprocess)
    return data


###################################################
#                                                 #
#             W2V MODEL CONSTRUCTION              #
#                                                 #
###################################################


# W2V MODEL CONFIG:

N_DIM = 300  # dimension du vecteur de mot
WINDOWS = 5  # distance maximale entre le mot cible et les mots autour du mot cible
SG = 0  # algorithme d'entrainement, soit CBOW (0) soit skip-gram (1)
MIN_COUNT = 10  # Mots qui apparaissent moins de MIN_COUNT fois seront ignorés


# Initialise le modèle Word2vec, crée son vocabulaire à partir du corpus et l'entraîne

def w2vmodel_builder(data):
    print("INITIALIZING THE W2V MODEL")
    w2v_model = Word2Vec(size=N_DIM, sg=SG, window=WINDOWS, min_count=MIN_COUNT)
    print("BUILDING THE VOCABULARY")
    w2v_model.build_vocab(sentences=data)
    print("TRAINING THE W2V MODEL")
    w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
    return w2v_model


###################################################
#                                                 #
#             SAVING/LOADING W2V MODEL            #
#                                                 #
###################################################

# Sauvegarde du modèle W2v sur l'ordinateur

def save_w2vmodel(model, filename):
    print("Saving the W2V model to disk...")
    model.save(filename)


# Charge le modèle W2v sur l'ordinateur

def load_w2vmodel(filename):
    print("Loading the W2V model from disk...")
    return gensim.models.Word2Vec.load(filename)


###################################################
#                                                 #
#            WORD VECTOR CONSTRUCTION             #
#                                                 #
###################################################

# Construction des vecteurs des mots à partir d'une liste de mots (les tokens des tweets)
# et la dimension du vecteur, on multiplie ensuite chaque terme du vecteur W2V avec
# son importance dans TFIDF. Ce dernier point me permet de bien améliorer la performance du modèle.

def build_word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v_model.wv[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# Construction des sets de training et testing pour le modèle à l'aide de build_word_vector(),
# on va utiliser x_train puis x_test comme argument.

def build_training_sets(x_set):
    set_vec = np.concatenate([build_word_vector(x, N_DIM) for x in x_set])
    set_vec = scale(set_vec)
    return set_vec


###################################################
#                                                 #
#       MODEL INITIALIZATION AND COMPILATION      #
#                                                 #
###################################################

# Définition et compilation du modèle

def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=300))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


###################################################
#                                                 #
#          MODEL TRAINING AND EVALUATION          #
#                                                 #
###################################################

# TRAINING CONFIG:

EPOCHS = 30    # nombre d'itérations/passages de toutes les données
BATCH_SIZE_TRAIN = 1024  # plus c'est élevé, plus l'entraînement va vite mais peut faire baisser la précision
VALIDATION_SPLIT = 0.1   # 10% des données training seront utilisés pour le test
VERBOSE_TRAIN = 2  # une ligne par epoch, plus lisible


# TESTING CONFIG:

BATCH_SIZE_TEST = 1024
VERBOSE_TEST = 2


def train_model(model, train_vec, y_train, test_vec, y_test):

    # Training:
    history = model.fit(train_vec, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE_TRAIN, validation_split=VALIDATION_SPLIT, verbose=VERBOSE_TRAIN)

    # Testing:
    score = model.evaluate(test_vec, y_test, batch_size=BATCH_SIZE_TEST, verbose=VERBOSE_TEST)
    print('Test loss: %.2f%%\nTest accuracy: %.2f%%' % (score[0], score[1]))

    return history, score


###################################################
#                                                 #
#                START THE ENGINE!                #
#                                                 #
###################################################

### Toutes nos fonctions sont prêtes, il est temps de démarrer la machine

#### IMPORTING & PROCESSING SENTIMENT140:

#data = import_dataset()
#data = postprocess(data)


#### SPLITTING OUR DATA:
# Sur 1M de données, 80% vont être utilisées pour l'entraînement, 20% pour le test de validation
# afin de pouvoir évaluer la performance du modèle. (1M parce qu'au delà mon OS suffoque et tue le processus)

#x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(1000000).tokens), np.array(data.head(1000000).sentiment), test_size=0.2, random_state=1)
#all_data = np.array(data.tokens)


#### BUILDING THE W2V MODEL & SAVING/LOADING IT TO/FROM DISK:

#w2v_model = w2vmodel_builder(all_data)

#save_w2vmodel(w2v_model, "my_w2vmodel8")
w2v_model = load_w2vmodel("pretrained/my_w2vmodel")

#print(w2v_model.most_similar("food"))


#### BUILDING THE TF-IDF MATRIX & SAVING/LOADING IT TO/FROM DISK:

#tfidf = tfidf_builder(all_data)

#save_tfidf(tfidf, "tf2")
tfidf = load_tfidf("pretrained/tfidf.pickle")

#print('TF-IDF vocabulary size:', len(tfidf))


#### BUILDING THE TRAINING & TESTING SETS:
#train_vec = build_training_sets(x_train)
#test_vec = build_training_sets(x_test)


#### BUILDING OUR SEQUENTIAL MODEL WITH KERAS
#model = build_model()


#### TRAINING THE MODEL & SAVING/LOADING IT TO/FROM DISK:
#train_model(model, train_vec, y_train, test_vec, y_test)

#save_modeljson(model)
model = load_modeljson("pretrained/model_config.json", "pretrained/model_weights.h5")
