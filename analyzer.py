import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
from collections import Counter
from utils import read_json, valid_json
from filter import filter_stopwords
from model import preprocess, build_word_vector, model, N_DIM
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


################ TOOLS TO ANALYZE USER'S DATASET ################


# Les fonctions ici vont permettre d'analyser les tweets récupéré via MINER.PY :

# 1) On commence par passer le dataset dans valid_json(), on va récupérer un fichier json valide
# 2) On passe ce nouveau fichier json dans read_json avant de pouvoir récupérer son contenu
# 3) On va tokenizer les données obtenues avec tokenize_tweets()
# 4) On passe ces tokens dans dataset_prediction()
# 5) On obtient grâce à calculate_result() les pourcentages de tweets positifs et négatifs.


# ------------------------------------------


# /!\ On doit passer le fichier obtenu avec miner.py  dans valid_json()

#valid_json('tweets.json', 'valid_tweets.json')


# On charge le nouveau fichier json avec read_json()

data = read_json('final_data.json')


# Transforme nos tweets en tokens (nettoyés avec un combo preprocess() + filter_stopwords())

def tokenize_tweets(data):
    l = len(data)
    list = []
    for i in range(l):
        tweet = data[i]['text']
        list.append(filter_stopwords(preprocess(tweet)))
    return list


tokens = tokenize_tweets(data)

print("Tokens in corpus: ", len(tokens))

# ------------------------------------------

# Si on veut connaître les N mots les plus communs dans notre corpus

def most_common_words(tokens, n):
    tf = Counter()
    for t in range(len(tokens)):
        tf.update(tokens[t])
    print(f"{n} most common words in corpus:")
    for tag, count in tf.most_common(n):
        print(f"{tag}: \t{count}")


#most_common_words(tokens, 20)

# ------------------------------------------

# Traduction des prédictions en pourcentage :

def calculate_result(result):
    neg = 0
    pos = 0
    for i, j in enumerate(result):
        if result[i].item() == 1:
            pos += 1
        else:
            neg += 1
    all_res = len(result)
    pos_tweets = (pos / all_res) * 100
    neg_tweets = (neg / all_res) * 100
    print("PREDICTION PERCENTAGES:")
    print("Positive: %.2f%%" % pos_tweets)
    print("Negative: %.2f%%" % neg_tweets)


# Prédictions sur une liste de tokens, on va se servir des tokens qu'on a récupéré dans notre corpus json avec tokenize_tweets()

def dataset_prediction(tokens):
    tokens = np.array(tokens, dtype=object)
    print("Analyzing sentiments...")
    query_vec = np.concatenate([build_word_vector(t, N_DIM) for t in tqdm(map(lambda x: x, tokens))])
    result = model.predict_classes(query_vec)
    calculate_result(result)


# Time to predict, on va calculer les taux de sentiments positifs et négatifs se trouvant dans notre corpus de tweets:

dataset_prediction(tokens)


# ------------------------------------------
# EXTRA FUNCTION :
# Prédiction d'une chaîne

def predict_this(str):
    query_tokens = preprocess(str)
    query_vec = build_word_vector(query_tokens, N_DIM)
    result = model.predict_classes(query_vec).item()
    if result == 1:
        print("POSITIVE: {%s}" % str)
    else:
        print("NEGATIVE: {%s}" % str)

