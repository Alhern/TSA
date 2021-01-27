import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
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

# MAIN FUNCTIONS:
# 1- tokenize_tweets(data)
# 2- calculate_result(result)
# 3- dataset_prediction(tokens)

# EXTRA FUNCTIONS:
# 1- predict_this(str)
# 1- most_common_words(tokens, n)


################# MAIN FUNCTIONS ################

# Transforme nos tweets en tokens (nettoyés avec un combo preprocess() + filter_stopwords())

def tokenize_tweets(data):
    data_length = len(data)
    tokens = []
    for i in range(data_length):
        tweet = data[i]['text']
        tokens.append(filter_stopwords(preprocess(tweet)))
    return tokens


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


# Prédictions sur une liste de tokens,
# on va se servir des tokens qu'on a récupéré dans notre corpus json avec tokenize_tweets()

def dataset_prediction(tokens):
    tokens = np.array(tokens, dtype=object)
    print("Analyzing sentiments...")
    query_vec = np.concatenate([build_word_vector(t, N_DIM) for t in tqdm(map(lambda x: x, tokens))])
    result = model.predict_classes(query_vec)
    calculate_result(result)



################# EXTRA FUNCTIONS ################

# Prédiction d'une chaîne (voir tests.py pour des tests) :

def predict_this(str):
    query_tokens = preprocess(str)
    query_vec = build_word_vector(query_tokens, N_DIM)
    result = model.predict_classes(query_vec).item()
    if result == 1:
        print("POSITIVE: {%s}" % str)
    else:
        print("NEGATIVE: {%s}" % str)


# predict_this("I'm tired because my computer is so slow and old, ugh")


# Si on veut connaître les N mots les plus communs dans notre corpus
# (Penser à ajouter les stopwords dans la stoplist de filter.py si on ne veut pas les voir)

def most_common_words(tokens, n):
    tf = Counter()
    for t in range(len(tokens)):
        tf.update(tokens[t])
    print(f"{n} most common words in corpus:")
    for tag, count in tf.most_common(n):
        print(f"{tag}: \t{count}")


# most_common_words(tokens, 20)


###################################################
#                                                 #
#                START THE ENGINE!                #
#                                                 #
###################################################


def main():

    # /!\ On doit passer le fichier obtenu avec miner.py  dans valid_json()
    # valid_json('data/raw_datasets/tweets_cp.json', 'data/valid_datasets/valid_tweets_cp.json')

    # On charge le nouveau fichier json avec read_json()
    data = read_json('data/valid_datasets/valid_trump_tweets.json')
    tokens = tokenize_tweets(data)

    # print("Tokens in corpus: ", len(tokens))

    # Time to predict, on va calculer les taux de sentiments positifs et négatifs se trouvant dans notre corpus de tweets:
    dataset_prediction(tokens)


if __name__ == "__main__":
    main()