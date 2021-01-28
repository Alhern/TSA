from model import load_w2vmodel
from analyzer import tokenize_tweets
from utils import read_json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from nltk import bigrams
import networkx as nx
import itertools
import pandas as pd


###################################################
#                                                 #
#               DATA VISUALIZATION                #
#                                                 #
###################################################

# On va ici pouvoir visualiser nos données (Sentiment140 + nos tweets) de 4 manières différentes :

# 1) word_embedding_space_plot(w2vmodel) (SENTIMENT140 ONLY)
# 2) zipf_plot()
# 3) term_distribution_plot()
# 4) bigram_network_plot

### REQUIRED ###
# For word embedding space:
w2v_model = load_w2vmodel("pretrained/my_w2vmodel")

# For Zipf's law:
data = read_json('data/valid_datasets/valid_trump_tweets.json')
tokens = tokenize_tweets(data)
tf = Counter()
for t in range(len(tokens)):
    tf.update(tokens[t])

# For bigram network:
terms = [list(bigrams(tweet)) for tweet in tokens]


# -------------------------------------------------------

###### VISUALISATION DE L'ESPACE DE REPRÉSENTATION DES MOTS ######

def word_embedding_space_plot(w2vmodel):

    w2v_model = w2vmodel

    # On récupère les vecteurs du modèle W2V
    vectors = w2v_model[w2v_model.wv.vocab]

    # On va utiliser l'analyse en composantes principales, la classe PCA va prendre 2 dimensions
    pca = PCA(n_components=2)

    # On entraîne le modèle
    result = pca.fit_transform(vectors)

    # C'est parti on fait le graph
    plt.figure(figsize=(18,18))
    plt.scatter(x=result[:500, 0], y=result[:500, 1])
    plt.xlabel("PC1",size=15)
    plt.ylabel("PC2",size=15)
    plt.title("Word Embedding Space (with 500 words)",size=20)

    # On va limiter à 500 mots pour des raisons de visibilité
    words = list(w2v_model.wv.vocab)[:500]

    for i, word in enumerate(words):
        plt.annotate(word, size=8, xy=(result[i, 0], result[i, 1]))

    plt.show()


# -------------------------------------------------------

###### VISUALISATION DE LA LOI DE ZIPF ######

def zipf_plot():

    y = [count for tag, count in tf.most_common(200)]
    x = range(1, len(y) + 1)

    plt.figure(figsize=(10,8))
    plt.bar(x, y, color='deepskyblue')
    plt.plot(x, y)
    plt.title("Fréquences des termes")
    plt.ylabel("Fréquence")
    plt.xlabel("Termes")
    plt.margins(0, 0)
    plt.tight_layout()

    plt.show()


# -------------------------------------------------------

###### VISUALISATION DE LA DISTRIBUTION DES TERMES ######

def term_distribution_plot():

    y = [count for tag, count in tf.most_common(30)]
    x = [tag for tag, count in tf.most_common(30)]
    
    plt.bar(x, y, color='deepskyblue')
    plt.plot(x, y, color='r', linestyle='-',linewidth=1)
    plt.title("Fréquences des termes (30)")
    plt.ylabel("Fréquence")
    plt.xlabel("Termes")

    plt.xticks(rotation=90)

    for i, (tag, count) in enumerate(tf.most_common(30)):
        plt.text(i, count, f' {count} ', size=6, rotation=90, ha='center', va='top', color='black')
    plt.xlim(-0.6, len(x) - 0.4)
    plt.tight_layout()

    plt.show()


# -------------------------------------------------------

###### VISUALISATION DU RÉSEAU DES BIGRAMMES ######

def bigram_network_plot(terms):

    # Applatissement de la liste des bigrammes
    bigrams = list(itertools.chain(*terms))

    counter = Counter(bigrams)
    bigram_df = pd.DataFrame(counter.most_common(40),
                             columns=['bigram', 'count'])

    # Dico des bigrammes + leur nombre
    dico = bigram_df.set_index('bigram').T.to_dict('records')

    # Initialisation du graph
    graph = nx.Graph()

    # Création des connections entre les nodes
    for key, val in dico[0].items():
        graph.add_edge(key[0], key[1], weight=(val*0.2))

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(graph, k=1)

    # Paramétrage du réseau
    nx.draw_networkx(graph, pos,
                 font_size=8,
                 width=1,
                 edge_color='black',
                 node_color='crimson',
                 with_labels=False,
                 ax=ax)

    # Étiquettes offset
    for key, value in pos.items():
        x, y = value[0] + .01, value[1] + .06
        ax.text(x, y,
            s=key,
            bbox=dict(facecolor='lavender', alpha=0.25),
            horizontalalignment='center', fontsize=10)

    plt.show()


###########################
#### CHOOSE YOUR PLOT! ####
###########################

#word_embedding_space_plot(w2v_model)
#zipf_plot()
#term_distribution_plot()
#bigram_network_plot(terms)
