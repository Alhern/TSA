from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


###########################################
##########   TF-IDF UTILITIES    ##########
###########################################

# FUNCTIONS:
# 1- tfidf_builder(corpus)
# 2- save_tfidf(tfidf)
# 3- load_tfidf(filename)

# ------------------------------------------

# TF-IDF - Term Frequency–Inverse Document Frequency (variante avancée de BoW - Bag of Words)
# Il va s'agir d'évaluer la pertinence (le poids) d'un terme suivant sa fréquence dans le corpus (Term Frequency) et le nombre de documents contenant ce terme (Inverse Document Frequency) au sein de ce même corpus. Plus un mot est fréquent dans le corpus, plus son poids sera léger. Il en va de même si le mot est très rare.

# J'obtiens une amélioration très nette de la performance de mon modèle en utilisant TF-IDF lors de la construction des vecteurs de mot (build_word_vector()).


# Construction du vecteur TF-IDF à partir du corpus de tweets

def tfidf_builder(corpus):
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    vectorizer.fit_transform([x.words for x in corpus])
    return dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))



# Sauvegarde du vecteur TF-IDF sur l'ordi via Pickle

def save_tfidf(tfidf):
    print("Pickling the TF-IDF vector...")
    with open("tfidf.pickle", "wb") as f:
        pickle.dump(tfidf, f)



# Chargement du vecteur TF-IDF à partir de l'ordi

def load_tfidf(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
