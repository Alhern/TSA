#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from creds import *
from tweepy.streaming import StreamListener
import json
import tweepy
from time import sleep


################ AUTHENTICATING ################
# Utilisez ici vos propres tokens pour vous connecter à Tweepy.
# Tous mes tokens se trouvent dans mon fichier privé creds.py

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


################ QUERIES ################
# On met ici les mots clés qui nous intéressent, on va collecter tous les tweets en rapport avec ce(s) mot(s).
# J'ai choisi ici le jeu vidéo Cyberpunk 2077 car c'est un sujet qui m'intéresse et parce qu'il y a énormément de controverses autour de ce jeu. Sa popularité fait qu'il est facile de collecter des milliers de tweets le concernant en laissant tourner mon Stream.
QUERY = ["#Cyberpunk2077"]


################ THE MINER'S HEART ################
# Cette façon de "miner" des tweets est similaire à l'ouverture d'un robinet : je l'ouvre pour récupérer de l'eau (des données), je ferme quand j'ai assez d'eau.
# On capture ici les tweets en temps réel contrairement à la méthode tweepy.Cursor() qui elle va récupérer des tweets déjà postés dans une limite temporelle fixée par Twitter (on ne peut pas aller au delà de 30 jours).
# Contrairement à cette méthode, l'ouverture du Stream me permet de ne passer qu'un seul appel à l'API Tweepy. Ce qui permet d'éviter de se faire blacklister par Twitter en effectuant trop d'appels...
# Cette méthode de Stream est particulièrement intéressante lorsque l'on souhaite récupérer des tweets relatifs à des évènements en cours (ex : élections), suivant la dimension de cet évènement on peut alors récupérer des millions de tweets assez rapidement (ex : l'assaut du Capitole aux États-Unis), par contre si on s'intéresse à un sujet qui N'est pas particulièrement actif (ex : actualités de Bernard Pivot), alors la collecte de données sera bien lente.

# Ce stream va ouvrir un fichier json créé au préalable et va y attacher tous les tweets collectés.
# Il faut noter que je ne récupère pas l'entièreté d'un tweet (cf. l'anatomie d'un tweet), les informations concernant l'identité de l'utilisateur, sa localisation, l'heure etc ne m'intéressent pas ici. J'ai donc fait le choix de ne garder que ce qui m'intéresse pour mon projet : le corps du tweet, son texte.

# Cette façon de récupérer les tweets ici fait que je peux arrêter le Stream et le reprendre à tout moment. Par exemple lorsqu'un évènement fait que beaucoup de gens sont en train de tweeter dessus. Les nouveaux tweets seront alors attachés au document tweets.json.

class Listener(StreamListener):
    def on_data(self, data):
        try:
            with open('tweets.json', 'a', encoding="utf-8") as f:
                status = json.loads(data)
                if not status['retweeted'] and 'RT @' not in status['text']:
                    if "extended_tweet" in status:
                        text = {'text': status['extended_tweet']['full_text']} # sinon les tweets ne montreront que ~115 caractères
                    else:
                        text = {'text': status['text']}
                    f.write(json.dumps(text, indent = 4))
                return True
        except BaseException as e:
            #print("Error on_data: %s" % str(e))
            #sleep(6)
            return True

    def on_error(self, status):
        print(status)
        return True


################ START THE STREAM ################

def main():
    stream = None
    try:
        print("NOW MINING.... Ctrl+C to interrupt the stream.")
        stream = tweepy.Stream(auth, Listener())
        stream.filter(track=QUERY, languages=["en"], encoding='utf-8')
    except KeyboardInterrupt as e:
        stream.disconnect()
        print("Stream disconnected. Goodbye!")


if __name__ == "__main__":
    main()