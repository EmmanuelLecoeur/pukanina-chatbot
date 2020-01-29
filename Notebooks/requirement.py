# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:59:40 2020

@author: Emmanuel
"""

import pandas as pd
import spacy
from collections import defaultdict, Counter
import numpy as np
from sklearn.externals import joblib
import tensorflow as tf
import re

tf.compat.v1.enable_eager_execution()

#Importation pré-traitement
import nltk
from nltk.tokenize.regexp import WordPunctTokenizer
from nltk.stem import SnowballStemmer
#from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('stopwords')
#sw=stopwords.words("french")
#sw += ['être','avoir']
nlp = spacy.load('fr_core_news_md')

#Importation modèles
#from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.linear_model import RidgeClassifier
#from sklearn.naive_bayes import MultinomialNB

#Importation données
QA = pd.read_csv('../Data/Q_A.csv',sep=";")
vectorizer2 = joblib.load('../Data/vectoriseur2.pkl')
classifieur2 = joblib.load('../Data/classifieur2.pkl')
vectorizer1 = joblib.load('../Data/vectoriseur1.pkl')
classifieur1 = joblib.load('../Data/classifieur1.pkl')
vectorizer_themes = joblib.load('../Data/vectorizer_themes.pkl')
vocab = joblib.load('../Data/vocab.pkl')

#Définition des fonctions de prétraitement du texte
def lemmatise_text(text):
  tw_nlp = nlp(text)
  list_lem = [token.lemma_ for token in tw_nlp]
  text_lem = ' '.join(list_lem)
  return text_lem

def stem_text(text):
  tokenizer = WordPunctTokenizer()
  stemmer = SnowballStemmer('french')
  liste_racines = [stemmer.stem(token) for token in tokenizer.tokenize(text)]
  return ' '.join(liste_racines)

def normalise(text):
  #stop words, strip accent et lowercase vont être fait automatiquement
  text = text.replace('\n','').replace('\r','').split(" ")
  text = " ".join([i for i in text if i!=""])
  lemmas = lemmatise_text(text) #lemme de notre texte
  stems = stem_text(lemmas) #stem de notre texte A VOIR
  return stems

def build_model():
  vocab_size = len(vocab)
  embedding_dim = 256
  rnn_units = 1024
  batch_size = 1
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  model.load_weights(tf.train.latest_checkpoint('../Data/training_checkpoints')).expect_partial()
  model.build(tf.TensorShape([1, None]))
  return model