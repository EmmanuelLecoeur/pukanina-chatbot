# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:58:51 2020

@author: Emmanuel
"""

from requirement import *
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class chatbotPerou():

  def __init__(self,
               QA=QA,
               vectorizer1=vectorizer1,
               vectorizer2=vectorizer2,
               classifieur1=classifieur1,
               classifieur2=classifieur2,
               vectorizer_themes=vectorizer_themes,
               normalise=normalise,
               vocab=vocab,
               model=build_model()):
     self.QA = QA
     self.vectorizer1 = vectorizer1 #vectorizer utilisé pour la pertinence globale
     self.vectorizer2 = vectorizer2 #vectorizer utilisé pour la pertinence thématique
     self.classifieur1 = classifieur1 #pertinence globale
     self.classifieur2 = classifieur2 #pertinence thématique
     self.vectorizer_themes = vectorizer_themes
     self.vocab = vocab
     self.normalise = normalise
     self.model = model
     #self.generateur = reseau_generateur
  
  def pertinence_metier(self,user_input):
    """
    :input user_input: question utilisateur brute
    :output: question pertinente à la thématique métier ? (bool)
    """
    user_norm1 = pd.Series([user_input]).apply(self.normalise)
    user_vect1 = self.vectorizer1.transform(user_norm1)
    return self.classifieur1.predict(user_vect1)[0]=="metier"
  
  def pertinence_theme(self,user_input):
    """
    :input user_vect: question utilisateur brute
    :output: thématique pour laquelle la question est la plus pertinente (str)
    """
    user_norm2 = pd.Series([user_input]).apply(self.normalise)
    user_vect2 = self.vectorizer2.transform(user_norm2)
    return self.classifieur2.predict(user_vect2)[0]

  def respond_bdd(self,user_input,theme): #A FAIRE
    """
    :input user_vect: user_input vectoriser de la même manière que pour les
                      données d'entrainement du classifieur2 (vect) EST CE QU'IL U'
    :input theme: le thème des questions/réponses sur lesquelles calculer 
                  la similarité (str)

    :output: réponse la plus pertinente parmi celles du thème
    """
    vect, dtm, ind = self.vectorizer_themes[theme]
    user_norm = pd.Series([user_input]).apply(self.normalise)
    user_vector = vect.transform(user_norm)
    query_corpus_sim = np.squeeze(cosine_similarity(dtm, user_vector))
    idx_most_sim = np.argmax(query_corpus_sim)
    best = ind[idx_most_sim%len(ind)]
    return(str(QA.iloc[best].Answers))

  def respond_generative(self,texte, temperature = 1.0):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 200

    # Converting our start string to numbers (vectorizing)
  
    char2idx = {u:i for i, u in enumerate(vocab)}
  
    input_eval = [char2idx[s] for s in texte]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.

    # Here batch size == 1
    self.model.reset_states()
    for i in range(num_generate):
      predictions = self.model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      
      idx2char = np.array(vocab)

      text_generated.append(idx2char[predicted_id])
      res = texte + ''.join(text_generated)
    return (re.sub(r"- ", "", res).split('.')[1] + '.')


  def converse(self, quit="quit"):
    try:
      user_input = input(">")
    except EOFError:
      print(user_input)

    while user_input != quit:
      if user_input:
          try:
              #Pertinence métier ?
              metier = self.pertinence_metier(user_input)
          except ValueError:
              print(self.respond_generative(user_input, temperature = 0.5))
          if metier:
              try:
                  #Thème pertinent ?
                  theme = self.pertinence_theme(user_input)
                  print(self.respond_bdd(user_input,theme))
              except ValueError:
                  (self.respond_generative(user_input, temperature = 0.5))
          else :
            #partie générative
            print(self.respond_generative(user_input, temperature = 0.5))
      user_input = quit
      try:
        user_input = input(">")
      except EOFError:
        print(user_input)
        
voyage = chatbotPerou()
voyage.converse()