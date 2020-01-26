# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:13:07 2020

@author: Emmanuel
"""
import tensorflow as tf
import numpy as np

## Restore the latest checkpoint
checkpoint_dir = './training_checkpoints'

tf.train.latest_checkpoint(checkpoint_dir)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

with open('Data/OpenSubtitles.fr-is.fr', encoding='utf-8') as t:
    text = t.read()
vocab = sorted(set(text))
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

## The prediction loop

def generate_text2(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  
  char2idx = {u:i for i, u in enumerate(vocab)}
  
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
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

  return (start_string + ''.join(text_generated))