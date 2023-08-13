import tensorflow as tf
from dataset import *

vocab_size = len(ids_from_chars.get_vocabulary())
embedding_dims = 256
rnn_units = 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # maps each char id to a vector with embdding_dim dims
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True, stateful=True, recurrent_initializer='glorot_uniform')
    # rnn_units - dimensionality of output_space
    # return_sequences - whether to return last output in out seq, or whole seq
    # return_state - whether to return last state along with output
    self.dense = tf.keras.layers.Dense(vocab_size)
    # output one logit for each char in the vocabulary
    # for each char, probabilities of next char

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x) # special case for when the char is the first char of sequence
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

