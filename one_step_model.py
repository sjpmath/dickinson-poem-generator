import tensorflow as tf
from dataset import *
from model import *
from train import *

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0): # model, charsfromids layer, idsfromchars layer
    # higher the temperature, the more random the predictions
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # create a mask to prevent '[UNK]' from being generated
    skip_ids = self.ids_from_chars(['[UNK]'])[:,None] # add an extra dimension e.g. (4,) to (4,1)
    # word to skip i.e. seq of chars to keep from being generated
    sparse_mask = tf.SparseTensor(
        # Put -inf at each bad index
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        dense_shape=[len(ids_from_chars.get_vocabulary())] # match shape to the vocab
    )#This makes a tensor with -inf at indices for bad chars, 0 for others

    self.prediction_mask = tf.sparse.to_dense(sparse_mask)
    # tf.sparse_to_dense converts sparse representation to dense vector - just changing representation to a tensor format
    # sparse rep is a form where only nonzero entry indices are marked

  @tf.function # makes graphs out of func
  def generate_one_step(self, inputs, states=None): # inputs is the char seq generated so far, states = states of machine so far
    # convert strings to token ids
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # run model, get predicted logits
    # shape of predicted logits is (batchsize, seq_length, vocab_size) - batchsize=1 if just one being done
    predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
    # get the shifted seq with last predicted char added

    # only use the last prediction
    predicted_logits = predicted_logits[:, -1, :] # predictions for last pos only
    predicted_logits = predicted_logits/self.temperature

    # apply prediction mask to prevent '[UNK]' from being generated
    predicted_logits = predicted_logits + self.prediction_mask # non bad chars unaffected (0 added), bad chars -inf added - negative prob
    # hence prevent from being chosen

    # sample output logits to generate token ids
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=1)

    # convert ids to chars
    predicted_chars = self.chars_from_ids(predicted_ids)

    # return the chars and model state
    return predicted_chars, states