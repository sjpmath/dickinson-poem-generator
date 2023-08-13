import tensorflow as tf
import numpy as np
import os
import time


FILE_URL = "emilydickinson.txt"
data = open(FILE_URL, 'rb')
text = data.read().decode(encoding='utf-8').strip()
vocab = sorted(set(text))

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
) # need be able to invert representation and recover human-readable strings from id

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
  # reduce_join used to join chars back into strings

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
  input_text = sequence[:-1]
  target_text = sequence[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

