import tensorflow as tf
from dataset import *
from model import *
from train import *
from one_step_model import OneStep

new_model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dims,
    rnn_units=rnn_units
)
new_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
new_model.build(tf.TensorShape([1, None]))

one_step_model = OneStep(new_model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(["Love"])
result = [next_char]
for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print("\nRun time:", end-start)

def generate_text(start_str):
  start = time.time()
  states = None
  next_char = tf.constant([start_str])
  result = [next_char]
  for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

  result = tf.strings.join(result)
  end = time.time()
  print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
  print("\nRun time:", end-start)

start_str = "Truth"
generate_text("Truth")