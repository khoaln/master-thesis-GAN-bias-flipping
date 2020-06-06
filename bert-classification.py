import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import  Model
from tqdm import tqdm
import numpy as np
from collections import namedtuple
import argparse
import os
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='.', help='input folder')
parser.add_argument('--output', type=str, default='.', help='output folder')
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--mode', default='eval', type=str)
parser.add_argument('--name', default='', type=str)
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[int(args.device_id)], 'GPU')
#   except RuntimeError as e:
#     # Visible devices must be set at program startup
#     print(e)

bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",trainable=True)

MAX_SEQ_LEN=128
input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                    name="segment_ids")


def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

FullTokenizer=bert.bert_tokenization.FullTokenizer

vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()

tokenizer=FullTokenizer(vocab_file,do_lower_case)

def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def create_single_input(sentence,MAX_LEN):
  
  stokens = tokenizer.tokenize(sentence)
  
  stokens = stokens[:MAX_LEN]
  
  stokens = ["[CLS]"] + stokens + ["[SEP]"]

  ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
  masks = get_masks(stokens, MAX_SEQ_LEN)
  segments = get_segments(stokens, MAX_SEQ_LEN)

  return ids,masks,segments

def create_input_array(sentences):

  input_ids, input_masks, input_segments = [], [], []

  for sentence in tqdm(sentences,position=0, leave=True):
  
    ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2)

    input_ids.append(ids)
    input_masks.append(masks)
    input_segments.append(segments)

  return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(x)

model = tf.keras.models.Model(
      inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=args.lr),
                  metrics=['accuracy'])

def load_data(path, whichclass):
  lines = []
  with open(path, 'r') as f:
    for line in f:
      lines.append((line.lower(), whichclass))
  return lines

checkpoint_path = os.path.join(args.output, "model.ckpt")

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Logging callback
logdir = "logs/scalars/" + args.name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

if args.mode == 'train':
  train = []
  train.extend(load_data(os.path.join(args.input, "train1.txt"), 1))
  train.extend(load_data(os.path.join(args.input, "train2.txt"), 0))
  random.shuffle(train)
  train = np.array(train)
  inputs=create_input_array(train[:,0])

  model.fit(
    inputs,
    np.array(train[:,1], dtype=int),
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_split=0,
    shuffle=True,
    callbacks=[cp_callback])
else:
  # Loads the weights
  model.load_weights(checkpoint_path)

  test = []
  test.extend(load_data(os.path.join(args.input, "valid1.txt"), 1))
  test.extend(load_data(os.path.join(args.input, "valid2.txt"), 0))
  random.shuffle(test)
  test = np.array(test)
  test_inputs=create_input_array(test[:,0])

  pred = model.predict(test_inputs)
  pred = np.array(pred, dtype=int)
  labels = np.array(test[:,1], dtype=int)

  print('Accuracy: {}'.format(accuracy_score(labels, pred)))
  print('Pre_Rec_F1: {}'.format(precision_recall_fscore_support(labels, pred, average='macro')))