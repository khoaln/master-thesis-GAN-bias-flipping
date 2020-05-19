import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import  Model
from tqdm import tqdm
import numpy as np
from collections import namedtuple
import argparse
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='.', help='input folder')
parser.add_argument('--output', type=str, default='.', help='output folder')
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)

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
out = tf.keras.layers.Dense(6, activation="sigmoid", name="dense_output")(x)

model = tf.keras.models.Model(
      inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

def load_data(path):
  lines = []
  with open(path, 'r') as f:
    for line in f:
      lines.append(line.lower())
  return lines

train1 = load_data(os.path.join(args.input, "train1.txt"))
train2 = load_data(os.path.join(args.input, "train2.txt"))

inputs1=create_input_array(train1)
inputs2=create_input_array(train2)

model.fit(inputs1,np.ones(len(inputs1)),epochs=1,batch_size=32,validation_split=0.2,shuffle=True)
model.fit(inputs2,np.ones(len(inputs2)),epochs=1,batch_size=32,validation_split=0.2,shuffle=True)

test1 = load_data(os.path.join(args.input, "valid1.txt"))
test2 = load_data(os.path.join(args.input, "valid2.txt"))

test_inputs1=create_input_array(test1)
test_inputs2=create_input_array(test2)

pred1 = model.predict(test_inputs1)
pred2 = model.predict(test_inputs2)

labels1 = np.ones(len(pred1))
labels2 = np.zeros(len(pred2))

print('Accuracy: {}'.format(accuracy_score(labels1, pred1)))
print('Pre_Rec_F1: {}'.format(precision_recall_fscore_support(labels1, pred1, average='micro')))

print('Accuracy: {}'.format(accuracy_score(labels2, pred2)))
print('Pre_Rec_F1: {}'.format(precision_recall_fscore_support(labels2, pred2, average='micro')))