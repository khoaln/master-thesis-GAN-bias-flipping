import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import xml.etree.ElementTree as ET
import nltk
import re
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from models import MLP_Classify, Seq2Seq2Decoder
from utils import to_gpu, batchify, Glove_Dictionary

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='classifier_model.pt', help='classifier model')
parser.add_argument('--autoencoder_model', type=str, default='autoencoder_model.pt', help='autoencoder_model model')
parser.add_argument('--inputDataset', type=str, help='input', default='.')
parser.add_argument('--outputDir', type=str, help='output', default='.')
parser.add_argument('--vocab', type=str, default="vocab.json",
                    help='path to load vocabulary from')
parser.add_argument('--ground_truth', type=str, default='')

parser.add_argument('--nhidden', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--arch_classify', type=str, default='128-128',
                    help='classifier architecture')
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--maxlen', type=int, default=100    ,
                    help='maximum sentence length')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='evaluation batch size')  
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_r', type=float, default=0.1,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr_classify', type=float, default=1e-04,
                    help='classifier learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--glove_vectors_file', type=str)
parser.add_argument('--glove_words_file', type=str)
parser.add_argument('--glove_word2idx_file', type=str)  
parser.add_argument('--mode', default='eval', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')

def tokenize(path):
  """Tokenizes a text file."""
  dropped = 0
  with open(path, 'r') as f:
    linecount = 0
    lines = []
    for line in f:
      linecount += 1
      L = line.lower()
      words = L.strip().split(" ")
      if args.maxlen > 0 and len(words) > args.maxlen:
        dropped += 1
        continue
      words = [BOS_WORD] + words + [EOS_WORD]
      # vectorize
      vocab = dictionary.word2idx
      unk_idx = vocab[UNK]
      indices = [vocab[w] if w in vocab else unk_idx for w in words]
      lines.append(indices)

  print("Number of sentences dropped from {}: {} out of {} total".
    format(path, dropped, linecount))
  return lines

def train_classifier(classifier, whichclass, batch):
  classifier.train()
  classifier.zero_grad()

  source, target, lengths = batch
  source = to_gpu(args.cuda, Variable(source))
  labels = to_gpu(args.cuda, Variable(torch.zeros(source.size(0)).fill_(whichclass-1)))

  # Train
  code = autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
  scores = classifier(code)
  classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
  classify_loss.backward()
  optimizer_classify.step()
  classify_loss = classify_loss.cpu().data[0]

  pred = scores.data.round().squeeze(1)
  accuracy = pred.eq(labels.data).float().mean()

  return classify_loss, accuracy

def eval_classifier(classifier, whichclass, batch):
  source, target, lengths = batch
  source = to_gpu(args.cuda, Variable(source))
  labels = to_gpu(args.cuda, Variable(torch.zeros(source.size(0)).fill_(whichclass-1)))

  code = autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
  scores = classifier(code)
  classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
  classify_loss.backward()
  classify_loss = classify_loss.cpu().data[0]

  pred = scores.data.round().squeeze(1)
  accuracy = pred.eq(labels.data).float().mean()

  return classify_loss, accuracy

def save_classifier_model(name='classifier_model.pt'):
  print("Saving model to {}".format(name))
  with open('{}/'.format(args.outputDir, name), 'wb') as f:
    torch.save(classifier.state_dict(), f)

########
args = parser.parse_args()
print(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
runOutputFileName = "prediction.txt"
PAD_WORD="<pad>"
EOS_WORD="<eos>"
BOS_WORD="<bos>"
UNK="<unk>"

#########
vocabdict = None
with open(args.vocab, 'r') as vocab_file:
  vocabdict = json.load(vocab_file)
  vocabdict = {k: int(v) for k, v in vocabdict.items()}
dictionary = Glove_Dictionary(vocabdict,
                glove_vectors_file=args.glove_vectors_file, 
                glove_words_file=args.glove_words_file, 
                glove_word2idx_file=args.glove_word2idx_file)

weights_matrix = np.zeros((len(dictionary.word2idx), args.emsize))
for word, i in dictionary.word2idx.items():
  try:
    weights_matrix[i] = dictionary.glove[word]
  except KeyError:
    weights_matrix[i] = np.random.normal(scale=0.6, size=(args.emsize, ))

weights_matrix = torch.from_numpy(weights_matrix).float()
weights_matrix = to_gpu(args.cuda, weights_matrix)

ntokens = len(dictionary.word2idx)
autoencoder = Seq2Seq2Decoder(emsize=args.emsize,
                      nhidden=args.nhidden,
                      ntokens=ntokens,
                      nlayers=args.nlayers,
                      noise_r=args.noise_r,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout,
                      gpu=args.cuda,
                      weights_matrix=weights_matrix)
autoencoder.load_state_dict(torch.load(args.autoencoder_model, map_location=lambda storage, loc: storage))
if args.cuda:
  autoencoder = autoencoder.cuda()

mode = args.mode
if mode == 'eval':
  classifier1 = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
  classifier1.load_state_dict(torch.load(os.path.join(args.outputDir, 'classifier1_model.pt'), 
    map_location=lambda storage, loc: storage))

  classifier2 = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
  classifier2.load_state_dict(torch.load(os.path.join(args.outputDir, 'classifier2_model.pt'), 
    map_location=lambda storage, loc: storage))

  print(classifier1)
  print(classifier2)
  if args.cuda:
    classifier1 = classifier1.cuda()
    classifier2 = classifier2.cuda()

  ground_truth = {}
  if args.ground_truth:
    gt_tree = ET.parse(args.ground_truth)
    gt_root = gt_tree.getroot()
    for child in gt_root:
      ground_truth[child.attrib['id']] = child.attrib['hyperpartisan']

  outFile = open("{}/{}".format(args.outputDir, runOutputFileName), 'w')
  test1_data = []
  test2_data = []
  dropped = 0
  linecount = 0
  article_ids = []
  labels = []
  for file in os.listdir(args.inputDataset):
    if file.endswith('.xml'):
      tree = ET.parse('{}/{}'.format(args.inputDataset, file))
      root = tree.getroot()
      for article in root.findall('article'):
        linecount += 1
        words = [word for word in nltk.word_tokenize(article.attrib['title'].lower()) if re.sub(r'[^\w\s]', '', word) != '']
        if args.maxlen > 0 and len(words) > args.maxlen:
          dropped += 1
          continue

        if len(words):
          words = [BOS_WORD] + words + [EOS_WORD]
          vocab = dictionary.word2idx
          unk_idx = vocab[UNK]
          indices = [vocab[w] if w in vocab else unk_idx for w in words]
          article_ids.append(article.attrib['id'])
          if article.attrib['id'] in ground_truth and ground_truth[article.attrib['id']] == 'true':
            test1_data.append(indices)
            labels.append('true')
          else:
            test2_data.append(indices)
            labels.append('false')

  print("Number of sentences dropped: {} out of {} total".format(dropped, linecount))
  print('Test set length: {}'.format(len(test1_data)))
  print('Test set length: {}'.format(len(test2_data)))

  test1_data = batchify(test1_data, args.eval_batch_size, shuffle=False)
  test2_data = batchify(test2_data, args.eval_batch_size, shuffle=False)

  # test classifier ----------------------------
  classify_loss, classify_acc = 0, 0
  for niter in range(len(test1_data)):
      classify_loss1, classify_acc1 = eval_classifier(classifier1, 1, test1_data[niter])
      classify_loss += classify_loss1
      classify_acc += classify_acc1

  classify_loss = classify_loss / (len(test1_data))
  classify_acc = classify_acc / (len(test1_data))
  print("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
                      classify_loss, classify_acc))

  classify_loss, classify_acc = 0, 0
  for niter in range(len(test2_data)):
      classify_loss2, classify_acc2 = eval_classifier(classifier2, 2, test2_data[niter])
      classify_loss += classify_loss2
      classify_acc += classify_acc2

  classify_loss = classify_loss / (len(test2_data))
  classify_acc = classify_acc / (len(test2_data))
  print("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
                      classify_loss, classify_acc))

elif mode == 'retrain':
  classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
  classifier.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
  optimizer_classify = optim.Adam(classifier.parameters(),
                                  lr=args.lr_classify,
                                  betas=(args.beta1, 0.999))

  print(classifier)
  if args.cuda:
    classifier = classifier.cuda()

  train1_data = tokenize(os.path.join(args.data_path, "train1.txt"))
  train2_data = tokenize(os.path.join(args.data_path, "train2.txt"))
  train1_data = batchify(train1_data, args.batch_size, shuffle=True)
  train2_data = batchify(train2_data, args.batch_size, shuffle=True)

  for niter in range(len(train1_data)):
    train_classifier(classifier, 1, train1_data[niter])
  save_classifier_model(name='classifier1_model.pt')

  for niter in range(len(train2_data)):
    train_classifier(classifier, 2, train2_data[niter])
  save_classifier_model(name='classifier2_model.pt')

else:
  print('Mode {} is not supported'.format(mode))



# classify_loss = classify_loss / (len(test1_data) + len(test2_data))
# classify_acc = classify_acc / (len(test1_data) + len(test2_data))
# print("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
#                     classify_loss, classify_acc))

# predictions = []
# for niter in range(len(test_data)):
#   # classifier.train()
#   # classifier.zero_grad()
#   source, target, lengths = test_data[niter]
#   source = to_gpu(args.cuda, Variable(source))
#   code = autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
#   scores = classifier(code)
#   # optimizer_classify.step()
#   pred = scores.data.round().squeeze(1)
#   for v in pred:
#     if v == 0:
#       predictions.append('true')
#     else:
#       predictions.append('false')

# print('{}, {}, {}'.format(len(predictions), len(article_ids), len(labels)))
# if len(article_ids) == len(predictions):
#   for i in range(len(article_ids)):
#     outFile.write('{} {} {} \n'.format(article_ids[i], predictions[i], labels[i]))

# outFile.close()

# print('Accuracy: {}'.format(accuracy_score(labels, predictions)))
# print('Pre_Rec_F1: {}'.format(precision_recall_fscore_support(labels, predictions, average='micro')))
