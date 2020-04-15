import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import os
import xml.etree.ElementTree as ET
import nltk
import re
import json

from models import MLP_Classify, Seq2Seq2Decoder
from utils import to_gpu, batchify, Dictionary

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='classifier_model.pt', help='model')
parser.add_argument('--inputDataset', type=str, help='input', default='.')
parser.add_argument('--outputDir', type=str, help='output', default='.')
parser.add_argument('--vocab', type=str, default="vocab.json",
                    help='path to load vocabulary from')

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
parser.set_defaults(cuda=False)

args = parser.parse_args()
print(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
runOutputFileName = "prediction.txt"
PAD_WORD="<pad>"
EOS_WORD="<eos>"
BOS_WORD="<bos>"
UNK="<unk>"

classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
classifier.load_state_dict(torch.load(args.model))
optimizer_classify = optim.Adam(classifier.parameters(),
                                lr=args.lr_classify,
                                betas=(args.beta1, 0.999))

print(classifier)
vocabdict = None
with open(args.vocab, 'r') as vocab_file:
  vocabdict = json.load(vocab_file)
  vocabdict = {k: int(v) for k, v in vocabdict.items()}
dictionary = Dictionary(vocabdict)

ntokens = len(dictionary.word2idx)
autoencoder = Seq2Seq2Decoder(emsize=args.emsize,
                      nhidden=args.nhidden,
                      ntokens=ntokens,
                      nlayers=args.nlayers,
                      noise_r=args.noise_r,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout,
                      gpu=args.cuda)

if args.cuda:
  autoencoder = autoencoder.cuda()
  classifier = classifier.cuda()

outFile = open("{}/{}".format(args.outputDir, runOutputFileName), 'w')
test_data = []
dropped = 0
linecount = 0
article_ids = []
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
        test_data.append(indices)

print("Number of sentences dropped: {} out of {} total".format(dropped, linecount))
print('Test set length: {}'.format(len(test_data)))

test_data = batchify(test_data, args.eval_batch_size, shuffle=False)
predictions = []
for niter in range(len(test_data)):
  classifier.train()
  classifier.zero_grad()
  source, target, lengths = test_data[niter]
  source = to_gpu(args.cuda, Variable(source))
  code = autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
  scores = classifier(code)
  optimizer_classify.step()
  pred = scores.data.round().squeeze(1)
  for v in pred:
    if v == 0:
      predictions.append('true')
    else:
      predictions.append('false')

print('{}, {}'.format(len(predictions), len(article_ids)))
if len(article_ids) == len(predictions):
  for i in range(len(article_ids)):
    outFile.write('{} {} \n'.format(article_ids[i], predictions[i]))

outFile.close()
