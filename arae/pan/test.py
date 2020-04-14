import argparse
import torch
import os

from models import MLP_Classify

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='model')
parser.add_argument('--inputDataset', type=str, help='input')
parser.add_argument('--outputDir', type=str, help='output')

parser.add_argument('--nhidden', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--arch_classify', type=str, default='128-128',
                    help='classifier architecture')
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id  

classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
classifier.load_state_dict(torch.load(args.model))
classifier = classifier.cuda()

print(classifier)
