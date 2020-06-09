import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='.', help='input file')
parser.add_argument('--output', type=str, default='.', help='output file')

args = parser.parse_args()

file = open(args.output, "w+")
with open(args.input, "r") as f:
  for line in f:
    sent = line.split('<eos>')
    sent = sent[0]
    sent = sent.replace(' <eos>', '')
    sent = sent.replace(' <unk>', '')
    sent = sent.replace('<eos>', '')
    sent = sent.replace('<unk>', '')
    file.write(sent + "\n")
file.close()