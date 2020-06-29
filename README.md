# Generative adversarial networks for bias flipping

## Datasets
In order to build the training and test data files, refer two notebooks:
  - _Get-Data-Webis-Bias-Flipper-18.ipynb_ for the "Webis Bias Flipper 18" dataset
  - _PAN-get-title-n-content.ipynb_ for the "Hyperpartisan News" dataset
  
## Run the model
The model code is placed in folder ```arae/pan```.
Sample command to train and test the model:
```
python train.py --data_path ./webis/ --epochs 500 --lr_ae 5 --lr_classify 1e-05 --lr_gan_g 5e-05 --dropout 0 --device_id 6 --batch_size 128 --maxlen 100 --cuda --glove_vectors_file ~/glove/6B.300.dat --glove_words_file ~/glove/6B.300_words.pkl --glove_word2idx_file --/glove/6B.300_idx.pkl --emsize 300 --outf output --mode train
```
Parameters:
  - data_path: path to the folder that contains the training and test files: train1.txt, train2.txt, valid1.txt, valid2.txt
  - epochs: number of epochs
  - lr_ae: autoencoder learning rate
  - lr_classify: critic learning rate
  - lr_gan_r: GAN generator learning rate
  - dropout
  - device_id: ID of the GPU to run the model with
  - batch_size: batch size
  - maxlen: maximum length of input sentence
  - cuda: train the model on GPUs
  - glove_vectors_file, glove_words_file, glove_word2idx_file: GloVe files. Can refer to https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76 to make the GloVe files from data.
  - emsize: embedding size
  - outf: output folder
  - mode: train | test. With "train" mode, the model will also run the test after training and generate the bias-flipped sentences. "test" mode is to run the test only.
  
## References
  - Martín Pellarolo, How to use Pre-trained Word Embeddings in PyTorch, 2018. https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
