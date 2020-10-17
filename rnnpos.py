import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy
import random
import torch
import torch.nn as nn
import torch.functional as F

mydevice = torch.device('cuda')

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size, vocab_size=0):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # Student added embedding layer
        self.word_embeddings = 0
        if vocab_size != 0:
          self.word_embeddings = nn.Embedding(vocab_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)
        # self.activation = nn.Softmax(dim=-1)

    def forward(self, sentence):
        if self.word_embeddings:
          sentence = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.activation(tag_space)
        #tag_scores = self.dropout(tag_scores)
        return tag_scores

def train(training_file):
    assert os.path.isfile(training_file), 'Training file does not exist'

    # Your code starts here
    from time import time
    from collections import Counter
    from numpy import asarray, zeros, float64
    from torch.optim import Adam, SGD

    embedding_matrix = {}
    e_dim = 0
    voca = Counter()
    rare = None
    # Try to find existing embedding matrix, vocabulary, and tag list.
    try:
      with open('embedding_matrix.pkl', mode='rb') as infile:
        embedding_matrix = pickle.load(infile)
        e_dim = len(embedding_matrix['the'])
      with open('vocabulary.pkl', mode='rb') as infile:
        voca = pickle.load(infile)
      rare = [word for word, count in voca.items() if count < 3]
      rare = set(rare)
      with open('alltags.pkl', mode='rb') as infile:
        alltags = pickle.load(infile)
      print('Found files for embedding matrix, vocabulary, and tag list.')
    
    except:
      print('No existing embedding matrix, vocabulary, or tag list.')
      # Load word embedding
      count = 0
      with open('glove.6B.300d.txt', mode='r', encoding='utf8') as infile:
          for row in infile.read().splitlines():
              row = row.split()
              embedding_matrix[row[0]] = asarray(row[1:], dtype=float64)
              count += 1

      e_dim = len(embedding_matrix['the'])
      
      # Make 'unknown' token in embedding matrix
      embedding_matrix['UNKA'] = zeros(e_dim)
      for _, val in embedding_matrix.items():
          embedding_matrix['UNKA'] += val
      embedding_matrix['UNKA'] /= count

      # Save embedding matrix as a file
      with open('embedding_matrix.pkl', mode='wb') as outfile:
        pickle.dump(embedding_matrix, outfile)

      # Make vocabulary and change rare words to 'UNKA'
      # (occurs less than 3 times)
      # Make vocabulary and tag set.
      alltags = set(['PAD'])
      with open(training_file, mode='r') as infile:
          for sentag in infile.read().splitlines():
              voca.update([word.lower() for word in sentag.split()[::2]])
              alltags.update(sentag.split()[1::2])
      rare = [word for word, count in voca.items() if count < 3]
      rare = set(rare)

      # Save tag list as a file
      with open('alltags.pkl', mode='wb') as outfile:
        pickle.dump(alltags, outfile)
      
      voca['UNKA'] = 0
      for key, val in voca.copy().items():
          if key in rare:
              voca['UNKA'] += val
              del voca[key]
      
      # Save vocabulary as a file
      with open('vocabulary.pkl', mode='wb') as outfile:
        pickle.dump(voca, outfile)
      print('Finish making files for embedding matrix, vocabulary, and tag list.')
          
    # Make a list of training data while changing rare words to 'UNKA'
    # [([word, ..., word], [tag, ..., tag]), ..., ([], [...tag])]
    training_list = []
    with open(training_file, mode='r') as infile:
        for sentag in infile.read().splitlines():
            sent = [word.lower() for word in sentag.split()[::2]]
            sent = ['UNKA' if word in rare else word for word in sent]
            tags = [tag for tag in sentag.split()[1::2]]
            sentag = (sent, tags)
            training_list.append(sentag)

    # Detach the last 10% of training data for validation
    validation_list = training_list[-len(training_list) // 10:]

    # Assign integers to each tags
    tag_to_idx = {'PAD':0}
    for tag in alltags:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)
    
    ts_size = len(alltags)
    model = LSTMTagger(embedding_dim=e_dim,
                       hidden_dim=ts_size * 2,
                       tagset_size=ts_size,
                       vocab_size=0 # 0 if using external embedding
                       ).to(device=mydevice)
    loss_function = nn.CrossEntropyLoss().to(device=mydevice)
    optimizer = SGD(model.parameters(), lr=0.1)
    last_epoch = 0
    
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[word] for word in seq]
        return torch.tensor(idxs, dtype=torch.long)
    
    def sent2embed(sent):
        from torch import tensor
        evec = tensor([embedding_matrix[word]
                        if word in embedding_matrix
                        else embedding_matrix['UNKA']
                        for word in sent])
        return evec

    try:
      # Try loading existing model
      checkpoint = torch.load('model.torch')
      last_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      loss = checkpoint['loss']
      print("Model loaded. Last epoch: %3d / Loss: %f" % (last_epoch, loss))
    
    except Exception as ee:
      # If model does not exist
      print("Excpetion Occured: ", ee)
      print("No model exists. Make a new one.")

    # Show score before training
    with torch.no_grad():
      model.eval()
      inputs = sent2embed(training_list[0][0]).to(device=mydevice)
      tag_scores = model(inputs.view(len(inputs), 1, -1).float())
      #print(tag_scores)
    
    # Check validation accuracy
    model.eval()
    prediction = []
    ground_truth = []
    for sent, tags in validation_list:
      model.zero_grad()
      # Change all tags of UNKA to PAD
      for ii in range(len(sent)):
        if sent[ii] == 'UNKA':
          tags[ii] = 'PAD'
      
      # Turn each sentence to tensor of word indices
      sent_embed = sent2embed(sent).to(device=mydevice)
      tags_idx = prepare_sequence(tags, tag_to_idx).to(device=mydevice)

      tag_scores = model(sent_embed.view(len(sent_embed), 1, -1).float())
      best_tag_idx = torch.argmax(tag_scores, axis=-1).tolist()
      prediction.extend(best_tag_idx)
      ground_truth.extend(tags_idx.tolist())
    
    print(f'Validation accuracy: {100*accuracy_score(prediction, ground_truth):6.2f}%')
    
    for epoch in range(last_epoch, 300):
        model.train()
        start_t = time()
        print("Epoch %3d Started" % (epoch + 1))
        for sent, tags in training_list:
            # Clear gradients before each epoch
            model.zero_grad()
            
            # Change all tags of UNKA to PAD
            for ii in range(len(sent)):
              if sent[ii] == 'UNKA':
                tags[ii] = 'PAD'

            # Turn each sentence to tensor of word indices
            sent_embed = sent2embed(sent).to(device=mydevice)
            tags_idx = prepare_sequence(tags, tag_to_idx).to(device=mydevice)
            
            # Run forward pass.
            tag_scores = model(sent_embed.view(len(sent), 1, -1).float())
            
            # Compute loss, grads, and update params using optimizer
            loss = loss_function(tag_scores, tags_idx)
            loss.backward()
            optimizer.step()

        # Check validation accuracy
        model.eval()
        prediction = []
        ground_truth = []
        for sent, tags in validation_list:
          model.zero_grad()
          # Change all tags of UNKA to PAD
          for ii in range(len(sent)):
            if sent[ii] == 'UNKA':
              tags[ii] = 'PAD'
          
          # Turn each sentence to tensor of word indices
          sent_embed = sent2embed(sent).to(device=mydevice)
          tags_idx = prepare_sequence(tags, tag_to_idx).to(device=mydevice)

          tag_scores = model(sent_embed.view(len(sent_embed), 1, -1).float())
          best_tag_idx = torch.argmax(tag_scores, axis=-1).tolist()
          prediction.extend(best_tag_idx)
          ground_truth.extend(tags_idx.tolist())
        
        print(f'Validation accuracy: {100*accuracy_score(prediction, ground_truth):6.2f}%')
        
        torch.save({'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss': loss,
                    'e_dim': e_dim,
                    'ts_size': ts_size,
                    'tag_to_idx': tag_to_idx},
                    'model.torch')
        print("Epoch %3d Ended. Elapsed time (s): %d / Loss: %f" % (epoch + 1, time() - start_t, loss))
        if (epoch + 1) % 100 == 0:
          torch.save({'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss': loss,
                    'e_dim': e_dim,
                    'ts_size': ts_size,
                    'tag_to_idx': tag_to_idx},
                    'model_%3d_epochs.torch' % (epoch + 1))
    
    
    # Your code ends here

    return model

def test(model_file, data_file, label_file):
    assert os.path.isfile(model_file), 'Model file does not exist'
    assert os.path.isfile(data_file), 'Data file does not exist'
    assert os.path.isfile(label_file), 'Label file does not exist'

    # Your code starts here
    from time import time
    start_t = time()
    #model = LSTMTagger(1,1,10)
    #model.load_state_dict(torch.load(model_file))

    checkpoint = torch.load(model_file)
    e_dim = checkpoint['e_dim']
    ts_size = checkpoint['ts_size']
    model = LSTMTagger(embedding_dim=e_dim, hidden_dim=ts_size * 4,tagset_size=ts_size).to(device=mydevice)
    model.load_state_dict(checkpoint['model_state_dict'])
    tag_to_idx = checkpoint['tag_to_idx']

    # Load word embedding
    # because saving embedding in the model is to heavy
    from numpy import asarray, zeros, float64
    embedding_matrix = {}
    count = 0
    with open('glove.6B.300d.txt', mode='r', encoding='utf8') as infile:
        for row in infile.read().splitlines():
            row = row.split()
            embedding_matrix[row[0]] = asarray(row[1:], dtype=float64)
            count += 1
    e_dim = len(embedding_matrix['the'])
    
    # Make 'unknown' token in embedding matrix
    embedding_matrix['UNKA'] = zeros(e_dim)
    for _, val in embedding_matrix.items():
        embedding_matrix['UNKA'] += val
    embedding_matrix['UNKA'] /= count

    # Change to evaluation mode (from training mode)
    model.eval()

    # Load test data and save it as a list.
    # Each sentence is a row of the list.
    data = []
    with open(data_file, mode='r') as infile:
        for sent in infile.read().splitlines():
            sent = [word.lower() if word != 'UNKA' else 'UNKA' for word in sent.split()]
            data.append(sent)

    def sent2embed(sent):
        from torch import tensor
        evec = tensor([embedding_matrix[word]
                        if word in embedding_matrix
                        else embedding_matrix['UNKA']
                        for word in sent])
        return evec
    print('Preparation time for test: ', time() - start_t)
    #prediction = model(torch.rand(1000,1,1))   # replace with inferrence from the loaded model
    #prediction = torch.argmax(prediction,-1).to(device=mydevice).numpy()
    # Infer
    start_t = time()
    prediction = []
    with torch.no_grad():
      model.eval()
      for sent in data:
        inputs = sent2embed(sent).to(device=mydevice)
        tag_scores = model(inputs.view(len(inputs), 1, -1).float())
        best_tag_idx = torch.argmax(tag_scores, axis=-1).tolist()
        prediction.extend(best_tag_idx)
    print('prediction[:10]: ', prediction[:10])

    # Load label and put it in a list
    ground_truth = []
    with open(label_file, mode='r', errors='ignore') as infile:
      for sentag in infile.read().splitlines():
        tags = sentag.split()[1::2]
        ground_truth.extend(tags)
    ground_truth = [tag_to_idx[tag] for tag in ground_truth]
    print("ground_truth[:10]: ", ground_truth[:10])

    print("Test time: ", time() - start_t)
    # Your code ends here

    print(f'The accuracy of the model is {100*accuracy_score(prediction, ground_truth):6.2f}%')

def main(params):
    if params.train:
        model = train(params.training_file)
        torch.save(model.state_dict(), params.model_file)
    else:
        test(params.model_file, params.data_file, params.label_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNN POS Tagger")
    parser.add_argument("--train", action='store_const', const=True, default=False)
    parser.add_argument('--model_file', type=str, default='model.torch')
    parser.add_argument('--training_file', type=str, default='')
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--label_file', type=str, default='')

    main(parser.parse_args())
