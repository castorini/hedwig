import os 
import sys

import time
import glob
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import utils

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Trainer(object):
    
    def __init__(self, model, eta, mom, no_loss_reg, vec_dim):
        # set the random seeds for every instance of trainer. 
        # needed to ensure reproduction of random word vectors for out of vocab terms
        torch.manual_seed(1234)
        np.random.seed(1234)
        self.unk_term = np.random.uniform(-0.25, 0.25, vec_dim) 

        self.reg = 1e-5
        self.no_loss_reg = no_loss_reg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=eta, momentum=mom, weight_decay=(0 if no_loss_reg else self.reg) ) 

        self.datasets = {}
        self.embeddings = {}
        self.vec_dim = vec_dim
        

    def load_input_data(self, dataset_root_folder, word_vectors_cache_file, train_set_folder, dev_set_folder, test_set_folder):
        for set_folder in [test_set_folder, dev_set_folder, train_set_folder]:
            if set_folder:
                self.datasets[set_folder] = utils.read_in_dataset(dataset_root_folder, set_folder)
                # NOTE: self.datasets[set_folder] = questions, sentences, labels, vocab, maxlen_q, maxlen_s, ext_feats 
                self.embeddings[set_folder] = utils.load_cached_embeddings(word_vectors_cache_file, 
                    self.datasets[set_folder][3], [] if "train" in set_folder else self.unk_term)
            
        
    def regularize_loss(self, loss):       
        
        flattened_params = []

        for p in self.model.parameters():
            f = p.data.clone()
            flattened_params.append(f.view(-1))
        
        fp = torch.cat(flattened_params)

        loss = loss + 0.5 * self.reg * fp.norm() * fp.norm()

        # for p in self.model.parameters():
        #     loss = loss + 0.5 * self.reg * p.norm() * p.norm()
            
        return loss    

    
    def _train(self, xq, xa, ext_feats, ys):
                
        self.optimizer.zero_grad()        
        output = self.model(xq, xa, ext_feats)                    
        loss = self.criterion(output, ys)        
        # logger.debug('loss after criterion {}'.format(loss))

        # NOTE: regularizing location 1
        if not self.no_loss_reg:
             loss = self.regularize_loss(loss)
        #     logger.debug('loss after regularizing {}'.format(loss))
        
        loss.backward()
        
        # logger.debug('AFTER backward')
        #logger.debug('params {}'.format([p for p in self.model.parameters()]))
        # logger.debug('params grads {}'.format([p.grad for p in self.model.parameters()]))
       
        # NOTE: regularizing location 2. It would seem that location 1 is correct?
        #if not self.no_loss_reg:
        #    loss = self.regularize_loss(loss)
            # logger.debug('loss after regularizing {}'.format(loss))

        self.optimizer.step()

        # logger.debug('AFTER step')
        #logger.debug('params {}'.format([p for p in self.model.parameters()]))
        # logger.debug('params grads {}'.format([p.grad for p in self.model.parameters()]))

        return loss.data[0], self.pred_equals_y(output, ys)


    def pred_equals_y(self, pred, y):        
        _, best = pred.max(1)        
        best = best.data.long().squeeze()        
        return torch.sum(y.data.long() == best)


    def test(self, set_folder, batch_size):
        logger.info('----- Predictions on {} '.format(set_folder))
                
        questions, sentences, labels, vocab, maxlen_q, maxlen_s, ext_feats = self.datasets[set_folder]
        word_vectors, vec_dim = self.embeddings[set_folder], self.vec_dim
        
        self.model.eval()

        batch_size = 1

        total_loss = 0.0
        total_correct = 0.0
        num_batches = np.ceil(len(questions)/batch_size )
        y_pred = np.zeros(len(questions))
        ypc = 0
            
        for k in xrange(int(num_batches)):
            batch_start = k * batch_size
            batch_end = (k+1) * batch_size
            # convert raw questions and sentences to tensors
            batch_inputs, batch_labels = self.get_tensorized_inputs(
                    questions[batch_start:batch_end], 
                    sentences[batch_start:batch_end], 
                    labels[batch_start:batch_end],
                    ext_feats[batch_start:batch_end], 
                    word_vectors, vocab, vec_dim
                )
            
            xq, xa, x_ext_feats = batch_inputs[0]
            y = batch_labels[0]
            
            pred = self.model(xq, xa, x_ext_feats)            
            loss = self.criterion(pred, y)        
            pred = torch.exp(pred)
            total_loss += loss
            # total_correct += self.pred_equals_y(pred, y)

            y_pred[ypc] = pred.data.squeeze()[1] # we want to score for relevance, NOT the predicted class
            ypc += 1         
               
        # logger.info('{}_correct {}'.format(set_folder, total_correct))
        # logger.info('{}_loss {}'.format(set_folder, total_loss.data[0]))
        logger.info('{} total {}'.format(set_folder, len(labels)))
        # logger.info('{}_loss = {:.4f}, acc = {:.4f}'.format( set_folder, total_loss.data[0]/len(labels), float(total_correct)/len(labels) ))
        #logger.info('{}_loss = {:.4f}'.format( set_folder, total_loss.data[0]/len(labels) ))

        return y_pred


    def train(self, set_folder, batch_size, debugSingleBatch):
        train_start_time = time.time()

        questions, sentences, labels, vocab, maxlen_q, maxlen_s, ext_feats = self.datasets[set_folder]
        word_vectors, vec_dim = self.embeddings[set_folder], self.vec_dim

        # set model for training modep
        self.model.train()

        train_loss, train_correct = 0., 0.
        num_batches = np.ceil(len(questions)/float(batch_size) )

        for k in xrange(int(num_batches)):
            batch_start = k * batch_size
            batch_end = (k+1) * batch_size

            # convert raw questions and sentences to tensors
            batch_inputs, batch_labels = self.get_tensorized_inputs(
                    questions[batch_start:batch_end], 
                    sentences[batch_start:batch_end], 
                    labels[batch_start:batch_end],
                    ext_feats[batch_start:batch_end], 
                    word_vectors, vocab, vec_dim
                )
            
            xq, xa, x_ext_feats = batch_inputs[0]

            ys = batch_labels[0]

            batch_loss, batch_correct = self._train(xq, xa, x_ext_feats, ys)                                        
            
            # logger.debug('batch_loss {}, batch_correct {}'.format(batch_loss, batch_correct))
            train_loss += batch_loss
            # train_correct += batch_correct
            if debugSingleBatch: break

        # logger.info('train_correct {}'.format(train_correct))
        logger.info('train_loss {}'.format(train_loss))
        logger.info('total training batches = {}'.format(num_batches))
        logger.info('train_loss = {:.4f}'.format(
            train_loss/num_batches
        ))
        logger.info('training time = {:.3f} seconds'.format(time.time() - train_start_time))
        return train_correct/num_batches
        

    def make_input_matrix(self,  sentence, word_vectors, vec_dim):
        terms = sentence.strip().split()        
        # word_embeddings = torch.zeros(max_len, vec_dim).type(torch.DoubleTensor)
        word_embeddings = torch.zeros(len(terms), vec_dim).type(torch.DoubleTensor)
        for i in xrange(len(terms)):
            word = terms[i]        
            emb = torch.from_numpy(word_vectors[word])                        
            word_embeddings[i] = emb            

        input_tensor = torch.zeros(1, vec_dim, len(terms))
        input_tensor[0] = torch.transpose(word_embeddings, 0 , 1)
        return input_tensor


    def get_tensorized_inputs(self, batch_ques, batch_sents, batch_labels, batch_ext_feats, word_vectors, vocab, vec_dim):
        batch_size = len(batch_ques)
        # NOTE: ideal batch size is one, because sentences are all of different length.
        # In other words, we have no option but to feed in sentences one by one into the model
        # and compute loss at the end.

        # TODO: what if the sentences in a batch are all of different lengths?
        # - should be have the longest sentence as 2nd dim?
        #   - would zero endings work for other smaller sentences?

        y = torch.LongTensor(batch_size).type(torch.LongTensor)

        tensorized_inputs = []
        for i in xrange(len(batch_ques)):
            xq = Variable(self.make_input_matrix(batch_ques[i], word_vectors, vec_dim) ) #, requires_grad=False)
            xs = Variable(self.make_input_matrix(batch_sents[i], word_vectors, vec_dim) ) #, requires_grad=False)            
            ext_feats = Variable(torch.FloatTensor(batch_ext_feats[i]))
            ext_feats =torch.unsqueeze(ext_feats, 0)
            y[i] = batch_labels[i]
            tensorized_inputs.append((xq, xs, ext_feats))

        return tensorized_inputs, Variable(y)




