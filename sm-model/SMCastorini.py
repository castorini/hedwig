import torch
from torch.autograd import Variable
import numpy as np
import os
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import *
from collections import defaultdict
import pickle
import string

from model import QAModel


class SMModelCastorini(object):

    def __init__(self, model_file, word_embeddings_cache_file, stopwords_file, word2dfs_file):
        # init torch random seeds
        torch.manual_seed(1234)
        np.random.seed(1234)        

        # load model
        self.model = QAModel.load('', model_file)
        # load vectors
        self.vec_dim = self._preload_cached_embeddings(word_embeddings_cache_file)
        self.unk_term_vec = np.random.uniform(-0.25, 0.25, self.vec_dim) 

        # stopwords
        self.stoplist = set([line.strip() for line in open(stopwords_file)])

        # word dfs
        if os.path.isfile(word2dfs_file):
		    self.word2dfs = pickle.load( open( "word2dfs.p", "rb" ) )


    def _preload_cached_embeddings(self, cache_file):
        
        with open(cache_file + '.dimensions') as d:
            vocab_size, vec_dim = [int(e) for e in d.read().strip().split()]

        self.W = np.memmap(cache_file, dtype=np.double, shape=(vocab_size, vec_dim))

        with open(cache_file + '.vocab') as f:            
            w2v_vocab_list = map(str.strip, f.readlines())

        self.vocab_dict = {w:k for k,w in enumerate(w2v_vocab_list)}
        return vec_dim


    def parser(self, q, a):
    	q_toks = TreebankWordTokenizer().tokenize(q)
        q_str = ' '.join(q_toks).lower()
        a_list = []
        for ans in a:
            ans_toks = TreebankWordTokenizer().tokenize(ans)
            a_str = ' '.join(ans_toks).lower()
            a_list.append(a_str)
        return q_str, a_list


    def compute_overlap_features(self, q_str, a_list, word2df=None, stoplist=None):
    	word2df = word2df if word2df else {}
        stoplist = stoplist if stoplist else set()
        feats_overlap = []
        for a in a_list:
            question = q_str.split()
            answer = a.split()            
            # q_set = set(question)
            # a_set = set(answer)
            q_set = set([q for q in question if q not in stoplist])
            a_set = set([a for a in answer if a not in stoplist])
            word_overlap = q_set.intersection(a_set)
            # overlap = float(len(word_overlap)) / (len(q_set) * len(a_set) + 1e-8)
            if len(q_set) == 0 and len(a_set) == 0:
                overlap = 0
            else:
                overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))

            # q_set = set([q for q in question if q not in stoplist])
            # a_set = set([a for a in answer if a not in stoplist])
            word_overlap = q_set.intersection(a_set)
            df_overlap = 0.0
            for w in word_overlap:
                df_overlap += word2df[w]
            
            if len(q_set) == 0 and len(a_set) == 0:
                df_overlap = 0
            else:
                df_overlap /= (len(q_set) + len(a_set))

            feats_overlap.append(np.array([
                                overlap,
                                df_overlap,
                                ]))
        return np.array(feats_overlap)


    def make_input_matrix(self,  sentence):
        terms = sentence.strip().split()                
        # word_embeddings = torch.zeros(max_len, vec_dim).type(torch.DoubleTensor)
        word_embeddings = torch.zeros(len(terms), self.vec_dim).type(torch.DoubleTensor)
        for i in range(len(terms)):
            word = terms[i]        
            if word not in self.vocab_dict:
                emb = torch.from_numpy(self.unk_term_vec)
            else:
                emb = torch.from_numpy(self.W[self.vocab_dict[word]])                        
            word_embeddings[i] = emb            

        input_tensor = torch.zeros(1, self.vec_dim, len(terms))
        input_tensor[0] = torch.transpose(word_embeddings, 0 , 1)
        return input_tensor


    def get_tensorized_inputs(self, batch_ques, batch_sents, batch_ext_feats):
        
        assert(1 == len(batch_ques) )
        
        tensorized_inputs = []
        for i in range(len(batch_ques)):
            xq = Variable(self.make_input_matrix(batch_ques[i]) ) 
            xs = Variable(self.make_input_matrix(batch_sents[i]) )
            ext_feats = Variable(torch.FloatTensor(batch_ext_feats[i]))
            ext_feats =torch.unsqueeze(ext_feats, 0)            
            tensorized_inputs.append((xq, xs, ext_feats))

        return tensorized_inputs


    def rerank_candidate_answers(self, question, answers):
        # tokenize
        q_str, a_list = self.parser(question, answers)

        # calculate overlap features
        overlap_feats = self.compute_overlap_features(q_str,a_list, stoplist=None, word2df=self.word2dfs)        
        overlap_feats_stoplist = self.compute_overlap_features(q_str, a_list, stoplist=self.stoplist, word2df=self.word2dfs)
        overlap_feats_vec = np.hstack([overlap_feats, overlap_feats_stoplist])

        # run through the model
        scores_sentences = []
        for i in range(len(a_list)):
            xq, xa, x_ext_feats = self.get_tensorized_inputs([q], [a_list[i]], [overlap_feats_vec[i]])[0]          
            pred = self.model(xq, xa, x_ext_feats)               
            pred = torch.exp(pred)
            scores_sentences.append( (pred.data.squeeze()[1], a_list[i]) )
        
        return scores_sentences


if __name__ == "__main__":
    

    smmodel = SMModelCastorini('../../data/TrecQA/sm.model.aquaint.castorini', 
                    '../../data/word2vec-models/aquaint+wiki.txt.gz.ndim=50.cache',
                    'stopwords.txt',
                    'word2dfs.p')
    
    
    q = "who is the author of the book , `` the iron lady : a biography of margaret thatcher '' ?"
    a = [ 
    "the iron lady ; a biography of margaret thatcher by hugo young -lrb- farrar , straus & giroux -rrb-",
    "in this same revisionist mold , hugo young , the distinguished british journalist , has performed a brilliant dissection of the notion of thatcher as a conservative icon .",
    "in `` the iron lady , '' young traces the winding staircase of fortune that transformed the younger daughter of a provincial english grocer into the greatest woman political leader since catherine the great .",
    "`` he is the very essence of the classless meritocrat , '' says hugo young , thatcher 's biographer .",
    "from her father , young argues , she inherited a `` joyless earnestness '' that combined with her early interest in science to produce the roots of her public character .",
    "this is not the answer",
    "asdfawe asdf sertse dgfsgsfg"
    ]
	
    ss = smmodel.rerank_candidate_answers(q, a)
    print('Question:', q)
    for score, sentence in ss:
        print(score, '\t', sentence)