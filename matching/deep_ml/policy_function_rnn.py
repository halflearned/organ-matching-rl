#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:28:15 2017


@author: vitorhadad
"""

import torch
from torch import nn, cuda
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers=1,
                 batch_size=32,
                 bidirectional = False):
        
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.seq_len = 1
        self.num_classes = 2
        
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           bidirectional = bidirectional)
        
        self.out_layer = nn.Sequential(
                            nn.Linear(hidden_size, self.num_classes)
                            )
        
        
        self.c0 = nn.Parameter(torch.randn(num_layers * self.num_directions, 
                              1,
                              hidden_size), requires_grad = True)
        
        self.h0 = nn.Parameter(torch.randn(num_layers * self.num_directions, 
                              1,
                              hidden_size), requires_grad = True)
        
        
        self.probs = []
        
        
 
        self.optim = optim.Adam(rnn.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss(reduce = False,
                        weight = torch.FloatTensor([1, 5]))
        
  
        
    def forward(self, inputs):    
        #if isinstance(seq, torch.nn.utils.rnn.PackedSequence):    
        this_batch_size = inputs.batch_sizes[0]
        initial_state = (self.c0.repeat(1, this_batch_size, 1), 
                         self.h0.repeat(1, this_batch_size, 1))
        outputs, staten = self.rnn(inputs, initial_state)
        outputs, lens = pad_packed_sequence(outputs) 
        outputs = outputs[:, :, :self.hidden_size] +\
                  outputs[:, :, self.hidden_size:]  
    
        return outputs, staten


    def run(self, inputs, true_outputs, lens = None):
        
        if lens is None:
            lens = inputs.any(2).sum(0)
        
        self.optim.zero_grad()
        
        yhat = self.forward(inputs, lens)
        ytruth = Variable(torch.LongTensor(true_outputs), requires_grad = False)
        
        loss = 0
        for i,l in enumerate(lens):
            l = lens[i]
            yh = yhat[i,:l]
            yt = ytruth[i,:l].view(-1)        
            #assert yt.data.numpy().sum() % 2 == 0
            loss += self.loss_fn(yh, yt).mean()        
        loss /= batch_size
    
        loss.backward()
        self.enc_optim.step()
        self.dec_optim.step()
    
        return loss.data.numpy()[0], yhat
    


#%%    
    
if __name__ == "__main__":
    
    from matching.utils.data_utils import open_file
    

    #%%
    encoder_input_size = 294
    encoder_hidden_size = 100
    decoder_hidden_size = 100
    bidirectional = True
    num_layers = 1
    batch_size = 32
    rnn = torch.load("results/policy_function_lstm")
    
    batch_size = 32
    open_every = 100
    log_every = 10
    save_every = 500
    
#%%
    for i_iter in range(1000):
        
        if i % open_ == 0:
            X, Y, GN = open_file(open_GN = True, open_A = False)
            SS = np.concatenate([X, GN], 2).transpose((1,0,2))
            n = SS.shape[1]
            num_ys = None
        
        optimizer.zero_grad()
        
        idx = np.random.choice(n, size=batch_size)
        subset = SS[:,idx,:]
        lens = np.argmax(subset.any(2) == 0, 0)
       
        inputs = Variable(torch.FloatTensor(subset),
                          requires_grad = False)
        
        order = np.flip(np.argsort(lens), 0).astype(int)
        order_t = torch.LongTensor(order)
        
        seq = pack_padded_sequence(inputs[:,order,:], 
                                   torch.LongTensor(lens[order]).tolist())
        truth = Variable(torch.LongTensor(Y[idx][order]), requires_grad = False)
 
        outputs, staten = rnn.forward(seq)
        probs = []
        loss = 0
        for i in range(batch_size):
            l = lens[order[i]]
            p = rnn.out_layer(outputs[:l,i])
            y = truth[i,:l,0]
            assert y.data.numpy().sum() % 2 == 0
            probs.append(p)
            loss += loss_fn(p, y).mean()
        loss /= batch_size

        
        compare.append(np.hstack([F.softmax(p)[:,1].data.numpy().reshape(-1,1),
           y.data.numpy().reshape(-1,1)]).round(2))
        
        print("Loss:", loss.data.numpy()[0], 
              " Avg prob:", F.softmax(p)[:,1].mean().data.numpy()[0],
              " Std prob:", F.softmax(p)[:,1].std().data.numpy()[0])
        
        loss.backward()
        optimizer.step()
        

        if i_iter % 100 == 0:
            pass #torch.save(rnn, "results/policy_function_lstm")
        
    #%%
    compare = np.vstack(compare)
    pos = compare[:,1] == 1
    neg = compare[:,1] == 0
    true_pos = (compare[:,0] == 1) & pos
    true_neg = (compare[:,0] == 0) & neg
    false_pos = (compare[:,0] == 1) & neg
    false_neg = (compare[:,0] == 0) & pos
    
    tp_ratio = np.sum(true_pos)/np.sum(pos)
    fp_ratio = np.sum(false_pos)/np.sum(pos)
    tn_ratio = np.sum(true_neg)/np.sum(neg)
    fn_ratio = np.sum(false_neg)/np.sum(neg)
    