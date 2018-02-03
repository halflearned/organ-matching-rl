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



class EncoderRNN(nn.Module):
    
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers=1,
                 batch_size=32,
                 bidirectional = False):
        
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.seq_len = 1
        
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           bidirectional = bidirectional)
        
        self.c0 = Variable(torch.randn(num_layers * num_directions, 
                              batch_size,
                              hidden_size), requires_grad = True)
        
        self.h0 = Variable(torch.randn(num_layers * num_directions, 
                              batch_size,
                              hidden_size), requires_grad = True)
        
        self.states = []
        self.outputs = []
        
        
        
    def forward(self, inputs):    
        initial_state = (self.c0, self.h0)
        outputs, staten = self.rnn(inputs, initial_state)
        outputs, lens = pad_packed_sequence(outputs) 
        outputs = outputs[:, :, :self.hidden_size] +\
                  outputs[:, :, self.hidden_size:]  
        return outputs, staten
    
    
    


class DecoderRNN(nn.Module):
    
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers=1,
                 batch_size=32,
                 bidirectional = False):
        
        super(DecoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.seq_len = 1
        self.num_classes = 2
        
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers)
        
        self.out_layer = nn.Sequential(
                            nn.Linear(hidden_size, self.num_classes)
                            )
        
        self.start_token = Variable(torch.randn(self.seq_len,
                                                batch_size,
                                                input_size))
        
        
        self.probs = []
        
        
        
    def forward(self, input_state, lengths):
        self.probs.clear()
        
        state = input_state
        output = self.start_token
        for b in range(self.batch_size):
            l = lengths[i]
            output, state = self.rnn(output[b:b+1,:,:], state)
            p = self.out_layer(output.transpose(0,1))
            p
            self.probs.append(prob)
            
        return torch.cat(self.probs, 1)
    
    

    


    
    



#%%    
    
if __name__ == "__main__":
    
    from sys import platform
    from itertools import cycle
    from random import choice
    from collections import deque
    
#    SS = np.load("data/optn_SS.npy")
#    AA = np.load("data/optn_AA.npy")
#    YY = np.load("data/optn_YY.npy").astype(int)
#    lengths = np.load("data/optn_lengths.npy")
#    SS = SS.transpose((1,0,2))
    
    
        #%%
    n = SS.shape[1]
    encoder_input_size = 294
    encoder_hidden_size = 100
    decoder_hidden_size = 100
    num_directions = 1
    num_layers = 1
    batch_size = 32

    #%%
    rnn_enc = EncoderRNN(encoder_input_size,
                     encoder_hidden_size,
                     num_layers)
    
    rnn_dec = DecoderRNN(encoder_hidden_size,
                     decoder_hidden_size,
                     num_layers)
    
    learning_rate = 0.001
    encoder_optimizer = optim.Adam(rnn_enc.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(rnn_dec.parameters(), lr=learning_rate)
    
#%%
    for _ in range(5000):
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        idx = np.random.randint(n, size = batch_size)
        inputs = Variable(torch.FloatTensor(SS[:,idx,:]),
                                requires_grad = False)
        
        lens = lengths[idx] 
        maxlen = max(lengths[idx])
        order = np.flip(np.argsort(lens), 0).astype(int)
        order_t = torch.LongTensor(order)
        
        seq = pack_padded_sequence(inputs[:,order,:], lens[order])
        truth = Variable(torch.LongTensor(YY[idx][order]), requires_grad = False)

        
        outs, enc_state = rnn_enc.forward(seq)
        dsad
        probs = rnn_dec.forward(enc_state, maxlen)
    
        loss_fn = nn.CrossEntropyLoss(reduce = False,
                            weight = torch.FloatTensor([1,10]))
        
        loss = 0
        for i,l in enumerate(lengths[idx]):
            loss += loss_fn(probs[i,:l], truth[i,:l,0]).mean()
        loss /= batch_size
        
        print("Loss:", loss.data.numpy()[0], 
                  " Avg prob:", F.sigmoid(probs[i,:l,1]).mean().data.numpy()[0])
            
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        
    