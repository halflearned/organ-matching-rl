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
                 num_layers=1):
        
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 
        self.seq_len = 1
        
        self.rnn = nn.GRU(input_size,
                           hidden_size,
                           num_layers,
                           bidirectional = True)

        self.states = []
        self.outputs = []
        
        
        
    def forward(self, inputs):    
        outputs, staten = self.rnn(inputs)
        outputs, lens = pad_packed_sequence(outputs, padding_value = 0.0)
        outputs = outputs[:, :, :self.hidden_size] +\
                  outputs[:, :, self.hidden_size:]  
        staten = staten[0:1,:,:] + staten[1:2,:,:]
        return outputs, staten
    

    
    
    

class AttentionDecoderRNN(nn.Module):
    
    def __init__(self, 
                 input_size,
                 hidden_size,
                 nprime = 50,
                 method = "dot"):
        
        super(AttentionDecoderRNN, self).__init__()
        
        self.method = method
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = 1
        
        
        self.gru = nn.GRU(input_size,
                           hidden_size,
                           num_layers)
        
        self.nprime = nprime
        
        self.W_init_state = nn.Parameter(
                torch.randn(hidden_size, hidden_size), 
                requires_grad = True)
        
        self.v = nn.Parameter(torch.randn(1, self.nprime),
                          requires_grad = True)
        
        self.W_blend = nn.Parameter(
                torch.randn(self.nprime, 2*hidden_size), 
                requires_grad = True)
        
        self.W_gen = nn.Parameter(
                torch.randn(hidden_size, hidden_size), 
                requires_grad = True)
        
        self.W_ctxt = nn.Parameter(
                torch.randn(hidden_size, 2*hidden_size),
                requires_grad = True)
        
        self.W_s = nn.Parameter(
                torch.randn(2, self.nprime),
                requires_grad = True)
        
        self.prob_layer = nn.Sequential(
                nn.Linear(2*hidden_size, 2),
                #nn.Softmax(dim = 2)
                )
        

    def forward(self, enc_outputs, last_enc_state, lengths):
        probs = []
        batch_size = enc_outputs.size(1)
        dec_output = Variable(torch.zeros(self.seq_len,
                                batch_size,
                                self.input_size))
        cur_state = last_enc_state
        for i in range(max(lengths)):
            dec_output, cur_state = self.gru(dec_output, cur_state)
            p, h_tilde = self.attention_step(enc_outputs,
                                       cur_state,
                                       lengths)
            #enc_outputs = torch.cat([enc_outputs, h_tilde], 0)
            probs.append(p)
            
        return torch.stack(probs, 1)



    def attention_step(self, enc_outputs, cur_state, lengths = None):
        weights = self.get_weights(enc_outputs, cur_state, lengths)
        
        context = weights.transpose(2,1) @ enc_outputs.transpose(0,1)
        
        sc = torch.cat([context, cur_state.transpose(0,1)], 2)
        htilde = self.W_ctxt @ sc.transpose(2,1)
        #probs = F.softmax(self.W_s @ htilde, dim = 1).squeeze()
        probs = self.prob_layer(sc).squeeze(1)
        return probs, htilde.transpose(1,2).transpose(0,1)
    
    


    def get_weights(self, enc_outputs, cur_state, lengths = None):
        if self.method == "concat":
            s = cur_state.repeat(1, enc_outputs.size(1), 1)
            sh = torch.cat([s, enc_outputs],2).transpose(2,1)
            align = self.v @ torch.tanh(self.W_blend @ sh)
            
        elif self.method == "dot":
            align = enc_outputs.transpose(1,0) @\
                    cur_state.transpose(0,1).transpose(1,2)
            
        elif self.method == "general":
            align = enc_outputs @ (self.W_gen @ cur_state.transpose(2,1))       
        else:
            raise ValueError("Invalid attention method")
    
        #import pdb; pdb.set_trace()
            
        if lengths is not None:
            for i,l in enumerate(lengths):
                try:
                    align[i,l:,0].data.fill_(-np.inf)
                except ValueError:
                    pass
            
        return F.softmax(align, dim = 1)



class AttentionRNN:
    
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers,
                 nprime = 50,
                 method = "dot",
                 path_enc = None,
                 path_dec = None,
                 learning_rate = 0.001):
        
        super(AttentionRNN, self).__init__()
        
 
        if path_enc is None:       
            self.rnn_enc = EncoderRNN(input_size,
                         hidden_size,
                         num_layers)
        else:
            self.rnn_enc = torch.load(path_enc)
        
        if path_dec is None:
            self.rnn_dec = AttentionDecoderRNN(hidden_size,
                         hidden_size,
                         num_layers,
                         method = method)
        else:
            self.rnn_dec = torch.load(path_dec)

        self.enc_optim = optim.Adam(self.rnn_enc.parameters(), lr=learning_rate)
        self.dec_optim = optim.Adam(self.rnn_dec.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduce = False,
                            weight = torch.FloatTensor([1,10]))
        
    
    def forward(self, inputs, lens = None):
        if lens is None:
            lens = np.argmax(inputs.any(2) == 0, 0)
            
        inputs = Variable(torch.FloatTensor(inputs),
                          requires_grad = False)
        order = np.flip(np.argsort(lens), 0).astype(int)
        order_r = torch.LongTensor(order[order])
        
        seq = pack_padded_sequence(inputs[:,order,:], lens[order])
        
        enc_outputs, last_enc_output = self.rnn_enc(seq)
        yhat = self.rnn_dec(enc_outputs, last_enc_output, lens[order])
        
        #import pdb; pdb.set_trace()
        return yhat[order_r] #TODO: Confirm this
     
    
    def run(self, inputs, true_outputs, lens = None):
        
        if lens is None:
            lens = inputs.any(2).sum(0)
        
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        
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
    
    
    def __str__(self):
        return "AttentionRNN_ENC{}-{}_DEC{}-{}"\
                .format(self.rnn_enc.hidden_size,
                        self.rnn_enc.num_layers,
                        self.rnn_dec.hidden_size,
                        self.rnn_dec.method)
                
#%%    
    
if __name__ == "__main__":
    
    from matching.utils.data_utils import open_file, confusion
        
    encoder_input_size = 294
    encoder_hidden_size = 100
    decoder_hidden_size = 100
    num_directions = 1
    num_layers = 1
    batch_size = 32
    
    open_every = 100
    log_every = 10
    save_every = 500
 
    net = AttentionRNN(input_size = 294,
                       hidden_size = 100,
                       num_layers = 1)

#%%
    for i in range(10000000):
        
        if i % open_every == 0:
            X, Y, GN = open_file(open_GN = True, open_A = False)
            SS = np.concatenate([X, GN], 2).transpose((1,0,2))
            n = SS.shape[1]
            num_ys = None
        
        idx = np.random.choice(n, size=batch_size, p=num_ys)
        inputs = SS[:,idx,:]
        ytrue = Y[idx]
        
        lens = inputs.any(2).sum(0)
        loss, yhat = net.run(inputs, ytrue, lens) 
            
        tp, tn, fp, fn = confusion(yhat, ytrue, lens)
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        acc = (tp + tn)/(tp+fp+tn+fn)
        msg = "{:1.4f},{:1.4f},{:1.4f},{:1.4f}"\
                .format(loss,
                    tpr, # True positive rate
                    tnr, # True negative rate
                    acc) # Accuracy
            
        print(msg)
        if i % log_every == 0:
            with open("results/" + str(net) + ".txt", "a") as f:
                print(msg, file = f)
        
#            if tpr > .8 and tnr < .5:
#                net.loss_fn = nn.CrossEntropyLoss(reduce = False,
#                            weight = torch.FloatTensor([1, 1]))
#            elif tpr < .5 and tnr > .8:
#                net.loss_fn = nn.CrossEntropyLoss(reduce = False,
#                            weight = torch.FloatTensor([1, 10]))
#            else:
#                net.loss_fn = nn.CrossEntropyLoss(reduce = False,
#                            weight = torch.FloatTensor([1, 5]))
#     
#            
         
        if i % save_every == 0:
            torch.save(net, "results/" + str(net))