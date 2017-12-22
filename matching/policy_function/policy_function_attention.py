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
        batch_size = enc_outputs.size(1)
        dec_output = Variable(torch.zeros(self.seq_len,
                                batch_size,
                                self.input_size))
        cur_state = last_enc_state
        htildes = []
        for i in range(max(lengths)):
            dec_output, cur_state = self.gru(dec_output, cur_state)
            htilde = self.attention_step(enc_outputs,
                                       cur_state,
                                       lengths)
            htildes.append(htilde)
            
        return torch.cat(htildes).transpose(1,0)



    def attention_step(self, enc_outputs, cur_state, lengths = None):
        weights = self.get_weights(enc_outputs, cur_state, lengths)
        
        context = weights.transpose(2,1) @ enc_outputs.transpose(0,1)
        
        sc = torch.cat([context, cur_state.transpose(0,1)], 2)
        htilde = self.W_ctxt @ sc.transpose(2,1)
        #probs = F.softmax(self.W_s @ htilde, dim = 1).squeeze()
        #probs = self.prob_layer(sc).squeeze(1)
        return htilde.transpose(1,2).transpose(0,1)
    
    


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


        self.logit_layer = nn.Linear(hidden_size, 2)
        self.count_layer = nn.Linear(hidden_size, 1)

        self.enc_optim = optim.Adam(self.rnn_enc.parameters(), lr=learning_rate)
        self.dec_optim = optim.Adam(self.rnn_dec.parameters(), lr=learning_rate)
        self.logit_loss_fn = nn.CrossEntropyLoss(reduce = False,
                            weight = torch.FloatTensor([1,10]))
        self.count_loss_fn = nn.MSELoss()
        
    
    def forward(self, inputs, lens = None):
        if lens is None:
            lens = np.argmax(inputs.any(2) == 0, 0)
            
        inputs = Variable(torch.FloatTensor(inputs),
                          requires_grad = False)
        order = np.flip(np.argsort(lens), 0).astype(int)
        order_r = torch.LongTensor(order[order])
        
        seq = pack_padded_sequence(inputs[:,order,:], lens[order])
        
        enc_outputs, last_enc_output = self.rnn_enc(seq)
        prelogits_r = self.rnn_dec(enc_outputs, last_enc_output, lens[order])
        prelogits = prelogits_r[order_r]
        
        count = self.count_layer(prelogits.sum(1))
        logits = self.logit_layer(prelogits)
        return  logits, count
     
    
    def run(self, inputs, true_outputs, lens = None):
        
        if lens is None:
            lens = inputs.any(2).sum(0)
        
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        
        ylogits,ycount = self.forward(inputs, lens)
        ytruth = Variable(torch.LongTensor(true_outputs), requires_grad = False)
    
        logit_loss = 0
        for i,l in enumerate(lens):
            l = lens[i]
            yh = ylogits[i,:l]
            yt = ytruth[i,:l].view(-1)  
            try:
                logit_loss += self.logit_loss_fn(yh, yt).mean() 
            except RuntimeError as e:
                print(e)
        logit_loss /= batch_size
        
        count_loss = self.count_loss_fn(ycount, ytruth.sum(1).float())
    
        loss = logit_loss + count_loss

        loss.backward()
        self.dec_optim.step()
        self.enc_optim.step()

        return (logit_loss.data.numpy()[0], 
                count_loss.data.numpy()[0], 
                ylogits, ycount.data.numpy())
    
    
    def __str__(self):
        return "AttentionRNN_ENC{}-{}_DEC{}-{}"\
                .format(self.rnn_enc.hidden_size,
                        self.rnn_enc.num_layers,
                        self.rnn_dec.hidden_size,
                        self.rnn_dec.method)
                
#%%    
    
if __name__ == "__main__":
    
    from matching.utils.data_utils import open_file, confusion
    from sys import platform, argv
    
    if platform == "darwin":
        argv.extend(["optn", 3, 100, .5, np.random.randint(1e8)])
        
    
    encoder_input_size = {"abo":24, "optn":294}
    env_type = argv[1]
    num_layers = int(argv[2])
    hidden_size = int(argv[3])
    c = float(argv[4])
    s = str(argv[5])
    
    batch_size = 32
    open_every = 10
    save_every = 500
    log_every = 10

    
      
    net = AttentionRNN(input_size = encoder_input_size[env_type],
                       hidden_size = hidden_size,
                       num_layers = num_layers)
  
    name = "{}-{}_{}".format(
            str(net),
            env_type,
            s)
#%%
    for i in range(10000000):
        
        if i % open_every == 0:
            X, Y, GN = open_file(env_type = env_type, open_GN = True, open_A = False)
            SS = np.concatenate([X, GN], 2).transpose((1,0,2))
            n = SS.shape[1]
        
        idx = np.random.choice(n, size=batch_size)
        inputs = SS[:,idx,:]
        ytrue = Y[idx]
        
        lens = inputs.any(2).sum(0)
        
        avg_ones = np.hstack([Y[k,:l,0] for k,l in zip(idx, lens)]).mean()
        if avg_ones > 0:
            w = c*1/avg_ones
        
        net.logit_loss_fn = nn.CrossEntropyLoss(reduce = False,
                        weight = torch.FloatTensor([1, w]))
        

        lloss,closs, ylogits, ycount = net.run(inputs, ytrue, lens) 
            
        cacc = np.mean(ycount.round() == ytrue.sum(1))
        tp, tn, fp, fn = confusion(ylogits, ytrue, lens)
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        lacc = (tp + tn)/(tp+fp+tn+fn)
        
        if tpr < .1:
            c *= 1.05
        if tnr < .1:
            c *= .95
        
        msg = "{:1.4f},{:1.4f},{:1.4f},"\
                "{:1.4f},{:1.4f},{:1.4f},{:1.4f}"\
                .format(lloss,
                        closs,
                    tpr, # True positive rate
                    tnr, # True negative rate
                    lacc, # Logits accuracy
                    cacc, # Count accuracy
                    w) 
        if i % log_every == 0:
            print(msg)
            if platform == "linux":
                with open("results/" + name + ".txt", "a") as f:
                    print(msg, file = f)
        
        if platform == "linux" and i % save_every == 0:
            torch.save(net, "results/" + name)
