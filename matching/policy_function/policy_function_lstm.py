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
                 bidirectional = False,
                 class_weights = [1, 10]):
        
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.seq_len = 1
        self.num_classes = 2
        self.class_weights = class_weights
        self.loss_fn = nn.CrossEntropyLoss(reduce = False,
                        weight = torch.FloatTensor(self.class_weights))
        
        self.count_loss_fn = nn.MSELoss()
        
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           bidirectional = bidirectional)
                    
        self.logit_layer = nn.Linear(hidden_size, self.num_classes)
        
        self.count_layer = nn.Linear(hidden_size, 1)
                            
        
        self.c0 = nn.Parameter(torch.randn(num_layers * self.num_directions, 
                              1,
                              hidden_size), requires_grad = True)
        
        self.h0 = nn.Parameter(torch.randn(num_layers * self.num_directions, 
                              1,
                              hidden_size), requires_grad = True)
        

        
 
        self.optim = optim.Adam(self.parameters(), lr=0.001)
        
        
    def forward(self, inputs, lens = None):    
        
        if lens is None:
            lens = inputs.any(2).sum(0)
        
        inputs = Variable(torch.FloatTensor(inputs),
                          requires_grad = False)
        
        order = np.flip(np.argsort(lens), 0).astype(int)
        order_r = torch.LongTensor(order[order])
        
        seq = pack_padded_sequence(inputs[:,order,:], lens[order])
        this_batch_size = seq.batch_sizes[0]
        initial_state = (self.c0.repeat(1, this_batch_size, 1), 
                         self.h0.repeat(1, this_batch_size, 1))
        
        outputs, staten = self.rnn(seq, initial_state)
        outputs, lens = pad_packed_sequence(outputs) 
        outputs = outputs[:, :, :self.hidden_size] +\
                  outputs[:, :, self.hidden_size:]  
    
        prelogits = outputs[:,order_r,:].transpose(1,0)
        
        logits = self.logit_layer(prelogits)
        
        precounts = prelogits.sum(1).squeeze()
        
        counts = self.count_layer(precounts)
        return logits, counts
        
        
        



    def run(self, inputs, true_outputs, lens = None):
        
        if lens is None:
            lens = inputs.any(2).sum(0)
        
        ylogits, ycount = self.forward(inputs, lens)
        ytruth = Variable(torch.LongTensor(true_outputs), requires_grad = False)
        
        logit_loss = 0
        for i,l in enumerate(lens):
            l = lens[i]
            yh = ylogits[i,:l]
            yt = ytruth[i,:l].view(-1)  
            try:
                logit_loss += self.loss_fn(yh, yt).mean() 
            except RuntimeError as e:
                print(e)
        logit_loss /= batch_size
        
        count_loss = self.count_loss_fn(ycount, ytruth.sum(1).float())
    
        loss = logit_loss + count_loss
    
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
        
        return (logit_loss.data.numpy()[0], 
                count_loss.data.numpy()[0], 
                ylogits, ycount.data.numpy())
    

    def __str__(self):
        return "RNN_{}-{}"\
                .format(self.hidden_size,
                        self.num_layers)


#%%    
    
if __name__ == "__main__":
    
    from matching.utils.data_utils import open_file, confusion
    from sys import argv, platform
    
    if platform == "darwin":
        argv.extend(["abo", 1, 100, .5, np.random.randint(1e8)])
    
    
    #if len(argv) > 1:
    print("Creating new RNN")
    env_type = argv[1]
    num_layers = int(argv[2])
    hidden_size = int(argv[3])
    c = float(argv[4])
    s = str(argv[5])
    input_size = {"abo":24, "optn":294}
    net = RNN(input_size=input_size[env_type],
          hidden_size=hidden_size,
          num_layers=num_layers,
          bidirectional=True,
          class_weights = [1,100*c])
    
    batch_size = 32
    open_every = 10
    save_every = 500
    log_every = 10

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
            
        net.loss_fn = nn.CrossEntropyLoss(reduce = False,
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

                
