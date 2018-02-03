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


from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_lstm import RNN

def to_var(x, requires_grad = True):
    return Variable(torch.FloatTensor(np.array(x, dtype = np.float32)),
                    requires_grad = requires_grad)


class RGCNet(nn.Module):
    
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout_prob = 0.4):
        
        super(RGCNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 
        self.seq_len = 1
        
        self.logit_loss_fn = nn.CrossEntropyLoss(reduce = True,
                        weight = torch.FloatTensor([1, 10]))
        self.count_loss_fn = nn.CrossEntropyLoss(reduce = True,
                        weight = torch.FloatTensor([1, 2]))
        
        
        self.gcn = GCNet(input_size,
                      [hidden_size]*num_layers,
                      dropout_prob = dropout_prob)
        self.rnn = RNN(input_size = hidden_size,
                       hidden_size = hidden_size,
                       num_layers = 1)
        self.logit_layer = nn.Linear(hidden_size, 2)
        self.count_layer = nn.Linear(hidden_size, 2)
        
        self.optim = torch.optim.Adam(self.parameters())
        
        
    def forward(self, A, X, lens = None):
        
        if lens is None:
            lens = X.any(2).sum(1)
            
        A, X = self.gcn.make_variables(A, X)
        
        h = X
        for gcnlayer in self.gcn.model:
            h = gcnlayer(A @ h)
        
        ht = h.transpose(1, 0)
        logits, counts = self.rnn.forward(ht, lens)
        return logits, counts
        
    
    
    
    def run(self, A, X, y, lengths):
        if lengths is None:
            lengths = X.any(2).sum(1)
            
        batch_size = X.shape[0]
        ylogits, ycount = self.forward(A, X)

        ytruth = to_var(y, False).long()
        ctruth = to_var(y.any(1).flatten(), False).long()

        logit_loss = 0
        for i,l in enumerate(lens):
            l = lens[i]
            yh = ylogits[i,:l]
            yt = ytruth[i,:l].view(-1)  
            logit_loss += self.logit_loss_fn(yh, yt).mean() 
                
        logit_loss /= batch_size
            
        count_loss = self.count_loss_fn(ycount, ctruth).mean() 
    
        loss = logit_loss + count_loss
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
        
        return (logit_loss.data.numpy()[0], 
                count_loss.data.numpy()[0], 
                ylogits, ycount.data.numpy())

        
    
        
    def __str__(self):
        return "RGCNet-{}-{}"\
                .format(self.hidden_size,
                        self.num_layers)


#%%    
    
if __name__ == "__main__":

    from sys import argv, platform
    from matching.utils.data_utils import open_file, confusion, confusion1d

    batch_size = 32
    open_every = 10
    log_every = 10
    save_every = 500
 
    if platform == "darwin":
        argv = [None, "abo", "3", "50", "True", np.random.randint(1e8)]

    env_type = argv[1]
    hidden = int(argv[3])
    num_layers = int(argv[2])
    use_gn = bool(argv[4])
    s = str(argv[5])
    
    input_size = {"abo":10, "optn":280}
        
    net = RGCNet(input_size[env_type] + 14*use_gn, 
                 hidden,
                 num_layers)
    
    name = "{}-{}_{}".format(
            str(net),
            env_type,
            s)
    c = .5
    c2 = .2


    for i in range(int(1e8)):
        if i % open_every == 0:
            print("new file!")
            if use_gn:
                A, X, GN, Y = open_file(env_type = env_type,
                                        open_A = True,
                                        open_GN = True)
                Z = np.concatenate([X, GN], 2)
            else:
                A, X, Y = open_file(env_type = env_type,
                                    open_A = True,
                                    open_GN = False)
                Z = X
            
        n = A.shape[0]
        idx = np.random.choice(n, size = batch_size)   
        inputs = (A[idx], Z[idx])
        ytrue = Y[idx]
        lens = inputs[1].any(2).sum(1)
        
        avg_ones = np.hstack([Y[k,:l,0] for k,l in zip(idx, lens)]).mean()
        if avg_ones > 0:
            w = c*1/avg_ones
        
        net.logit_loss_fn = nn.CrossEntropyLoss(reduce = True,
                        weight = torch.FloatTensor([1, w]))
        
        lloss,closs, ylogits, ycount = net.run(*inputs, ytrue, lens)
        ctrue = ytrue.any(1)

        ltp, ltn, lfp, lfn = confusion(ylogits, ytrue, lens)
        ctp, ctn, cfp, cfn = confusion1d(ycount, ytrue.any(1).flatten())
        ltpr = ltp/(ltp+lfn)
        ltnr = ltn/(ltn+lfp)
        lacc = (ltp + ltn)/(ltp+lfp+ltn+lfn)
        
        ctpr = ctp/(ctp+cfn)
        ctnr = ctn/(ctn+cfp)
        cacc = (ctp + ctn)/(ctp+cfp+ctn+cfn)
        if ltpr < .1 and ltnr > .5:
            c = np.minimum(c*1.001, 5)
        elif ltnr < .1 and ltpr > .5:
            c = np.maximum(.999*c, .1)
            
        if ctpr < .1 and ctnr > .5:
            c2 = np.minimum(c2*1.01, 5)
        elif ctnr < .1 and ctpr > .5:
            c2 = np.maximum(.99*c2, .1)         
#            
        msg = ",".join(["{:1.4f}"]*10)\
                .format(lloss,
                        closs,
                    ltpr, # True positive rate
                    ltnr, # True negative rate
                    lacc, # Logits accuracy
                    ctpr,
                    ctnr,
                    cacc, # Count accuracy
                    c, 
                    c2) 
                
        if i % log_every == 0:
            print(msg)
            if platform == "linux":
                with open("results/" + name + ".txt", "a") as f:
                    print(msg, file = f)
        
        if platform == "linux" and i % save_every == 0:
            torch.save(net, "results/" + name)
