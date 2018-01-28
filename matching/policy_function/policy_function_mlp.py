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


def all_sum(x):
    return x.sum()

use_cuda = cuda.is_available()

def to_var(x, requires_grad = True):
    return Variable(torch.FloatTensor(np.array(x, dtype = np.float32)),
                    requires_grad = requires_grad)


class MLPNet(nn.Module):
    
    def __init__(self, 
                 feature_size,
                 hidden_size,
                 num_layers,
                 dropout_prob = 0.2,
                 activation_fn = nn.SELU,
                 output_fn = nn.Sigmoid,
                 opt = optim.Adam,
                 opt_params = dict(lr = 0.001),
                 loss_fn = nn.MSELoss,
                 seed = None):
    
        
        if seed : torch.manual_seed(seed)
        
        super(MLPNet, self).__init__()
        
        self.feature_size  = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = 2
        
        self.activation_fn = activation_fn
        self.dropout_prob = dropout_prob
        self.output_fn = output_fn()
        
        
        self.logit_loss_fn = nn.CrossEntropyLoss(reduce = True,
                        weight = torch.FloatTensor([1, 10]))
        self.count_loss_fn = nn.CrossEntropyLoss(reduce = False,
                        weight = torch.FloatTensor([1, 4]))
                
                
        self.model = self.build_model()
        
        self.logit_layer = nn.Linear(hidden_size, 2)
        self.count_layer = nn.Linear(hidden_size, 2)

    
        self.optim=opt(self.parameters(), **opt_params)
  
        
    def forward(self, XX):
        lens =  XX.any(2).sum(1)
        
        XX = to_var(XX, requires_grad=False)
        h = self.model(XX)
        logits = self.logit_layer(h)
        counts =         logits = self.logit_layer(h)
        counts = self.count_layer(
                    torch.stack([h[i,:l].mean(0)
                    for i,l in enumerate(lens)])
        )
        return logits, counts
    


    def build_model(self):
        
        layer = lambda inp, out: \
                nn.Sequential(
                    nn.Linear(inp, out),
                    self.activation_fn(),
                    nn.AlphaDropout(self.dropout_prob))
        
        sizes = [self.feature_size] + [self.hidden_size]*self.num_layers
        mlp = [layer(h0,h1) for h0,h1 in zip(sizes[:-1], sizes[1:])]
        return nn.Sequential(*mlp)
   
        
            
    
    
    def run(self, inputs, true_outputs, lens = None):
        
        batch_size = inputs.shape[0]
        if lens is None:
            lens = inputs.any(2).sum(1)
        ytruth = Variable(torch.LongTensor(true_outputs), requires_grad = False)
        
        
        ylogits, ycount = self.forward(inputs)
        ctruth = to_var(true_outputs.any(1).flatten(), False).long()
        
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
            
        count_loss = self.count_loss_fn(ycount, ctruth).mean() 
    
        loss = logit_loss + count_loss
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
        
        return (logit_loss.data.numpy()[0], 
                count_loss.data.numpy()[0], 
                ylogits, ycount.data.numpy())
    
        
    
    
    def __str__(self):
        return "MLP-{}-{}-{}"\
            .format(self.hidden_size,
                    self.num_layers,
                    int(100*self.dropout_prob))
    
    
    
    

#%%    
    
if __name__ == "__main__":
    
    from sys import argv, platform
    from matching.utils.data_utils import open_file, confusion, confusion1d

    batch_size = 32
    open_every = 10
    log_every = 10
    save_every = 500
 
    if platform == "darwin":
        argv = [None, "abo", "4", "100", "True", np.random.randint(1e8)]

    env_type = argv[1]
    hidden = int(argv[3])
    num_layers = int(argv[2])
    use_gn = bool(argv[4])
    s = str(argv[5])
    
    input_size = {"abo":10, "optn":280}
        
    net = MLPNet(input_size[env_type] + 14*use_gn, 
                hidden,
                num_layers,
                dropout_prob = .2) 
    
    name = "{}-{}_{}".format(
            str(net),
            env_type,
            s)
    c = .5
    c2 = .25
    
    #%%
    for i in range(int(1e8)):
        if i % open_every == 0:
            if use_gn:
                X, Y, GN = open_file(env_type = env_type,
                                        open_A = False,
                                        open_GN = True)
                Z = np.concatenate([X, GN], 2)
            else:
                X, Y = open_file(env_type = env_type,
                                    open_A = False,
                                    open_GN = False)
                Z = X
            
        n = X.shape[0]
        idx = np.random.choice(n, size = batch_size)   
        inputs = Z[idx]
        ytrue = Y[idx]
        lens = inputs.any(2).sum(1)
        
        avg_ones = np.hstack([Y[k,:l,0] for k,l in zip(idx, lens)]).mean()
        if avg_ones > 0:
            w = c*1/avg_ones
        
        net.logit_loss_fn = nn.CrossEntropyLoss(reduce = False,
                        weight = torch.FloatTensor([1, w]))
        net.count_loss_fn = nn.CrossEntropyLoss(reduce = False,
                        weight = torch.FloatTensor([1, c2*1/avg_ones]))
        

        lloss,closs, ylogits, ycount = net.run(inputs, ytrue, lens)
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
