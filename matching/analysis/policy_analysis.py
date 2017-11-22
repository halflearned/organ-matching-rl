#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:07:06 2017

@author: vitorhadad
"""

import pickle
import numpy as np
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt

from matching.utils.data_utils import get_n_matched, flatten_matched
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet


from matching.utils.data_utils import get_dead

#

#for f in listdir("results/"):
#    if f.startswith("acc") and f.endswith("pkl"):
#        with open("results/" + f, "rb") as pkl:
#            res = pickle.load(pkl)
#            print(f, np.mean(res[-100:]))
#        
#%%

files = ["1227509700.pkl", "1249933113.pkl", "1251332146.pkl",
 "1392300224.pkl", "1400031610.pkl", "1404271352.pkl",
 "1456883791.pkl", "1701633712.pkl", "1737220349.pkl",
 "1896280098.pkl", "197775103.pkl", "2008169261.pkl",
 "2125269095.pkl", "2132463045.pkl", "2173870632.pkl",
 "224937377.pkl", "2287432928.pkl", "2317426309.pkl",
 "2402752358.pkl", "2706485922.pkl", "279822710.pkl",
 "3183676271.pkl", "3238084043.pkl", "3359947721.pkl",
 "3371311178.pkl", "3579632022.pkl", "3930005441.pkl",
 "4069171949.pkl", "4095039375.pkl", "4128822848.pkl",
 "4135425278.pkl", "4270208675.pkl", "4346503824.pkl",
 "4538566031.pkl", "4585125807.pkl", "4791042500.pkl",
 "4797010387.pkl", "485198510.pkl", "5013985615.pkl",
 "5327397497.pkl", "5375836688.pkl", "5411314183.pkl",
 "5472656265.pkl", "5496651267.pkl", "5605998765.pkl",
 "5764259822.pkl", "5981760915.pkl", "6124977828.pkl",
 "6300382061.pkl", "6419442605.pkl", "649243322.pkl", "6737876865.pkl", "6804235097.pkl", "6955442067.pkl", "7024244987.pkl", "7084938285.pkl", "7117177341.pkl", "7147502763.pkl", "716943830.pkl", "7345694730.pkl", "7379197968.pkl", "7456642288.pkl", "7639435505.pkl", "7684315084.pkl", "7866960019.pkl", "7989881574.pkl", "8266363213.pkl", "8334074398.pkl", "8610046317.pkl", "8626177240.pkl", "8917629327.pkl", "8955784677.pkl", "8981762681.pkl", "8987795486.pkl", "9026259202.pkl", "9171721092.pkl", "9196327239.pkl", "9204139920.pkl", "9254495667.pkl", "9313600988.pkl", "9428939241.pkl", "9514238077.pkl", "9635858307.pkl", "981017221.pkl", "9880975432.pkl"]

t_begin = 50
t_end = 200
T = 200

table = []

for f in files:
    with open("results/" + f, "rb") as file:
        res = pickle.load(file)
#        g = get_n_matched(res["greedy"],T)[t_begin:t_end].cumsum()
#        o = get_n_matched(res["opt"],T)[t_begin:t_end].cumsum()
#        t = get_n_matched(res["this"],T)[t_begin:t_end].cumsum()
##        print(g.sum(), t.sum(), o.sum(), end = " ")
##        if g.sum() < t.sum():
##            print("<----BETTER")
##        else:
##            print("")
#        ts = np.arange(1,201)
#        plt.figure()
#        plt.plot(ts, g/ts, color = "green"); 
#        plt.plot(ts, o/ts, color = "blue");
#        plt.plot(ts, t/ts, color = "orange")
#        plt.title(f, ":", )
        
        env = SaidmanKidneyExchange(5, .1, 200, seed = res["seed"])
    
        opt_matched    = flatten_matched(res["opt"],    t_begin, t_end)
        greedy_matched = flatten_matched(res["greedy"], t_begin, t_end)
        this_matched   = flatten_matched(res["this"],   t_begin, t_end)
        
        opt_dead = get_dead(env, opt_matched, t_begin, t_end)
        greedy_dead = get_dead(env, greedy_matched, t_begin, t_end)
        this_dead = get_dead(env, this_matched, t_begin, t_end)
        
        n = len(env.get_living(t_begin, t_end))
        opt_loss = len(opt_dead)/env.number_of_nodes()
        greedy_loss = len(greedy_dead)/env.number_of_nodes()
        this_loss = len(this_dead)/env.number_of_nodes()
        
        print("{:1.3f} {:1.3f} {:1.3f}: {} {} {} {} {} {} {} "\
              .format(opt_loss, greedy_loss, this_loss,
                      res["scl"], 
                      res["criterion"],
                      res["tpa"],
                      res["t_horiz"],
                      res["r_horiz"],
                      res["n_rolls"],
                      res["net_file"]),
              end = " ")
        if greedy_loss > this_loss:
            print("<--BETTER")
        else:
            print("")
            
        table.append([opt_loss, 
                      greedy_loss,
                      this_loss,
                      res["scl"], 
                      res["criterion"],
                      res["tpa"],
                      res["t_horiz"],
                      res["r_horiz"],
                      res["n_rolls"],
                      res["net_file"],
                      greedy_loss > this_loss])
    
    
#%% 
tab = pd.DataFrame(table, columns = ["opt_loss", "greedy_loss",
                              "this_loss", "scl",
                              "criterion", "tpa",
                              "t_horiz", "r_horiz",
                              "n_rolls", "net_file",
                              "better"])
