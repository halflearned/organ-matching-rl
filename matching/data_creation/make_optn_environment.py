#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:41:59 2018

@author: vitorhadad
"""

import numpy as np
import pandas as pd
from tqdm import trange
import pickle

from matching.environment.base_environment import BaseKidneyExchange


class OPTNKidneyExchange(BaseKidneyExchange):
    
    
    patients = pickle.load(open("matching/optn_data/patient.pkl","rb"))
    donors = pickle.load(open("matching/optn_data/donor.pkl","rb"))
    donor_cpra = pickle.load(open("matching/optn_data/donor_cpra.pkl", "rb"))
    patient_cpra = pickle.load(open("matching/optn_data/patient_cpra.pkl", "rb"))

    
    def __init__(self, 
             entry_rate,
             death_rate,
             time_length,
             initial_size = None,
             seed=None,
             populate=True):
        
        self.initial_size = initial_size
    
        super(self.__class__, self)\
                  .__init__(entry_rate=entry_rate,
                    death_rate=death_rate,
                    time_length=time_length,
                    seed=seed,
                    populate=False)

        