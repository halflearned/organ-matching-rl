#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:34:28 2018

@author: vitorhadad
"""

import numpy as np
import pandas as pd
from tqdm import trange
import pickle

patients = pickle.load(open("matching/optn_data/patient.pkl","rb"))
donors = pickle.load(open("matching/optn_data/donor.pkl","rb"))
#%%
tissue_cols = [c for c in donors.columns if "blood" not in c]
blood_cols = [c for c in donors.columns if "blood"  in c]

donor_cpra = []
for d in trange(len(donors.index)):
    donor_cpra.append((donors.loc[d, tissue_cols] & patients.loc[:,tissue_cols]).any(1).mean())
   
donor_cpra = np.array(donor_cpra)
pickle.dump(donor_cpra, open("matching/optn_data/donor_cpra.pkl", "wb"))


#%%
tissue_cols = [c for c in donors.columns if "blood" not in c]

patient_cpra = []
for d in trange(len(patients.index)):
    patient_cpra.append((patients.loc[d, tissue_cols] & donors.loc[:,tissue_cols]).any(1).mean())
    
patient_cpra = np.array(patient_cpra)
pickle.dump(patient_cpra, open("matching/optn_data/patient_cpra.pkl", "wb"))

#%% Put together
n = int(1e6)
ip_rnd = np.random.randint(len(patients), size = n)
id_rnd = np.random.randint(len(donors), size = n) 

donor_tissue = donors.loc[id_rnd, tissue_cols].values
donor_blood  = donors.loc[id_rnd, blood_cols].values
patient_tissue = patients.loc[ip_rnd, tissue_cols].values
patient_blood  = patients.loc[ip_rnd, blood_cols].values

tissue_incompatible = (donor_tissue & patient_tissue).any(1)

blood_incompatible = np.logical_not(np.any(patient_blood & donor_blood, 1) | \
                                  donors.loc[id_rnd, "blood_O"].values |  \
                                  patients.loc[ip_rnd, "blood_AB"].values)

incompatible = tissue_incompatible | blood_incompatible

i_p = ip_rnd[incompatible]
i_d = id_rnd[incompatible] 

data = pd.concat([patients.loc[i_p].rename(lambda s: s+"_pat", axis = "columns").reset_index(drop=True),
                  donors.loc[i_d].rename(lambda s: s+"_don", axis = "columns").reset_index(drop=True)],
                  axis = 1)

data["cpra_pat"] = patient_cpra[i_p]
data["cpra_don"] = donor_cpra[i_d]

pickle.dump(data, open("matching/optn_data/optn_pairs.pkl", "wb"))

