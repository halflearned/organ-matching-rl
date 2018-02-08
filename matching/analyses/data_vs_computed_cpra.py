#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:43:21 2018

@author: halflearned
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
sns.set_style("white")

# Path shenanigans
base_path = "/Users/vitorhadad/Documents/kidney/data/"
main_labels_filename = "kidney_pancreas_data/KIDPAN_DATA.txt"
main_input_filename = "kidney_pancreas_data/KIDPAN_DATA.DAT"

#%%
#  FIRST PATIENT DATA SET
# ---------------
#
# Relevant columns from main kidney panel dataset
main_labels = pd.read_table(base_path+main_labels_filename, sep = "\t")["LABEL"].values
df_main = pd.read_table(base_path+main_input_filename,
                       na_values = [".", "Unknown"],
                       names=main_labels,
                       date_parser=lambda x: pd.to_datetime(x, format="%m/%d/%Y"),
                       error_bad_lines=False,
                       encoding="latin1",
                       usecols = [#"A1", "A2",
                                  #"B1", "B2",
                                  #"DR1", "DR2",
                                  "ABO","ABO_DON",
                                  "WL_ORG", "PT_CODE",
                                  "PREV_TX",
                                  "WL_ID_CODE",
                                  "DONOR_ID",
                                  "TRR_ID_CODE",
                                  "INIT_CPRA", "END_CPRA"])

cond = (df_main["WL_ORG"] == "KI") & \
      (df_main["PREV_TX"] != "N") & \
      (df_main["WL_ID_CODE"].notnull()) 
      
df = df_main.loc[cond]

#%%
fig, ax = plt.subplots(1, 2, figsize=(12,4), sharey=True)
df["INIT_CPRA"].hist(ax=ax[0], bins=50, normed=True)
df["END_CPRA"].hist(ax=ax[1], bins=50, normed=True)
ax[0].set_title("Init cPRA")
ax[1].set_title("End cPRA")
fig.suptitle("Waiting list patient cPRA")
fig.savefig("../../figures/star_cpra.pdf")

#%%
#df = pickle.load(open("matching/optn_data/optn_pairs.pkl", "rb"))

fig, ax = plt.subplots(1, 2, figsize=(12,4), sharey=True, sharex=True)
df["cpra_pat"].hist(ax=ax[0], bins=50, normed=True)
df["cpra_don"].hist(ax=ax[1], bins=50, normed=True)
ax[0].set_title("Patient cPRA")
ax[1].set_title("Donor cPRA")
for a in ax:
    a.set_xlim([-.01,1.01])
    a.set_xticks([0,.2,.4,.6,.8,1.0])
    a.set_xticklabels([0,.2,.4,.6,.8,1.], fontsize=14)
#fig.suptitle("Computed cPRA")
fig.savefig("../../figures/computed_cpra.pdf")
















