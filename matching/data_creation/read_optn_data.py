#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:18:28 2018

@author: vitorhadad
"""

import pandas as pd
import numpy as np
import pickle

# Path shenanigans
base_path = "/Users/vitorhadad/Documents/kidney/matching/matching/optn_data/"
main_labels_filename = "kidney_pancreas_data/KIDPAN_DATA.txt"
main_input_filename = "kidney_pancreas_data/KIDPAN_DATA.DAT"
donor_labels_filename = "living_donor/LIVING_DONOR_DATA.TXT"
donor_input_filename = "living_donor/LIVING_DONOR_DATA.DAT"
hla_labels_filename = "hla_additional/KIDPAN_ADDTL_HLA.TXT"
hla_input_filename = "hla_additional/KIDPAN_ADDTL_HLA.DAT"
antigen_input_filename = "unacceptable_antigen_data/KIDPAN_NON_STD.csv"

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
                                  "TRR_ID_CODE"])

cond = (df_main["WL_ORG"] == "KI") & \
      (df_main["PREV_TX"] != "N") & \
      (df_main["WL_ID_CODE"].notnull()) 
      
df_main = df_main.loc[cond, ["WL_ID_CODE", "DONOR_ID", "TRR_ID_CODE", "PT_CODE", "ABO", "ABO_DON"]]
df_main["WL_ID_CODE"] = df_main["WL_ID_CODE"].astype(int)

#%% Patient unacceptable antigen info
df_antigen = pd.read_csv(base_path + antigen_input_filename,
                         dtype={"wl_id_code":int},
                         date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d"))\
                         .drop(labels=["Unnamed: 0","wlreg_audit_id_code"], axis = 1)\
                         .drop_duplicates()

#%%
# Columns containing the unacceptable loci contain data of string type
# First, change those into sets
locus_cols = [c for c in df_antigen.columns if "locus" in c]

def str_to_set(x):
    if isinstance(x, str):
        return eval("{" + x + "}")
    else:
        return set()

df_antigen[locus_cols] = df_antigen[locus_cols].applymap(str_to_set)

# This information is still not unique per WL_ID_CODE.
# The next line makes it so.
df_antigen_loci = pd.concat([df_antigen.groupby("wl_id_code")[c].apply(lambda x: set.union(*x.values))
                         for c in locus_cols], axis = 1)

# Merge this information back with main dataset 
df_antigen_by_wl_code = pd.merge(df_main, df_antigen_loci, left_on = "WL_ID_CODE", right_index=True, how="inner")

# Last merge was one-to-many, so make it unique by patient by taking union over PT_CODE
df_antigen_by_pt_code = pd.concat([df_antigen_by_wl_code.groupby("PT_CODE")[c].apply(lambda x: set.union(*x.values)) 
                    for c in locus_cols], axis = 1)

# For ABO, take any one
df_patient_abo = df_antigen_by_wl_code.groupby("PT_CODE")["ABO"].last()

# Put these two together to make a first patient dataset
patient = pd.concat([df_antigen_by_pt_code, df_patient_abo], axis = 1)

      

#%%
# FIST DONOR DATA SET
# ---------------                  
                         
#%% Donor HLA info
hla_labels = pd.read_table(base_path+hla_labels_filename, sep = "\t")["LABEL"].values
df_hla = pd.read_table(base_path+hla_input_filename,
                       na_values = [".", "Unknown"],
                       names=hla_labels,
                       error_bad_lines=False,
                       encoding="latin1",
                       usecols=["TRR_ID_CODE",
                                "DA1","DA2",
                                "DB1","DB2",
                                #"DBW4","DBW6", No usable patient-side info
                                "DC1","DC2",
                                "DDP1","DDP2",
                                #"DDPA1","DDPA2", No usable patient-side info
                                "DDQ1","DDQ2",
                                "DDQA1","DDQA2",
                                "DDR1","DDR2",
                                "DDR51","DDR52","DDR53"])

#%% More living donor data
## (bizarrely absent from HLA code) 
donor_labels = pd.read_table(base_path+donor_labels_filename, sep = "\t")["LABEL"].values
df_donor = pd.read_table(base_path+donor_input_filename,
                           na_values = [".", "Unknown"],
                           names=donor_labels,
                           error_bad_lines=False,
                           encoding="latin1",
                           usecols=["ABO", "DONOR_ID"])

cond = df_donor["DONOR_ID"].notnull()
df_donor = df_donor.loc[cond]
df_donor["DONOR_ID"] = df_donor["DONOR_ID"].astype(int)


# Merge both partial datasets
donor_transplant = pd.merge(df_main[["DONOR_ID","TRR_ID_CODE"]], 
                            df_hla,
                            on = "TRR_ID_CODE", how="inner")
donor = pd.merge(donor_transplant,
                 df_donor,
                 on = "DONOR_ID",
                 how = "inner")

#%% Checkpoint
patient.to_csv("patient.csv")
donor.to_csv("donor.csv")
    
#%%   
# CLEARING UNUSED HLA CODES
# ---------------
pat_locus = {"A": "locusa", 
             "B": "locusb",
             "C": "locusc",
             "DQ": "locusdq",
             "DR": "locusdr", 
             "DPB": "locusdpb", 
             "DQA": "locusdqa"} 

don_locus = {"A": ["DA1", "DA2"], 
             "B": ["DB1", "DB2"],
             "C": ["DC1", "DC2"],
             "DQ": ["DDQ1", "DDQ2"],
             "DR": ["DDR1", "DDR2"], 
             "DPB": ["DDP1", "DDP2"], 
             "DQA": ["DDQA1", "DDQA2"]} 

# In the donor data set, many antigens are marked 97,98,99 or some other stupid escape code
# Let's figure out which are the valid antigen numbers
valid = {}
for locus in pat_locus:
    pat_colname = pat_locus[locus]
    don_colname1, don_colname2 = don_locus[locus]
    patient_values = set.union(*patient[pat_colname].values)
    donor_values = set(donor[don_colname1].unique()).union(donor[don_colname2].unique())
    valid[locus] = patient_values.intersection(donor_values) 

#%% Now update the donor dataset by sending stupid codes to np.nan
for k,dcols in don_locus.items():
    for dcol in dcols:
        donor[dcol] = donor[dcol].map({x:x for x in valid[k]})
    
        
        
        

#%% Second checkpoint
patient.to_csv("patient2.csv")
donor.to_csv("donor2.csv")

#%%   
# ONE-HOT ENCODING ALL VALUES
# ---------------
#patient = pd.read_csv("patient2.csv")

# Drop if blood type not simple ABO
patient = patient.loc[patient["ABO"].isin(["A","B","O","AB"])]

# For each column, we will first recode the numbers to enumeration
# E.g., 1,2,3,4,5,6,16,18 ---> 0,1,2,3,4,5,6,7


def recode(s, mapping):
    output = []
    for x in s:
        try:
            output.append(mapping[x])
        except KeyError:
            continue
    return output

dummies = []
for locus, plocus in pat_locus.items():
    locus_enum = {v:k for k,v in enumerate(valid[locus])}
    patient_recoded = patient[plocus].map(lambda s: recode(s, locus_enum))
    
    dummy = np.zeros((patient.shape[0], len(locus_enum)))    
    for i,js in enumerate(patient_recoded.values):
        dummy[i, js] = 1
    
    dummy = pd.DataFrame(dummy, columns = [locus + "_" + str(int(x)) for x in valid[locus]])
    dummies.append(dummy)
    
patient_locus_dummies = pd.concat(dummies, axis = 1)

#%%
# We'll drop anyone who is not simply A,B,O,AB
blood = pd.get_dummies(patient["ABO"])
blood.columns = ["blood_" + c for c in blood.columns]
patient_data = pd.concat([blood.reset_index(drop=True),
                          patient_locus_dummies.reset_index(drop=True)],
                axis = 1).astype(bool)

patient_data.to_csv("patient_data.csv")
      

#output_filename = "/Users/vitorhadad/Desktop/optn_data.h5"
#%%

# Donor features
# Same idea: drop ABO, recode, dummify
#donor = pd.read_csv("donor2.csv")


donor = donor.loc[donor["ABO"].isin(["A","B","O","AB"])]

don_locus = {"A": ["DA1", "DA2"], 
             "B": ["DB1", "DB2"],
             "C": ["DC1", "DC2"],
             "DQ": ["DDQ1", "DDQ2"],
             "DR": ["DDR1", "DDR2"], 
             "DPB": ["DDP1", "DDP2"], 
             "DQA": ["DDQA1", "DDQA2"]} 


donor_locus_data = []
for locus, dlocus  in don_locus.items():
    locus_enum = {v:k for k,v in enumerate(valid[locus])}
    dummy = np.zeros((donor.shape[0], len(locus_enum)))
    for dl in dlocus:
        for k,d in enumerate(donor[dl].map(locus_enum)):
            if not np.isnan(d):
                dummy[k,int(d)] = 1        
    dummy = pd.DataFrame(dummy, columns = [locus + "_" + str(int(x)) for x in valid[locus]])
    donor_locus_data.append(dummy)
    
    #%%
    
donor_locus_data = pd.concat(donor_locus_data, axis = 1).reset_index(drop=True)
donor_blood_data = pd.get_dummies(donor["ABO"], prefix = "blood").reset_index(drop=True)
donor_data = pd.concat([donor_locus_data, donor_blood_data], axis = 1).astype(bool)

donor_data.to_csv("donor_data.csv")

#%% Pickle for faster IO
pickle.dump(patient_data, open("matching/optn_data/patient.pkl", "wb"))
pickle.dump(donor_data, open("matching/optn_data/donor.pkl", "wb"))








