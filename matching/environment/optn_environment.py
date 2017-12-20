#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:17:27 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from matching.environment.base_environment import BaseKidneyExchange
#%%
class OPTNKidneyExchange(BaseKidneyExchange):
    
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

        
        self.unacceptable_antigen_cols = ['notA1','notA2','notA3','notA9','notA10','notA11','notA19','notA23',
             'notA24','notA25','notA26','notA28','notA29','notA30','notA31','notA32',
             'notA33','notA34','notA36','notA43','notA66','notA68','notA69','notA74',
             'notA80','notA203','notA210','notA2403','notA6601','notA6602','notB5',
             'notB7','notB8','notB12','notB13','notB14','notB15','notB16','notB17',
             'notB18','notB21','notB22','notB27','notB35','notB37','notB38','notB39',
             'notB40','notB41','notB42','notB44','notB45','notB46','notB47','notB48',
             'notB49','notB50','notB51','notB52','notB53','notB54','notB55','notB56',
             'notB57','notB58','notB59','notB60','notB61','notB62','notB63','notB64',
             'notB65','notB67','notB70','notB71','notB72','notB73','notB75','notB76',
             'notB77','notB78','notB81','notB82','notB703','notB2708','notB3901',
             'notB3902','notB3905','notB4005','notB5102','notB5103','notB8201','notC1',
             'notC2','notC3','notC4','notC5','notC6','notC7','notC8','notC9','notC10',
             'notC11','notC12','notC13','notC14','notC15','notC16','notC17','notC18',
             'notDR1','notDR2','notDR3','notDR4','notDR5','notDR6','notDR7','notDR8',
             'notDR9','notDR10','notDR11','notDR12','notDR13','notDR14','notDR15',
             'notDR16','notDR17','notDR18','notDR103','notDR1403','notDR1404']
                    
        self.donor_antigen_cols = ['donor_A1','donor_A2','donor_A3','donor_A9','donor_A10','donor_A11',
             'donor_A19','donor_A23','donor_A24','donor_A25','donor_A26','donor_A28',
             'donor_A29','donor_A30','donor_A31','donor_A32','donor_A33','donor_A34',
             'donor_A36','donor_A43','donor_A66','donor_A68','donor_A69','donor_A74',
             'donor_A80','donor_A203','donor_A210','donor_A2403','donor_A6601',
             'donor_A6602','donor_B5','donor_B7','donor_B8','donor_B12','donor_B13',
             'donor_B14','donor_B15','donor_B16','donor_B17','donor_B18','donor_B21',
             'donor_B22','donor_B27','donor_B35','donor_B37','donor_B38','donor_B39',
             'donor_B40','donor_B41','donor_B42','donor_B44','donor_B45','donor_B46',
             'donor_B47','donor_B48','donor_B49','donor_B50','donor_B51','donor_B52',
             'donor_B53','donor_B54','donor_B55','donor_B56','donor_B57','donor_B58',
             'donor_B59','donor_B60','donor_B61','donor_B62','donor_B63','donor_B64',
             'donor_B65','donor_B67','donor_B70','donor_B71','donor_B72','donor_B73',
             'donor_B75','donor_B76','donor_B77','donor_B78','donor_B81','donor_B82',
             'donor_B703','donor_B2708','donor_B3901','donor_B3902','donor_B3905',
             'donor_B4005','donor_B5102','donor_B5103','donor_B8201','donor_C1',
             'donor_C2','donor_C3','donor_C4','donor_C5','donor_C6','donor_C7',
             'donor_C8','donor_C9','donor_C10','donor_C11','donor_C12','donor_C13',
             'donor_C14','donor_C15','donor_C16','donor_C17','donor_C18','donor_DR1',
             'donor_DR2','donor_DR3','donor_DR4','donor_DR5','donor_DR6','donor_DR7',
             'donor_DR8','donor_DR9','donor_DR10','donor_DR11','donor_DR12',
             'donor_DR13','donor_DR14','donor_DR15','donor_DR16','donor_DR17',
             'donor_DR18','donor_DR103','donor_DR1403','donor_DR1404']
                
                
        self.patient_blood_cols = ['patient_blood_A', 'patient_blood_O', 'patient_blood_B']
        self.donor_blood_cols = ['donor_blood_A', 'donor_blood_B', 'donor_blood_AB']
        
        self.antigen_column_pairs = [('notA1', 'donor_A1'), ('notA2', 'donor_A2'), ('notA3', 'donor_A3'), ('notA9', 'donor_A9'), ('notA10', 'donor_A10'), ('notA11', 'donor_A11'), ('notA19', 'donor_A19'), ('notA23', 'donor_A23'), ('notA24', 'donor_A24'), ('notA25', 'donor_A25'), ('notA26', 'donor_A26'), ('notA28', 'donor_A28'), ('notA29', 'donor_A29'), ('notA30', 'donor_A30'), ('notA31', 'donor_A31'), ('notA32', 'donor_A32'), ('notA33', 'donor_A33'), ('notA34', 'donor_A34'), ('notA36', 'donor_A36'), ('notA43', 'donor_A43'), ('notA66', 'donor_A66'), ('notA68', 'donor_A68'), ('notA69', 'donor_A69'), ('notA74', 'donor_A74'), ('notA80', 'donor_A80'), ('notA203', 'donor_A203'), ('notA210', 'donor_A210'), ('notA2403', 'donor_A2403'), ('notA6601', 'donor_A6601'), ('notA6602', 'donor_A6602'), ('notB5', 'donor_B5'), ('notB7', 'donor_B7'), ('notB8', 'donor_B8'), ('notB12', 'donor_B12'), ('notB13', 'donor_B13'), ('notB14', 'donor_B14'), ('notB15', 'donor_B15'), ('notB16', 'donor_B16'), ('notB17', 'donor_B17'), ('notB18', 'donor_B18'), ('notB21', 'donor_B21'), ('notB22', 'donor_B22'), ('notB27', 'donor_B27'), ('notB35', 'donor_B35'), ('notB37', 'donor_B37'), ('notB38', 'donor_B38'), ('notB39', 'donor_B39'), ('notB40', 'donor_B40'), ('notB41', 'donor_B41'), ('notB42', 'donor_B42'), ('notB44', 'donor_B44'), ('notB45', 'donor_B45'), ('notB46', 'donor_B46'), ('notB47', 'donor_B47'), ('notB48', 'donor_B48'), ('notB49', 'donor_B49'), ('notB50', 'donor_B50'), ('notB51', 'donor_B51'), ('notB52', 'donor_B52'), ('notB53', 'donor_B53'), ('notB54', 'donor_B54'), ('notB55', 'donor_B55'), ('notB56', 'donor_B56'), ('notB57', 'donor_B57'), ('notB58', 'donor_B58'), ('notB59', 'donor_B59'), ('notB60', 'donor_B60'), ('notB61', 'donor_B61'), ('notB62', 'donor_B62'), ('notB63', 'donor_B63'), ('notB64', 'donor_B64'), ('notB65', 'donor_B65'), ('notB67', 'donor_B67'), ('notB70', 'donor_B70'), ('notB71', 'donor_B71'), ('notB72', 'donor_B72'), ('notB73', 'donor_B73'), ('notB75', 'donor_B75'), ('notB76', 'donor_B76'), ('notB77', 'donor_B77'), ('notB78', 'donor_B78'), ('notB81', 'donor_B81'), ('notB82', 'donor_B82'), ('notB703', 'donor_B703'), ('notB2708', 'donor_B2708'), ('notB3901', 'donor_B3901'), ('notB3902', 'donor_B3902'), ('notB3905', 'donor_B3905'), ('notB4005', 'donor_B4005'), ('notB5102', 'donor_B5102'), ('notB5103', 'donor_B5103'), ('notB8201', 'donor_B8201'), ('notC1', 'donor_C1'), ('notC2', 'donor_C2'), ('notC3', 'donor_C3'), ('notC4', 'donor_C4'), ('notC5', 'donor_C5'), ('notC6', 'donor_C6'), ('notC7', 'donor_C7'), ('notC8', 'donor_C8'), ('notC9', 'donor_C9'), ('notC10', 'donor_C10'), ('notC11', 'donor_C11'), ('notC12', 'donor_C12'), ('notC13', 'donor_C13'), ('notC14', 'donor_C14'), ('notC15', 'donor_C15'), ('notC16', 'donor_C16'), ('notC17', 'donor_C17'), ('notC18', 'donor_C18'), ('notDR1', 'donor_DR1'), ('notDR2', 'donor_DR2'), ('notDR3', 'donor_DR3'), ('notDR4', 'donor_DR4'), ('notDR5', 'donor_DR5'), ('notDR6', 'donor_DR6'), ('notDR7', 'donor_DR7'), ('notDR8', 'donor_DR8'), ('notDR9', 'donor_DR9'), ('notDR10', 'donor_DR10'), ('notDR11', 'donor_DR11'), ('notDR12', 'donor_DR12'), ('notDR13', 'donor_DR13'), ('notDR14', 'donor_DR14'), ('notDR15', 'donor_DR15'), ('notDR16', 'donor_DR16'), ('notDR17', 'donor_DR17'), ('notDR18', 'donor_DR18'), ('notDR103', 'donor_DR103'), ('notDR1403', 'donor_DR1403'), ('notDR1404', 'donor_DR1404')]
        
        self.X_attributes = ['death', 'donor_A1', 'donor_A10', 'donor_A11', 'donor_A19', 'donor_A2', 'donor_A203', 'donor_A210', 'donor_A23', 'donor_A24', 'donor_A2403', 'donor_A25', 'donor_A26', 'donor_A28', 'donor_A29', 'donor_A3', 'donor_A30', 'donor_A31', 'donor_A32', 'donor_A33', 'donor_A34', 'donor_A36', 'donor_A43', 'donor_A66', 'donor_A6601', 'donor_A6602', 'donor_A68', 'donor_A69', 'donor_A74', 'donor_A80', 'donor_A9', 'donor_B12', 'donor_B13', 'donor_B14', 'donor_B15', 'donor_B16', 'donor_B17', 'donor_B18', 'donor_B21', 'donor_B22', 'donor_B27', 'donor_B2708', 'donor_B35', 'donor_B37', 'donor_B38', 'donor_B39', 'donor_B3901', 'donor_B3902', 'donor_B3905', 'donor_B40', 'donor_B4005', 'donor_B41', 'donor_B42', 'donor_B44', 'donor_B45', 'donor_B46', 'donor_B47', 'donor_B48', 'donor_B49', 'donor_B5', 'donor_B50', 'donor_B51', 'donor_B5102', 'donor_B5103', 'donor_B52', 'donor_B53', 'donor_B54', 'donor_B55', 'donor_B56', 'donor_B57', 'donor_B58', 'donor_B59', 'donor_B60', 'donor_B61', 'donor_B62', 'donor_B63', 'donor_B64', 'donor_B65', 'donor_B67', 'donor_B7', 'donor_B70', 'donor_B703', 'donor_B71', 'donor_B72', 'donor_B73', 'donor_B75', 'donor_B76', 'donor_B77', 'donor_B78', 'donor_B8', 'donor_B81', 'donor_B82', 'donor_B8201', 'donor_C1', 'donor_C10', 'donor_C100', 'donor_C11', 'donor_C12', 'donor_C13', 'donor_C14', 'donor_C15', 'donor_C16', 'donor_C17', 'donor_C18', 'donor_C2', 'donor_C3', 'donor_C4', 'donor_C5', 'donor_C6', 'donor_C7', 'donor_C8', 'donor_C9', 'donor_DR1', 'donor_DR10', 'donor_DR103', 'donor_DR11', 'donor_DR12', 'donor_DR13', 'donor_DR14', 'donor_DR1403', 'donor_DR1404', 'donor_DR15', 'donor_DR16', 'donor_DR17', 'donor_DR18', 'donor_DR2', 'donor_DR3', 'donor_DR4', 'donor_DR5', 'donor_DR6', 'donor_DR7', 'donor_DR8', 'donor_DR9', 'donor_blood_A', 'donor_blood_B', 'donor_blood_O', 'entry', 'notA1', 'notA10', 'notA11', 'notA19', 'notA2', 'notA203', 'notA210', 'notA23', 'notA24', 'notA2403', 'notA25', 'notA26', 'notA28', 'notA29', 'notA3', 'notA30', 'notA31', 'notA32', 'notA33', 'notA34', 'notA36', 'notA43', 'notA66', 'notA6601', 'notA6602', 'notA68', 'notA69', 'notA74', 'notA80', 'notA9', 'notB12', 'notB13', 'notB1304', 'notB14', 'notB15', 'notB16', 'notB17', 'notB18', 'notB21', 'notB22', 'notB25', 'notB27', 'notB2708', 'notB35', 'notB37', 'notB38', 'notB39', 'notB3901', 'notB3902', 'notB3905', 'notB40', 'notB4005', 'notB41', 'notB42', 'notB44', 'notB45', 'notB46', 'notB47', 'notB48', 'notB49', 'notB5', 'notB50', 'notB51', 'notB5102', 'notB5103', 'notB52', 'notB53', 'notB54', 'notB55', 'notB56', 'notB57', 'notB58', 'notB59', 'notB60', 'notB61', 'notB62', 'notB63', 'notB64', 'notB65', 'notB67', 'notB7', 'notB70', 'notB703', 'notB71', 'notB72', 'notB73', 'notB75', 'notB76', 'notB77', 'notB78', 'notB7801', 'notB8', 'notB804', 'notB81', 'notB82', 'notB8201', 'notC1', 'notC10', 'notC11', 'notC12', 'notC13', 'notC14', 'notC15', 'notC16', 'notC17', 'notC18', 'notC2', 'notC3', 'notC4', 'notC5', 'notC6', 'notC7', 'notC8', 'notC9', 'notDR1', 'notDR10', 'notDR103', 'notDR11', 'notDR12', 'notDR13', 'notDR14', 'notDR1403', 'notDR1404', 'notDR15', 'notDR16', 'notDR17', 'notDR18', 'notDR2', 'notDR3', 'notDR4', 'notDR5', 'notDR51', 'notDR52', 'notDR53', 'notDR6', 'notDR7', 'notDR8', 'notDR9', 'patient_blood_A', 'patient_blood_B', 'patient_blood_O']
        self.data = None
        
        if populate: self.populate(seed = seed)
        
        

    def __class___(self):
        return OPTNKidneyExchange
    
    

    def draw_node_features(self, t_begin, t_end):
        
        if self.data is None:
            self.data = pd.read_hdf("matching/environment/optn_data.5", "data")
        
        if t_begin == 0:
            np.random.seed(self.seed)
        
        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size = duration)
        if self.initial_size is not None:
            n_periods[0] = self.initial_size
            
        n = np.sum(n_periods)
        entries = np.repeat(np.arange(t_begin, t_end), n_periods)
        
        sojourns = np.random.geometric(self.death_rate, n) - 1
        deaths = entries + sojourns
        
        rnd_idx = np.random.randint(self.data.shape[0],
                                    size = n)
        out = self.data.loc[rnd_idx]
        
        with pd.option_context("chained_assignment", None):
            out["entry"] = entries
            out["death"] = deaths
        
        out.reset_index(drop = True, inplace = True)
       
        result= out.apply(lambda x: dict(zip(out.columns,
                                            x.values)),
                                axis = 1).tolist()
        

        return result
        
    
    def get_time_compatible(self, source_nodes, target_nodes):
        s_entry = self.attr("entry", nodes = source_nodes)
        s_death = self.attr("death", nodes = source_nodes)
        t_entry = self.attr("entry", nodes = target_nodes)
        t_death = self.attr("death", nodes = target_nodes)
            
        time_comp = (s_entry <= t_death.T) & \
               (s_death >= t_entry.T)
               
        idx = np.argwhere(time_comp)
                    
        return set(((source_nodes[i], target_nodes[j]) for i,j in idx))
    
    
    
    def get_blood_compatible(self, source_nodes, target_nodes):
        s_blood = self.attr("donor_blood_O", "donor_blood_A", "donor_blood_B", nodes = source_nodes)
        t_blood = self.attr("patient_blood_AB", "patient_blood_A", "patient_blood_B", nodes = target_nodes)
        
        blood_comp = (s_blood[:,:1] | t_blood[:,:1].T) | \
                     (s_blood[:,1:] @ t_blood[:,1:].T)
                     
        idx = np.argwhere(blood_comp)
        
        return set(((source_nodes[i], target_nodes[j]) for i,j in idx))
    
    
    def get_histo_compatible(self, source_nodes, target_nodes):
        s_antigen = self.attr(*self.donor_antigen_cols, nodes = source_nodes)
        t_antigen = self.attr(*self.unacceptable_antigen_cols, nodes = target_nodes)
        
        histo_comp = np.logical_not(s_antigen @ t_antigen.T)
                     
        idx = np.argwhere(histo_comp)
        
        return set(((source_nodes[i], target_nodes[j]) for i,j in idx))
    
    



    def draw_edges(self, source_nodes, target_nodes):
       
        time_comp = self.get_time_compatible(source_nodes, target_nodes)
        
        blood_comp = self.get_blood_compatible(source_nodes, target_nodes)
                
        hist_comp = self.get_histo_compatible(source_nodes, target_nodes)
                
        comp = time_comp.intersection(blood_comp).intersection(hist_comp)
        
        return ((i,j) for (i,j) in comp if i != j)
    

            
    def X(self, t, graph_attributes = True, dtype = "numpy"):
        alive = self.get_living(t)
        np.random.seed(t)
        np.random.shuffle(alive)
        X = self.attr(*self.X_attributes, nodes = alive)
        d_idx = self.X_attributes.index("death")
        e_idx = self.X_attributes.index("entry")
        X[:,d_idx] = X[:,d_idx] - t
        X[:,e_idx] = t - X[:,e_idx]
        columns = self.X_attributes.copy()
        if graph_attributes:
            entry_rate = np.full((len(alive),1), 
                                 fill_value = self.entry_rate)
            death_rate = np.full((len(alive),1), 
                                 fill_value = self.death_rate)
            columns += ["entry_rate", "death_rate"]
            X = np.hstack((X, entry_rate, death_rate))
            
        if dtype == "pandas":
            return pd.DataFrame(index = alive,
                                 data= X,
                                 columns = columns)
        elif dtype == "numpy":
            return X
        elif dtype == "sparse":
            return csr_matrix(X)
        else:
            raise ValueError("Invalid dtype")

    
    
    
    
    
#%%    
if __name__ == "__main__":
    
    from matching.solver.kidney_solver import KidneySolver
    import matplotlib.pyplot as plt
   
    T = 20
    env = OPTNKidneyExchange(5, .1, T, populate=True)
