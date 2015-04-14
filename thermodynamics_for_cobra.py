import pandas as pd
import numpy as np
import sys, os
from cobra.io.sbml import create_cobra_model_from_sbml_file
from copy import deepcopy
from component_contribution.kegg_reaction import KeggReaction
from component_contribution.kegg_model import KeggModel
from component_contribution.component_contribution import ComponentContribution

from python.thermodynamic_constants import R, default_T

class THERMODYNAMICS_FOR_COBRA(object):

    def __init__(self, model):

        self.pH = 7.5
        self.I = 0.2
        self.T = default_T

        self.model = deepcopy(model)
        
        '''mapping file with model metabolites and kegg CIDs'''
        self.metabolites = pd.DataFrame.from_csv("data/metaboliteList.txt", sep='\t')
        self.metabolites.drop(['cas_id', 'Unnamed: 8'], axis=1, inplace=True)
#        self.metabolites.set_index('name', inplace=True)
        
    def metabolite2cid(self, metabolite_list):
        '''
            map COBRA metabolite objects to KEGG CIDs
            
            Arguments:
                List of metabolite objects
            Returns:
                Dictionary:
                    keys - metabolites
                    values - cids
        '''
        out = {}
        for m in metabolite_list:
            key = m.id[:-2] #remove '_c' from compound name
            try:
                cid = self.metabolites.kegg_id[key]
                out[m] = cid
            except:
                try:
                    cid = self.metabolites.kegg_id[key.replace('_', '-')]
                    out[m] = cid
                except:
                    out[m] = 'not mapped : %s' %m.id
        return out
        
    def reaction2string(self, reaction_list):
        '''
            map COBRA reactions to reactions strings with compound as KEGG cids
            For example: "2 C19610 + C00027 + 2 C00080 <=> 2 C19611 + 2 C00001"
            
            Arguments:
                List of model reaction ids
            Returns:
                Reaction strings
        '''

        sparse = {}
        reaction_strings = {}
        for r in reaction_list:
            cids = self.metabolite2cid(r.metabolites)
            sparse = {cids[m]:v for m,v in r.metabolites.iteritems()}
#            assert len(sparse) == len(r.metabolites), 'False reactants mapping!'
                
            # remove protons (H+) from reaction 
            #since not relevant for dG calculations
            if 'C00080' in sparse:
                del sparse['C00080']            
                
            assert r not in reaction_strings, 'Duplicate reaction!'

            try:
                kegg_reaction = KeggReaction(sparse)
                reaction_strings[r] = str(kegg_reaction)
            except TypeError: 
                continue
        return reaction_strings

    def reaction2dG0(self,reaction_list):

        cc = ComponentContribution.init()
        
        reaction_strings = self.reaction2string(reaction_list)            
        reactions = []
        reac_strings = []
        for r, string in reaction_strings.iteritems():
#            if 'not mapped' not in string:
            reactions.append(r)
            reac_strings.append(string)
            
        Kmodel = KeggModel.from_formulas(reac_strings)
        Kmodel.add_thermo(cc)
        dG0_prime, dG0_std = Kmodel.get_transformed_dG0(pH=7.5, I=0.2, T=298.15)
        dG0_prime = np.array(map(lambda x: x[0,0], dG0_prime))
        
        return dG0_prime, dG0_std
        
    def reaction2Keq(self,reaction_list):
        dG0_prime, dG0_std = self.reaction2dG0(reaction_list)
        return np.exp( -dG0_prime / (R*default_T) )
            
    def reaction2RI(self, reaction_list, fixed_conc=0.1):

        keq = self.reaction2Keq(reaction_list)

        N_P = np.array(map(lambda x: len(x.products), reaction_list))
        N_S = np.array(map(lambda x: len(x.reactants), reaction_list))        
        N = N_P + N_S
        Q_2prime = fixed_conc**(N_P-N_S)
        
        RI = ( keq*Q_2prime )**( 2.0/N )
        return RI
        
def test():
    model_fname = "data/iJO1366.xml"
    model = create_cobra_model_from_sbml_file(model_fname)
    TFC = THERMODYNAMICS_FOR_COBRA(model)
    reactions = map(model.reactions.get_by_id, ['MDH', 'FBA'])

    STR = TFC.reaction2string(reactions)
    dG0 = TFC.reaction2dG0(reactions)
    Keq = TFC.reaction2Keq(reactions)
    RI  = TFC.reaction2RI(reactions)
    return CID, STR, dG0, Keq, RI

CID, STR, dG0, Keq, RI = test()
#RI = {k.id:v for k, v in RI.iteritems()}

#import matplotlib.pyplot as plt
#plt.hist(np.log10(np.array(RI.values())))
