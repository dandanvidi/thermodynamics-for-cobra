import pandas as pd
import sys, os
from cobra.io.sbml import create_cobra_model_from_sbml_file
from copy import deepcopy

sys.path.append(os.path.expanduser('~/git/component-contribution'))
from python.kegg_reaction import KeggReaction
from python.kegg_model import KeggModel
from python.component_contribution import ComponentContribution
from python.thermodynamic_constants import R, default_T

class Reversibility(object):

    def __init__(self, model):

        self.pH = 7.5
        self.I = 0.2
        self.T = default_T

        self.model = deepcopy(model)
        
        '''map model metabolites to kegg CIDs'''
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
            try:
                key = m.id[:-2] #remove '_c' from compound name
                cid = self.metabolites.kegg_id[key]
                out[m] = cid
            except:
                out[m] = 'None'
        return out
        
    def reaction2string(self, reaction_list):
        '''
            map COBRA reactions to reactions strings with compound as KEGG cids
            For example: "2 C19610 + C00027 + 2 C00080 <=> 2 C19611 + 2 C00001"
            
            Arguments:
                List of model reaction ids
            Returns:
                Dictionary:
                    keys - reaction ids
                    values - reaction strings
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
        
        CC_CACHE_FNAME = os.path.expanduser('../component-contribution/cache/component_contribution.mat')
        cc = ComponentContribution.from_matfile(CC_CACHE_FNAME)

        reaction_strings = self.reaction2string(reaction_list)            
        reactions = []
        reac_strings = []
        for r, string in reaction_strings.iteritems():
            if 'None' not in string:
                reactions.append(r)
                reac_strings.append(string)
            
        Kmodel = KeggModel.from_formulas(reac_strings)
        
        Kmodel.add_thermo(cc)
        dG0_prime, dG0_std = Kmodel.get_transformed_dG0(pH=7.5, I=0.2, T=298.15)
        return dG0_prime, dG0_std
        
model_fname = "data/iJO1366.xml"
model = create_cobra_model_from_sbml_file(model_fname)
REV = Reversibility(model)
reactions = model.reactions[1100:1102]
#print REV.metabolite2cid(model.metabolites[5:10])
#print REV.reaction2string(reactions)
print REV.reaction2dG0(reactions)
#def get_reaction_strings(reactions):
#for key, val in reaction_strings.iteritems():    
#    reactions.append(key)
#    reac_strings.append(val)
#    
