'''

Requirements:

1 . uncertainties package - http://packages.python.org/uncertainties/index.html

'''
import warnings
import pandas as pd
import numpy as np
import uncertainties.unumpy as unumpy  
from cobra.io.sbml import create_cobra_model_from_sbml_file
from copy import deepcopy
from component_contribution.kegg_reaction import KeggReaction
from component_contribution.kegg_model import KeggModel
from component_contribution.component_contribution import ComponentContribution
from component_contribution.thermodynamic_constants import R, default_T

class THERMODYNAMICS_FOR_COBRA(object):

    def __init__(self, model, reaction_list):

        self.pH = 7.5
        self.I = 0.2
        self.T = default_T
        
        self.model = deepcopy(model)        
        self.reactions = reaction_list
        
        '''mapping file with model metabolites and kegg CIDs'''
        self.metabolites = pd.DataFrame.from_csv("data/metaboliteList.txt", sep='\t')
        self.metabolites.drop(['cas_id', 'Unnamed: 8'], axis=1, inplace=True)
#        self.metabolites.set_index('name', inplace=True)
        
    def _metabolite2cid(self, metabolite_list):
        '''
            map COBRA metabolite to KEGG CIDs
            
            Arguments:
                List of metabolite ids
            Returns:
                List of KEGG cids
        '''
        CIDS = []
        for m in metabolite_list:
            try:
                CIDS.append(self.metabolites.kegg_id[m[:-2].replace('_', '-')])
            except KeyError:
                warnings.warn("%s can not be mapped to kegg" %m)
                CIDS.append(None)
        return CIDS
        
    def _reaction2string(self):
        '''
            map COBRA reactions to reactions strings with KEGG cids, e.g.,
            "2 C19610 + C00027 + 2 C00080 <=> 2 C19611 + 2 C00001"
            
            Arguments:
                List of model reaction ids
            Returns:
                Reaction strings
        '''
        reaction_list = map(self.model.reactions.get_by_id, self.reactions)        
        reaction_strings = []
        reaction_sparses = []        
        indices = set()        
        
        for i, r in enumerate(reaction_list):

            metabolites = map(lambda x: x.id, r.metabolites)
            CIDS = self._metabolite2cid(metabolites)
    
            # if not all reactants could be mapped to kegg CIDS, 
            #return empty reaction string
            if None in CIDS:
                indices.add(i)
                warnings.warn("%s was removed from reaction_list. " %r.id +
                                "see class reactions for updated list" )
                continue
            
            reactant2cid = dict(zip(metabolites, CIDS))            
            sparse = {reactant2cid[m.id]:v for m,v in r.metabolites.iteritems()}
                
            # remove protons (H+) from reactions
            if 'C00080' in sparse:
                del sparse['C00080']            
                            
            assert r not in reaction_strings, 'Duplicate reaction!'

            try:
                kegg_reaction = KeggReaction(sparse)
                if kegg_reaction.is_balanced():
                    reaction_strings.append(str(KeggReaction(sparse)))
                    reaction_sparses.append(sparse)
                else:
                    indices.add(i)
            except TypeError: 
                warnings.warn("%s not be converted to reaction string" %str(sparse))

        for i in sorted(indices, reverse=True):
            del self.reactions[i]

        return reaction_sparses, reaction_strings

    def reaction2dG0(self):
        '''
            Calculates the dG0 of a list of a reaction.
            Uses the component-contribution package (Noor et al) to estimate
            the standard Gibbs Free Energy of reactions based on 
            component contribution  approach and measured values (NIST and Alberty)
            
            Arguments:
                List of cobra model reaction ids
            Returns:
                Array of dG0 values and standard deviation of estimates
        '''
        cc = ComponentContribution.init()
        
        reaction_sparses, reaction_strings = self._reaction2string()
        Kmodel = KeggModel.from_formulas(reaction_strings)
        Kmodel.add_thermo(cc)
        dG0_prime, dG0_std = Kmodel.get_transformed_dG0(pH=7.5, I=0.2, T=298.15)
        
        return dG0_prime, dG0_std
        
    def reaction2Keq(self):
        '''
            Calculates the equilibrium constants of a reaction, using dG0.
            
            Arguments:
                List of cobra model reaction ids
            Returns:
                Array of K-equilibrium values
        '''
        dG0_prime, dG0_std = self.reaction2dG0()
        
#        print dG0_prime
               
        # error propagation
        dG0_std = np.matrix([x if x>0 else 0 for x in np.diag(dG0_std)]).T
        Keq = unumpy.umatrix(np.zeros(len(dG0_prime)), np.zeros(len(dG0_prime)))
        
        for i, (dG0, std) in enumerate(zip(dG0_prime, dG0_std)):
             udG0 = unumpy.uarray(dG0[0,0], std[0,0])
             try:                 
                 k = unumpy.exp( -udG0 / (R*default_T) ).max()
             except OverflowError:
                 k = 10**80 #value to bit to use exponent
             Keq[0,i] = k
        return Keq
            
    def reaction2RI(self, fixed_conc=0.1):
        '''
            Calculates the reversibility index (RI) of a reaction.
            The RI represent the change in concentrations of metabolites
            (from equal reaction reactants) that will make the reaction reversible.
            That is, the higher RI is, the more irreversible the reaction.
            A convenient threshold for reversibility is RI>=1000, that is a change of
            1000% in metabolite concentrations is required in ordeer to flip the
            reaction direction. 
            
            Arguments:
                List of cobra model reaction objects
            Returns:
                Array of RI values
    '''
        reaction_sparses, reaction_strings = self._reaction2string()
        N_P = np.zeros(len(reaction_sparses))
        N_S = np.zeros(len(reaction_sparses))
        
        for i,sparse in enumerate(reaction_sparses):   
            N_P[i] = sum([v for v in sparse.itervalues() if v>0])
            N_S[i] = -sum([v for v in sparse.itervalues() if v<0])

        N = N_P + N_S
        Q_2prime = np.matrix(fixed_conc**(N_P-N_S))
        Keq = self.reaction2Keq()

        RI = np.power( np.multiply(Keq,Q_2prime) , 2.0/N )
        
        return RI
        
if __name__ == "__main__":
    
    model_fname = "data/iJO1366.xml"
    model = create_cobra_model_from_sbml_file(model_fname)

    reactions = ['MDH', 'FBA', 'TPI', 'FBP', 'PGM', 'SERAT', 'TMDS', 'DBTS', 
                 'CS', 'BTS5', 'ENO', 'PANTS', 'METAT', 'GND', 'PGI']
                 
    reactions = map(lambda x: x.id, model.reactions)
    TFC = THERMODYNAMICS_FOR_COBRA(model, reactions)
    sparses, strings = TFC._reaction2string()
    dG0, std = TFC.reaction2dG0()    
    Keq = TFC.reaction2Keq()
    RI  = TFC.reaction2RI()

#    import matplotlib.pyplot as plt
#    x = unumpy.nominal_values(RI)
#    xerr = unumpy.std_devs(RI)
#    fig = plt.figure()
#    plt.hist(np.log10(x), cumulative=True, histtype='step', normed=True)
#    plt.scatter(np.arange(len(x))+1, x, s=100, alpha=0.5)
#    plt.yscale('log')
#    plt.ylim(1e-5, 1e5)    
#    plt.xlim(0, len(x)+1)    