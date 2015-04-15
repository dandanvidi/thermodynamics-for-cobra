'''

Requirements:

1 . uncertainties package - http://packages.python.org/uncertainties/index.html
2 . 

'''
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

    def __init__(self, model):

        self.pH = 7.5
        self.I = 0.2
        self.T = default_T

        self.model = deepcopy(model)
        
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
            except:
                CIDS.append('%s:not mapped' %m)
        return CIDS
        
    def _reaction2string(self, reaction_list):
        '''
            map COBRA reactions to reactions strings with KEGG cids, e.g.,
            "2 C19610 + C00027 + 2 C00080 <=> 2 C19611 + 2 C00001"
            
            Arguments:
                List of model reaction ids
            Returns:
                Reaction strings
        '''
        reaction_strings = []
        for r in reaction_list:

            reactants = map(lambda x: x.id, r.metabolites)
            reactant2cid = dict(zip(reactants, self._metabolite2cid(reactants)))
            sparse = {reactant2cid[m.id]:v for m,v in r.metabolites.iteritems()}
                
            # remove protons (H+) from reactions
            if 'C00080' in sparse:
                del sparse['C00080']            
                
            assert r not in reaction_strings, 'Duplicate reaction!'

            try:
                reaction_strings.append(str(KeggReaction(sparse)))
            except TypeError: 
                print sparse
                continue
        return reaction_strings

    def reaction2dG0(self,reaction_list=[]):
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
        
        reaction_strings = self._reaction2string(reaction_list)
        Kmodel = KeggModel.from_formulas(reaction_strings)
        Kmodel = KeggModel.from_formulas(reaction_strings)
        Kmodel.add_thermo(cc)
        dG0_prime, dG0_std = Kmodel.get_transformed_dG0(pH=7.5, I=0.2, T=298.15)
        dG0_prime = np.array(map(lambda x: x[0,0], dG0_prime))
        
        return dG0_prime, dG0_std
        
    def reaction2Keq(self,reaction_list):
        '''
            Calculates the equilibrium constants of a reaction, using dG0.
            
            Arguments:
                List of cobra model reaction ids
            Returns:
                Array of K-equilibrium values
        '''
        dG0_prime, dG0_std = self.reaction2dG0(reaction_list)
        
        # error propagation
        udG0 = unumpy.uarray(dG0_prime, np.diag(dG0_std))             
        Keq = unumpy.exp( -udG0 / (R*default_T) )

        return Keq
            
    def reaction2RI(self, reaction_list, fixed_conc=0.1):
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
        keq = self.reaction2Keq(reaction_list)
        
#        reaction_strings = self._reaction2string(reaction_list)
        sparse = map(lambda x: x.metabolites, reaction_list)

        N_P = np.zeros(len(sparse))
        N_S = np.zeros(len(sparse))    
        for i,s in enumerate(sparse):   
            N_P[i] = sum([v for v in s.itervalues() if v>0])
            N_S[i] = -sum([v for v in s.itervalues() if v<0])

        N = N_P + N_S
        Q_2prime = fixed_conc**(N_P-N_S)
        
        RI = ( keq*Q_2prime )**( 2.0/N )
        return RI
        
if __name__ == "__main__":
    
    model_fname = "data/iJO1366.xml"
    model = create_cobra_model_from_sbml_file(model_fname)
    TFC = THERMODYNAMICS_FOR_COBRA(model)

    reactions = ['MDH', 'FBA', 'TPI', 'FBP', 'PGM', 'SERAT', 'TMDS', 'DBTS', 
                 'CS', 'BTS5', 'ENO', 'PANTS', 'METAT', 'PANTS']
    reactions = map(model.reactions.get_by_id, reactions)
#    reactions = model.reactions
    
    strings = TFC._reaction2string(reactions)
    dG0, std = TFC.reaction2dG0(reactions)    
    Keq = TFC.reaction2Keq(reactions)
#    RI  = TFC.reaction2RI(reactions)

#    import matplotlib.pyplot as plt
#    x = unumpy.nominal_values(RI)
#    xerr = unumpy.std_devs(RI)
#    fig = plt.figure()
#    plt.hist(np.log10(x), cumulative=True, histtype='step', normed=True)
#    plt.scatter(np.arange(len(x))+1, x, s=100, alpha=0.5)
#    plt.yscale('log')
#    plt.ylim(1e-5, 1e5)    
#    plt.xlim(0, len(x)+1)    