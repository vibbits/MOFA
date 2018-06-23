import scipy as s

from .variational_nodes import Variational_Node
from .nodes import Constant_Node

"""
This module defines nodes that are a mix of variational and constant. 
Examples:
- For some factors the spike and slab sparsity parameter (theta) is fixed, whereas for others is learnt.
- For some factors the weights can be made sparse whereas for others we might not want to do any sparsity (i.e. mean or covariates)
"""

class Mixed_Theta_Nodes(Variational_Node, Constant_Node):
    """
    Class for a mixture of LearningTheta and ConstantTheta nodes.
    For a total of K factors, some (Klearn) will learn Theta whereas for the others (Kconst) it will be constant
        K = Klearn + Kconst
    """
    def __init__(self, LearnTheta, ConstTheta, idx):
        # Inputs:
        # - LearnTheta: Theta_Node with dimensions (Klearn,)
        # - ConstTheta: Theta_Constant_Node with dimensions (D,Kconst) or (Kconst,1) - NOT IMPLEMENTED YET -
        # - idx: list or numpy array indicating which factors are LearnTheta(idx=1. or idx=True) and which are ConstTheta(idx=0. or idx=False)
        self.constTheta = ConstTheta
        self.learnTheta = LearnTheta

        self.K = ConstTheta.dim[0] + LearnTheta.dim[0]
        self.D = ConstTheta.dim[0]

        self.idx = idx
        
    def precompute(self):
        self.constTheta.precompute()
        self.learnTheta.precompute()

    def addMarkovBlanket(self, **kargs):
        # SHOULD WE ALSO ADD MARKOV BLANKET FOR CONSTHTETA???
        self.learnTheta.addMarkovBlanket(**kargs)

    def getExpectations(self):
        
        # Get expectations from ConstTheta nodes (D,Kconst)
        Econst = self.constTheta.getExpectations()

        # Get expectations from LearnTheta nodes
        Elearn = self.learnTheta.getExpectations()

        # Concatenate
        # Concatenate expectations to (D,K)
        E = s.concatenate((Econst["E"], Elearn["E"]), axis=0)
        lnE = s.concatenate((Econst["lnE"], Elearn["lnE"]), axis=0)
        lnEInv = s.concatenate((Econst["lnEInv"], Elearn["lnEInv"]), axis=0)        

        # Permute to the right order given by self.idx
        # idx = s.concatenate((s.nonzero(1-self.idx)[0],s.where(self.idx)[0]), axis=0)
        # E, lnE, lnEinv = E[idx], lnE[idx], lnEInv[idx]

        return dict({'E':E, 'lnE':lnE, 'lnEInv':lnEInv})

    def getExpectation(self):
        return self.getExpectations()['E']

    def updateExpectations(self):
        self.learnTheta.updateExpectations()

    def updateParameters(self):
        # the argument contains the indices of the non_annotated factors
        self.learnTheta.updateParameters(s.nonzero(self.idx)[0])

    def calculateELBO(self):
        return self.learnTheta.calculateELBO()

    def removeFactors(self, *idx):
        for i in idx:
            if self.idx[idx] == 1:
                self.learnTheta.removeFactors(s.where(i == s.nonzero(self.idx)[0])[0])
            else:
                self.constTheta.removeFactors(s.where(i == s.nonzero(1-self.idx)[0])[0])
            self.idx = self.idx[s.arange(self.K)!=i]
            self.K -= 1

