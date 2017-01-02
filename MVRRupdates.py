from __future__ import division
import numpy.linalg  as linalg
import numpy.ma as ma

from variational_nodes import *
from utils import *

"""
    ###########################################
    ## Updates for the a non-sparse multi-view regularized regression
    ###########################################
    
    Required nodes:
    Covariate_Node: observed data
    Response_Node: observed response
    Coeff_Node: coefficients
    Tau_Node: precision of the noise
    Alpha_Node: groupwise ARD precision
    
    Each node is a Variational_Node() class with the following main variables:
    Important methods:
    - precompute: precompute some terms to speed up the calculations
    - calculateELBO: calculate evidence lower bound using current estimates of expectations/params
    - getParameters: return current parameters
    - getExpectations: return current expectations
    - updateParameters: update parameters using current estimates of expectations
    - updateExpectations: update expectations using current estimates of parameters
    - removeFactors: remove a set of latent variables from the node
    
    Important attributes:
    - markov_blanket: dictionary that defines the set of nodes that are in the markov blanket of the current node
    - Q: an instance of Distribution() which contains the specification of the variational distribution
    - P: an instance of Distribution() which contains the specification of the prior distribution
    - dim: dimensionality of the node
    
    Which class should Response and Covariates Nodes have?
    - Response has a distribution associates with it but is not updated , contributes vi likelihood to ELBO->Constant_Variational_Node
    - Covariates are assumed fixed, no updates, no contribution to ELBO -> Constant_Node

    TODO
    - need to fix assignment of beta to a certain gamma, cannot be done via markov blanket as all coefficients are in one node...
    """

class Response_Node(Constant_Variational_Node):
    def __init__(self, dim, value, liklihood):
        if(liklihood!="gaussian"):
            print("Only Gaussian response implemented so far")
            exit() 
        Constant_Variational_Node.__init__(self, dim, value)
        
        # Create a boolean mask of the data to hidden missing values
        if type(self.value) != ma.MaskedArray:
            self.mask()
        
        # Precompute some terms
        self.precompute()
    
    def precompute(self):
        # Precompute some terms to speed up the calculations
        # self.N = self.dim[0]
        self.N = self.dim[0] - ma.getmask(self.value).sum(axis=0)
        self.likconst = -0.5*self.N*s.log(2*s.pi)
    
    def mask(self):
        # Mask the observations if they have missing values
        self.value = ma.masked_invalid(self.value)
    
    def calculateELBO(self):
        tau_paramP = self.markov_blanket["Tau"].getParameters(dist="P")
        tau_paramQ = self.markov_blanket["Tau"].getParameters(dist="Q")
        tau_exp = self.markov_blanket["Tau"].getExpectations()

        lik = self.likconst + 0.5*self.N*(tau_exp["lnE"]) - tau_exp["E"]*(tau_paramQ["b"]-tau_paramP["b"])
        return lik


#node for covariates constant
#no updates, no contiburion to ELBO
class Covariate_Node(Constant_Node):
    def __init__(self, dim, value, annot):
        Constant_Node.__init__(self, dim, value)

        #save annotation  of features to views
        self.annot=annot
        # Create a boolean mask of the data to hidden missing values
        if type(self.value) != ma.MaskedArray:
            self.mask()
        # Precompute some terms
        self.precompute()    

    #pre-compute commonly used terms for speed issues
    def precompute(self):
        print("Computing XtX, might take some time, crashes for large numbers of features ~60,000")
        self.XtX=ma.dot(ma.transpose(self.value),self.value)
        print("Done")
   
    # Mask the observations if missing values
    def mask(self):
        self.value = ma.masked_invalid(self.value)


class Coefficient_Node(MultivariateGaussian_Unobserved_Variational_Node):
    def __init__(self, dim, qmean, qcov, qE=None, qE2=None):
        MultivariateGaussian_Unobserved_Variational_Node.__init__(self, dim=dim, qmean=qmean, qcov=qcov, qE=qE)
        self.precompute()


    def precompute(self):
        self.D = self.dim[0]
        self.lbconst = self.D*(s.log(2*s.pi)+1)

    def updateParameters(self):
        gammaSingle = self.markov_blanket["gamma"].getExpectation()
        annot=self.markov_blanket["Covariate"].annot
        gammaVec = np.repeat(gammaSingle, annot)
        tau = self.markov_blanket["Tau"].getExpectation()
        Response = self.markov_blanket["Response"].getExpectation()
        XtX=self.markov_blanket["Covariate"].XtX
        X=self.markov_blanket["Covariate"].getValue()

        self.Q.cov = linalg.inv(tau* XtX+ s.diag(gammaVec))
        tmp1 = tau*self.Q.cov
        tmp2 = ma.dot(ma.transpose(X), Response)
        self.Q.mean = ma.dot(tmp1,tmp2)


    def calculateELBO(self):
        print(self.markov_blanket["gamma"].getExpectations()['E'])
        gammaSingle = self.markov_blanket["gamma"].getExpectations()["E"]
        loggammaSingle = self.markov_blanket["gamma"].getExpectations()["lnE"]
        annot=self.markov_blanket["Covariate"].annot
        gammaVec = np.repeat(gammaSingle, annot)
        loggammaVec = np.repeat(loggammaSingle, annot)

        lb_p = self.D*s.sum(loggammaVec) - s.sum(self.Q.E2 * s.diag(gammaVec))
        lb_q = - self.lbconst - logdet(self.Q.cov).sum()
        return (lb_p - lb_q)/2


class Tau_Node(Gamma_Unobserved_Variational_Node):
    def __init__(self, dim, pa, pb, qa, qb, qE=None):
        Gamma_Unobserved_Variational_Node.__init__(self, dim=dim, pa=pa, pb=pb, qa=qa, qb=qb, qE=qE)
        self.precompute()
    
    def precompute(self):
        # Precompute some terms to speed up the calculations
        self.D = self.dim[0]
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb = P['a'], P['b']
        self.lbconst = s.sum(self.D*(Pa*s.log(Pb) - special.gammaln(Pa)))
    
    def updateParameters(self):
        X= self.markov_blanket["Covariate"].getExpectation()
        XtX=self.markov_blanket["Covariate"].XtX
        coeffE = self.markov_blanket["Coefficient"].getExpectation()
        coeffE2 = self.markov_blanket["Coefficient"].getExpectations()["E2"]
        Response = self.markov_blanket["Response"].getExpectation()

        # Collect parameters from the P and Q distributions of this node
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb = P['a'], P['b']

        # Perform updates of the Q distribution
        Qa = Pa + (Response.shape[0] - ma.getmask(Response).sum(axis=0))/2

        term1 = (Response**2).sum(axis=0).data
        term2 = (Response*s.dot(X,coeffE)).sum(axis=0).data
        term3 = s.dot(XtX,coeffE2).sum()
        tmp = term1 -2*term2 + term3 
        Qb = Pb + tmp/2

        # Save updated parameters of the Q distribution
        self.Q.setParameters(a=Qa, b=Qb)
        
        pass
    
    def calculateELBO(self):
        # Collect parameters and expectations
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb, Qa, Qb = P['a'], P['b'], Q['a'], Q['b']
        QE, QlnE = self.Q.expectations['E'], self.Q.expectations['lnE']

        # Calculate Variational Evidence Lower Bound
        lb_p =  (Pa-1)*QlnE- Pb*QE
        lb_q = Qa-log(Qb)+special.gammaln(Qa)+(1-Qa)*special.digamma(Qa) 
        
        return lb_p + lb_q


class gamma_Node(Gamma_Unobserved_Variational_Node):
    def __init__(self, dim, pa, pb, qa, qb, qE=None):
        Gamma_Unobserved_Variational_Node.__init__(self, dim=dim, pa=pa, pb=pb, qa=qa, qb=qb, qE=qE)

    def updateParameters(self):
        #Collect expextations from other nodes
        CoeffE2 = self.markov_blanket["Coefficient"].getExpectations()["E2"]

        # Collect parameters from the P and Q distributions of this node
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb = P['a'], P['b']
        #update
        Qa = Q['a']
        Qb = Pb + s.sum(CoeffE2)/2

       # Save updated parameters of the Q distribution
        self.Q.setParameters(a=Qa, b=Qb)

    def calculateELBO(self):
        # Collect parameters and expectations
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pa, Pb, Qa, Qb = P['a'], P['b'], Q['a'], Q['b']
        QE, QlnE = self.Q.expectations['E'], self.Q.expectations['lnE']
        
        #which dimesnion is used for gamma? M or D?
        lb_p = (Pa-1)*s.sum(QlnE)- Pb*s.sum(QE)
        lb_q = s.sum(Qa-log(Qb)+special.gammaln(Qa)+(1-Qa)*special.digamma(Qa) )
        
        return lb_p + lb_q


