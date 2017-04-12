from __future__ import division
import scipy as s
import numpy.ma as ma

from variational_nodes import *
from nodes import Node,Constant_Node
from utils import sigmoid, lambdafn

"""
Module to define updates for non-conjugate matrix factorisation models using the Seeger approach
Reference: 'Fast Variational Bayesian Inference for Non-Conjugate Matrix Factorisation models' by Seeger and Bouchard (2012)
"""


################
## Pseudodata ##
################

class PseudoY(Unobserved_Variational_Node):
    """
    General class for pseudodata nodes

    IMPROVE EXPLANATION

    """
    def __init__(self, dim, obs, Zeta=None, E=None):
        # Inputs:
        #  dim (2d tuple): dimensionality of each view
        #  obs (ndarray): observed data
        #  Zeta: parameter
        #  E (ndarray): initial expected value of pseudodata
        Unobserved_Variational_Node.__init__(self, dim)

        # Initialise observed data
        assert obs.shape == dim, "Problems with the dimensionalities"
        self.obs = obs

        # Initialise Zeta
        self.Zeta = Zeta

        # Create a boolean mask of the data to hidden missing values
        if type(self.obs) != ma.MaskedArray:
            self.mask()

        # Precompute some terms
        self.precompute()

        # Initialise expectation
        if E is not None:
            assert E.shape == dim, "Problems with the dimensionalities"
            E = ma.masked_invalid(E)
        # else:
            # E = ma.masked_invalid(s.zeros(self.dim))
        self.E = E

    def updateParameters(self):
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        self.Zeta = s.dot(Z,W.T)

    def mask(self):
        # Mask the observations if they have missing values
        self.obs = ma.masked_invalid(self.obs)

    def precompute(self):
        # Precompute some terms to speed up the calculations
        # self.N = self.dim[0]
        # self.D = self.dim[1]
        # self.lbconst = -0.5*self.N*self.D*s.log(2*s.pi)
        # self.N = self.dim[0] - ma.getmask(self.obs).sum(axis=0)
        # self.D = self.dim[1]
        # self.lbconst = -0.5*s.sum(self.N)*s.log(2*s.pi)
        pass

    def updateExpectations(self):
        pass

    def getExpectation(self):
        return self.E

    def getObservations(self):
        return self.obs

    def getValue(self):
        return self.obs

    def getExpectations(self):
        return { 'E':self.getExpectation() }

    def getParameters(self):
        return { 'zeta':self.Zeta }

    # def getExpectations(self):
        # return { 'obs':self.getObservations() }

    def calculateELBO(self):
        # TODO is it ever used ??
        # Compute Lower Bound using the Gaussian likelihood with pseudodata
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        kappa = self.markov_blanket["kappa"].getExpectation()

        lb = self.lbconst + s.sum(self.N*s.log(kappa))/2 - s.sum( kappa * (self.E-s.dot(Z,W.T))**2 )/2
        return lb

class Poisson_PseudoY_Node(PseudoY):
    """
    Class for a Poisson pseudodata node with the following likelihood:
        p(y|x) \prop gamma(x) * e^{-gamma(x)}  (1)
    where gamma(x) is a rate function that is chosen to be convex and log-concave
    A simple choise for the rate function is e^{x} but this rate function is non-robust
    in the presence of outliers, so in Seeger et al they chose the function:
        gamma(x) = log(1+e^x)

    The data follows a Poisson distribution, but Followung Seeger et al the pseudodata Yhat_ij
    follows a normal distribution with mean E[W_{i,:}]*E[Z_{j,:}] and precision 'kappa_j'
    where 'kappa_j' is an upper bound of the second derivative of the loglikelihood:
        x_ij = sum_k^k w_{i,k}*z_{k,j}
        f_ij''(x_ij) <= kappa_j for all i,j

    For the Poisson likelihood with rate function (1), the upper bound kappa is calculated as follows:
        f_ij''(x_ij) = 0.25 + 0.17*ymax_j   where ymax_j = max(Y_{:,j})

    Pseudodata is updated as follows:
        yhat_ij = zeta_ij - f'(zeta_ij)/kappa_j = ...
    The bound degrades with the presence of entries with large y_ij, so one should consider
    clipping overly large counts

    """
    def __init__(self, dim, obs, Zeta=None, E=None):
        # - dim (2d tuple): dimensionality of each view
        # - obs (ndarray): observed data
        # - E (ndarray): initial expected value of pseudodata
        PseudoY.__init__(self, dim=dim, obs=obs, Zeta=Zeta, E=E)

        # Initialise the observed data
        assert s.all(s.mod(self.obs, 1) == 0), "Data must not contain float numbers, only integers"
        assert s.all(self.obs >= 0), "Data must not contain negative numbers"

    def ratefn(self, X):
        # Poisson rate function
        return s.log(1+s.exp(X))

    def clip(self, threshold):
        # The local bound degrades with the presence of large values in the observed data, which should be clipped
        pass

    def updateExpectations(self):
        # Update the pseudodata
        kappa = self.markov_blanket["kappa"].getValue()
        self.E = self.Zeta - sigmoid(self.Zeta)*(1-self.obs/self.ratefn(self.Zeta))/kappa[None,:]

    def calculateELBO(self):
        # Compute Lower Bound using the Poisson likelihood with observed data
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        tmp = self.ratefn(s.dot(Z,W.T))
        lb = s.sum( self.obs*s.log(tmp) - tmp)
        return lb
class Bernoulli_PseudoY_Node(PseudoY):
    """
    Class for a Bernoulli (0,1 data) pseudodata node with the following likelihood:
        p(y|x) = (e^{yx}) / (1+e^x)  (1)
        f(x) = -log p(y|x) = log(1+e^x) - yx

    The second derivative is upper bounded by kappa=0.25

    Folloiwng Seeger et al, the data follows a Bernoulli distribution but the pseudodata follows a
    normal distribution with mean E[W]*E[Z] and precision 'kappa'

    IMPROVE EXPLANATION

    Pseudodata is updated as follows:
        yhat_ij = zeta_ij - f'(zeta_ij)/kappa
                = zeta_ij - 4*(sigmoid(zeta_ij) - y_ij)

    """
    def __init__(self, dim, obs, Zeta=None, E=None):
        # - dim (2d tuple): dimensionality of each view
        # - obs (ndarray): observed data
        # - E (ndarray): initial expected value of pseudodata
        PseudoY.__init__(self, dim=dim, obs=obs, Zeta=Zeta, E=E)

        # Initialise the observed data
        assert s.all( (self.obs==0) | (self.obs==1) ), "Data must be binary"

    def updateExpectations(self):
        # Update the pseudodata
        self.E = self.Zeta - 4.*(sigmoid(self.Zeta) - self.obs)

    def calculateELBO(self):
        # Compute Lower Bound using the Bernoulli likelihood with observed data
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        tmp = s.dot(Z,W.T)
        lik = s.sum( self.obs*tmp - s.log(1+s.exp(tmp)) )
        return lik
class Binomial_PseudoY_Node(PseudoY):
    """
    Class for a Binomial pseudodata node with the following likelihood:
        p(x|N,theta) = p(x|N,theta) = binom(N,x) * theta**(x) * (1-theta)**(N-x)
        f(x) = -log p(x|N,theta) = -log(binom(N,x)) - x*theta - (N-x)*(1-theta)

    The second derivative is upper bounded:
        f''(x_ij) <= 0.25*max(N_{:,j})

    Folloiwng Seeger et al, the pseudodata yhat_{nd} follows a normal distribution with mean
    E[w_{i,:}]*E[z_{j,:}] and precision 'kappa_j'

    IMPROVE EXPLANATION

    Pseudodata is updated as follows
        yhat_ij = zeta_ij - f'(zeta_ij)/kappa_j
                = zeta_ij - (N_{ij}*sigmoid(zeta_ij) - y_ij)/kappa_d

    """
    def __init__(self, dim, obs, tot, Zeta=None, E=None):
        # - dim (2d tuple): dimensionality of each view
        # - obs (ndarray): observed data
        # - E (ndarray): initial expected value of pseudodata
        PseudoY.__init__(self, dim=dim, obs=None, Zeta=Zeta, E=E)

        # Initialise the observed data
        assert s.all(s.mod(obs, 1) == 0) and s.all(s.mod(tot, 1) == 0), "Data must not contain float numbers, only integers"
        assert s.all(obs >= 0) and s.all(tot >= 0), "Data must not contain negative numbers"
        assert s.all(obs <= tot), "Observed counts have to be equal or smaller than the total counts"
        self.obs = obs
        self.tot = tot


    def updateExpectations(self):
        # Update the pseudodata
        kappa = self.markov_blanket["kappa"].getValue()
        self.E = self.zeta - s.divide(self.tot*sigmoid(self.Zeta)-self.obs, kappa)
        pass

    def calculateELBO(self):
        # Compute Lower Bound using the Bernoulli likelihood with observed data
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()

        tmp = sigmoid(s.dot(Z,W.T))

        # TODO change apprximation
        tmp[tmp==0] = 0.00000001
        tmp[tmp==1] = 0.99999999
        lik = s.log(s.special.binom(self.tot,self.obs)).sum() + s.sum(self.obs*s.log(tmp)) + \
            s.sum((self.tot-self.obs)*s.log(1-tmp))
        return lik



## Jakkola ##

class Bernoulli_PseudoY_Node_Jaakkola(PseudoY):
    """
    Class for a Bernoulli (0,1 data) pseudodata node with the following likelihood:
        p(y|x) = (e^{yx}) / (1+e^x) 
    Following Jaakola et al and intterpreting the bound as a liklihood on gaussian pseudodata
    leads to the folllowing updates

    Pseudodata is given by
            yhat_ij = (2*y_ij-1)/(4*lambadfn(xi_ij))
        where lambdafn(x)= tanh(x/2)/(4*x).
    
    Its conditional distribution is given by 
            N((ZW)_ij, 1/(2 lambadfn(xi_ij)))       
    
    Updates for the variational parameter xi_ij are given by
            sqrt(E((ZW)_ij^2))

    xi_ij in above notation is the same as zeta (variational parameter)         

    NOTE: For this class to work the noise variance tau needs to be updated according to 
        tau_ij <- 2*lambadfn(xi_ij)
    """
    def __init__(self, dim, obs, Zeta=None, E=None):
        # - dim (2d tuple): dimensionality of each view
        # - obs (ndarray): observed data
        # - E (ndarray): initial expected value of pseudodata
        PseudoY.__init__(self, dim=dim, obs=obs, Zeta=Zeta, E=E)

        # Initialise the observed data
        assert s.all( (self.obs==0) | (self.obs==1) ), "Data must be binary"

    def updateExpectations(self):
        # Update the pseudodata
        self.E = (2.*self.obs - 1.)/(4.*lambdafn(self.Zeta))

    def updateParameters(self):
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        ZZ = self.markov_blanket["Z"].getExpectations()["E2"]
        WW = self.markov_blanket["W"].getExpectations()["ESWW"]
        self.Zeta = s.sqrt( s.square(Z.dot(W.T)) - s.dot(s.square(Z),s.square(W.T)) + s.dot(ZZ, WW.T) )

    def calculateELBO(self):
        # Compute Lower Bound using the Bernoulli likelihood with observed data
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        tmp = s.dot(Z,W.T)
        lik = s.sum( self.obs*tmp - s.log(1+s.exp(tmp)) )
        return lik
class Tau_Node_Jaakkola(Node):
    def __init__(self, dim, value):
        Node.__init__(self, dim=dim)
        if isinstance(value,(int,float)):
            self.value = value * s.ones(dim)
        else:
            assert value.shape == dim, "dimensionality mismatch"
            self.value = value

    def updateExpectations(self):
        Z = self.markov_blanket["Z"].getExpectation()
        W = self.markov_blanket["W"].getExpectation()
        self.value = 2*lambdafn(s.dot(Z,W.T))

    def getValue(self):
        return self.value

    def getExpectation(self):
        return self.getValue()
        
    def getExpectations(self):
        return { 'E':self.getValue(), 'lnE':s.log(self.getValue()) }
class Z_Node_Jaakkola(UnivariateGaussian_Unobserved_Variational_Node):
    def __init__(self, dim, pmean, pvar, qmean, qvar, qE=None, qE2=None, idx_covariates=None):
        # UnivariateGaussian_Unobserved_Variational_Node.__init__(self, dim=dim, pmean=pmean, pvar=pvar, qmean=qmean, qvar=qvar, qE=qE)
        super(Z_Node_Jaakkola,self).__init__(dim=dim, pmean=pmean, pvar=pvar, qmean=qmean, qvar=qvar, qE=qE, qE2=qE2)
        self.precompute()

        # Define indices for covariates
        if idx_covariates is not None:
            self.covariates[idx_covariates] = True

    def precompute(self):
        # Precompute terms to speed up computation
        self.N = self.dim[0]
        self.covariates = np.zeros(self.dim[1], dtype=bool)
        self.factors_axis = 1

    def getLvIndex(self):
        # Method to return the index of the latent variables (without covariates)
        latent_variables = np.array(range(self.dim[1]))
        if any(self.covariates):
            latent_variables = np.delete(latent_variables, latent_variables[self.covariates])
        return latent_variables

    def updateParameters(self):

        # Collect expectations from the markov blanket
        Y = self.markov_blanket["Y"].getExpectation()
        SWtmp = self.markov_blanket["SW"].getExpectations()
        tau = self.markov_blanket["Tau"].getExpectation()
        Mu = self.markov_blanket['Cluster'].getExpectation()

        # Collect parameters from the P and Q distributions of this node
        P,Q = self.P.getParameters(), self.Q.getParameters()
        Pvar, Qmean, Qvar = P['var'], Q['mean'], Q['var']

        # Concatenate multi-view nodes to avoid looping over M (maybe its not a good idea)
        M = len(Y)
        Y = ma.concatenate([Y[m] for m in xrange(M)],axis=1)
        SW = s.concatenate([SWtmp[m]["E"]for m in xrange(M)],axis=0)
        SWW = s.concatenate([SWtmp[m]["ESWW"] for m in xrange(M)],axis=0)
        tau = s.concatenate([tau[m] for m in xrange(M)],axis=1)


        D = SW.shape[0]
        K = SW.shape[1]
        N = Y.shape[0]
        for n in xrange(N):
            for k in xrange(K):
                # Variance
                tmp = 0
                for d in xrange(D):
                    tmp += tau[n,d]*SWW[d,k]
                Qvar[n,k] = 1/(tmp + 1)
                # Mean
                tmp = 0
                for d in xrange(D):
                    tmp += tau[n,d]*SW[d,k] * (Y[n,d] - s.sum(SW[d,s.arange(K)!=k]*Qmean[n,s.arange(K)!=k]))
                Qmean[n,k] = Qvar[n,k]*tmp

        # Save updated parameters of the Q distribution
        self.Q.setParameters(mean=Qmean, var=Qvar)

    def calculateELBO(self):
        # Collect parameters and expectations of current node
        Ppar,Qpar,Qexp = self.P.getParameters(), self.Q.getParameters(), self.Q.getExpectations()
        Pvar, Qmean, Qvar = Ppar['var'], Qpar['mean'], Qpar['var']
        PE, PE2 = self.markov_blanket['Cluster'].getExpectations()['E'], self.markov_blanket['Cluster'].getExpectations()['E2']
        QE, QE2 = Qexp['E'],Qexp['E2']

        # This ELBO term contains only cross entropy between Q and P,and entropy of Q. So the covariates should not intervene at all
        latent_variables = self.getLvIndex()
        Pvar, Qmean, Qvar = Pvar[:, latent_variables], Qmean[:, latent_variables], Qvar[:, latent_variables]
        PE, PE2 = PE[:, latent_variables], PE2[:, latent_variables]
        QE, QE2 = QE[:, latent_variables],QE2[:, latent_variables]

        # compute term from the exponential in the Gaussian
        tmp1 = 0.5*QE2 - PE*QE + 0.5*PE2
        tmp1 = -(tmp1/Pvar).sum()

        # compute term from the precision factor in front of the Gaussian
        tmp2 = - (s.log(Pvar)/2.).sum()

        lb_p = tmp1 + tmp2
        lb_q = - (s.log(Qvar).sum() + self.N*self.dim[1])/2.

        return lb_p-lb_q
class SW_Node_Jaakkola(BernoulliGaussian_Unobserved_Variational_Node):
    # def __init__(self, dim, pmean_S0, pmean_S1, pvar_S0, pvar_S1, ptheta, qmean_S0, qmean_S1, qvar_S0, qvar_S1, qtheta, qEW_S0=None, qEW_S1=None, qES=None):
    def __init__(self, dim, pmean_S0, pmean_S1, pvar_S0, pvar_S1, ptheta, qmean_S0, qmean_S1, qvar_S0, qvar_S1, qtheta, qEW_S0=None, qEW_S1=None, qES=None):
        super(SW_Node_Jaakkola,self).__init__(dim, pmean_S0, pmean_S1, pvar_S0, pvar_S1, ptheta, qmean_S0, qmean_S1, qvar_S0, qvar_S1, qtheta, qEW_S0, qEW_S1, qES)
        # BernoulliGaussian_Unobserved_Variational_Node.__init__(self, dim, pmean_S0, pmean_S1, pvar_S0, pvar_S1, ptheta, qmean_S0, qmean_S1, qvar_S0, qvar_S1, qtheta, qEW_S0, qEW_S1, qES)
        self.precompute()

    def precompute(self):
        self.D = self.dim[0]
        # self.K = self.dim[1]
        self.factors_axis = 1

    def updateParameters(self):

        # Collect expectations from other nodes
        tmp = self.markov_blanket["Z"].getExpectations()
        Z,ZZ = tmp["E"],tmp["E2"]
        tau = self.markov_blanket["Tau"].getExpectation()
        Y = self.markov_blanket["Y"].getExpectation()
        alpha = self.markov_blanket["Alpha"].getExpectation()
        SW = self.Q.getExpectations()["E"].copy()
        Q = self.Q.getParameters()
        Qmean_S1, Qvar_S1, Qtheta = Q['mean_S1'].copy(), Q['var_S1'].copy(), Q['theta'].copy()

        thetatmp = self.markov_blanket['Theta'].getExpectations() # TODO make general in mixed nodw
        theta_lnE, theta_lnEInv  = thetatmp['lnE'], thetatmp['lnEInv']  
        if theta_lnE.shape != Qmean_S1.shape:
            theta_lnE = s.repeat(theta_lnE[None,:],Qmean_S1.shape[0],0)
        if theta_lnEInv.shape != Qmean_S1.shape:
            theta_lnEInv = s.repeat(theta_lnEInv[None,:],Qmean_S1.shape[0],0)

        all_term1 = theta_lnE - theta_lnEInv

        N = Y.shape[0]
        D = self.dim[0]
        K = self.dim[1]
        for d in xrange(D):
            for k in xrange(K):
                term1 = all_term1[d,k]
                term2 = 0.5*s.log(alpha[k])
                tmp = 0
                for n in xrange(N):
                    tmp += ZZ[n,k]*tau[n,d]
                term3 = 0.5*s.log(tmp + alpha[k])

                term4 = 0
                term4_tmp1 = 0
                for n in xrange(N):
                    if not s.isnan(Y.data[n,d]):
                        term4_tmp1 += Y[n,d]*Z[n,k]*tau[n,d]
                term4_tmp2 = 0
                for j in xrange(K):
                    if j!=k:
                        for n in xrange(N):
                            if not s.isnan(Y.data[n,d]): 
                                term4_tmp2 += SW[d,j]*Z[n,k]*Z[n,j]*tau[n,d]

                term4_tmp3 = alpha[k]
                for n in xrange(N):
                    if not s.isnan(Y.data[n,d]): 
                        term4_tmp3 += tau[n,d]*ZZ[n,k]

                term4 = 0.5*(term4_tmp1 - term4_tmp2)**2 / term4_tmp3

                # Update S
                Qtheta[d,k] = 1/(1+s.exp(-(term1+term2-term3+term4)))

                # Update W
                Qmean_S1[d,k] = Qvar_S1[d,k]*(term4_tmp1 - term4_tmp2)
                Qvar_S1[d,k] = 1/term4_tmp3

                # Update expectations
                SW[d,k] = Qtheta[d,k] * Qmean_S1[d,k]
                
        # Save updated parameters of the Q distribution
        self.Q.setParameters(mean_S0=s.zeros((self.D,self.dim[1])), var_S0=s.repeat(1/alpha[None,:],self.D,0),
                             mean_S1=Qmean_S1, var_S1=Qvar_S1, theta=Qtheta )

    def calculateELBO(self):

        # Collect parameters and expectations
        Qpar,Qexp = self.Q.getParameters(), self.Q.getExpectations()
        S,WW = Qexp["ES"], Qexp["EWW"]
        Qvar = Qpar['var_S1']
        alpha = self.markov_blanket["Alpha"].getExpectations()
        theta = self.markov_blanket['Theta'].getExpectations()


        # Calculate ELBO for W
        lb_pw = (self.D*alpha["lnE"].sum() - s.sum(alpha["E"]*WW))/2
        # lb_qw = -0.5*self.dim[1]*self.D - 0.5*s.log(S*Qvar + ((1-S)/alpha["E"])).sum()
        lb_qw = -0.5*self.dim[1]*self.D - 0.5*(S*s.log(Qvar) + (1-S)*s.log(1/alpha["E"])).sum()
        lb_w = lb_pw - lb_qw

        # Calculate ELBO for S
        lb_ps_tmp = S*theta['lnE'] + (1.-S)*theta['lnEInv']
        lb_qs_tmp = S*s.log(S) + (1.-S)*s.log(1.-S)

        # Ignore NAs
        lb_ps_tmp[s.isnan(lb_ps_tmp)] = 0.
        lb_qs_tmp[s.isnan(lb_qs_tmp)] = 0.

        lb_ps = s.sum(lb_ps_tmp)
        lb_qs = s.sum(lb_qs_tmp)
        lb_s = lb_ps - lb_qs

        return lb_w + lb_s






class Y_Node(Constant_Variational_Node):
    def __init__(self, dim, value):
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
        self.D = self.dim[1]
        # self.likconst = -0.5*self.N*self.D*s.log(2*s.pi)
        self.likconst = -0.5*s.sum(self.N)*s.log(2.*s.pi)

    def mask(self):
        # Mask the observations if they have missing values
        self.value = ma.masked_invalid(self.value)

    def calculateELBO(self):
        # tauQ_param = self.markov_blanket["Tau"].getParameters("Q")
        # tauP_param = self.markov_blanket["Tau"].getParameters("P")
        # tau_exp = self.markov_blanket["Tau"].getExpectations()
        # lik = self.likconst + 0.5*s.sum(self.N*(tau_exp["lnE"])) - s.dot(tau_exp["E"],tauQ_param["b"]-tauP_param["b"])
        # return lik

        tau = self.markov_blanket["Tau"].getExpectations()
        SW = self.markov_blanket["SW"].getExpectations()
        Z = self.markov_blanket["Z"].getExpectations()
        Y = self.value.data

        N = self.dim[0]
        D = self.dim[1]
        K = SW["ES"].shape[1]

        lik = 0
        for n in xrange(N):
            for d in xrange(D):
                tmp = 0
                if not s.isnan(Y[n,d]):
                    for k in xrange(K):
                        tmp += SW["ES"][d,k]*SW["EWW"][d,k]*Z["E2"][n,k]
                        for j in xrange(k+1,K):
                            tmp += 2*SW["E"][d,k]*Z["E"][n,k] * SW["E"][d,j]*Z["E"][n,j]
                    lik += -0.5*s.log(2.*s.pi) + 0.5*tau["lnE"][n,d] - 0.5*tau["E"][n,d]*(Y[n,d]**2 - 2*Y[n,d]*s.dot(SW["E"][d,:],Z["E"][n,:]) + tmp)
        return lik
