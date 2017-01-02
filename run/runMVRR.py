"""
To-do:
- Add covariates
"""

# Import required modules
import argparse
import os
from time import time
import pandas as pd
import scipy as s
# import pprint
from sys import path
from joblib import Parallel, delayed
from socket import gethostname

# Import manual functions
path.insert(0,"../")
from init_nodes import *
from BayesNet import BayesNet

# pp = pprint.PrettyPrinter(indent=4)


def pprint(d, indent=0):
    for key, value in d.iteritems():
        print '\t' * indent + str(key)
        if isinstance(value, dict):
            pprint(value, indent+1)
        else:
            print '\t' * (indent+1) + str(value)

# Function to load the data
def loadData(data_opts, verbose=True):

    print "\n"
    print "#"*18
    print "## Loading covariates data ##"
    print "#"*18
    print "\n"

    Y = list()
    for m in xrange(len(data_opts['input_files'])):
        file = data_opts['input_files'][m]
        print "Loading %s..." % file

        # Read file (with row and column names)
        tmp = pd.read_csv(file, delimiter=data_opts["delimiter"], header=data_opts["colnames"], index_col=data_opts["rownames"])
        print(tmp.shape)
        # Center the data
        if data_opts['center'][m]: 
            tmp = (tmp - tmp.mean())

        Y.append(tmp)
    return Y

#Function to load the response data
def loadResponse(data_opts, verbose=True):
    print "\n"
    print "#"*18
    print "## Loading response data ##"
    print "#"*18
    print "\n"

    file = data_opts['inputfile_Response']
    print "Loading %s..." % file

    response = pd.read_csv(file, delimiter=data_opts["delimiter"], header=data_opts["colnames"], index_col=data_opts["rownames"])
    print(response.shape)
    # Center the data
    if data_opts['center_Response']: 
        response = (response - response.mean())

    return response


# Function to run a single trial of the model
def runSingleTrial(data, response, model_opts, train_opts, seed=None, trial=1, verbose=False):

    # set the seed
    if seed is None:
        seed = int(round(time()*1000)%1e6)
    # s.random.seed(seed)

    print "\n"
    print "#"*45
    print "## Running trial number %d with seed %d ##" % (trial,seed)
    print "#"*45
    print "\n"


    ######################
    ## Define the model ##
    ######################

    # Define dimensionalities
    M = len(data)
    N = data[0].shape[0]
    if N!=len(response):
        print "Dimensions of response and features do not agree"
        exit()
    #define annotation of features per view
    annot = s.asarray([ data[m].shape[1] for m in xrange(M) ])
    D = s.sum(annot)

    dim = {'M':M, 'N':N, 'D':D}

    ## Define and initialise the nodes ##

    if verbose: print "Initialising nodes...\n"

    init = init_mvRR(dim, data, response, model_opts["liklihood"])

    init.initCovariate(annot=annot)

    init.initResponse()

    init.initgamma(pa=model_opts["priorgamma"]["a"], pb=model_opts["priorgamma"]["b"], qa=model_opts["initgamma"]['a'], qb=model_opts["initgamma"]['b'], qE=model_opts["initgamma"]['E'])

    init.initTau(pa=model_opts["priorTau"]['a'], pb=model_opts["priorTau"]['b'], 
                 qa=model_opts["initTau"]['a'], qb=model_opts["initTau"]['b'], qE=model_opts["initTau"]['E'])

    init.initCoefficients(qmean=s.zeros(s.sum(D)), qcov=s.eye(s.sum(D)))

    # Define the markov blanket of each node
    print "Defining Markov Blankets...\n"
    init.MarkovBlanket()

    ##################################
    ## Add the nodes to the network ##
    ##################################

    # Initialise Bayesian Network
    print "Initialising Bayesian network...\n"
    net = BayesNet(dim=dim, trial=trial, schedule=model_opts["schedule"], nodes=init.getNodes(), options=train_opts)

    ####################
    ## Start training ##
    ####################
    print "Start Training...\n"
    #print(model_opts["schedule"])
    #('Coefficient', 'gamma', 'Tau', 'Response', 'Covariate')
    net.iterate()

    #####################
    ## Finish training ##
    #####################

    return net

# Function to run multiple trials of the model
def runMultipleTrials(data_opts, model_opts, train_opts, cores, verbose=True):
    
    # If it doesnt exist, create the output folder
    outdir = os.path.dirname(train_opts['outfile'])
    if not os.path.exists(outdir): os.makedirs(outdir)

    ###################
    ## Load the data ##
    ###################

    data = loadData(data_opts, verbose)
    response = loadResponse(data_opts, verbose)

    #########################
    ## Run parallel trials ##
    ########################

    trials = Parallel(n_jobs=cores, backend="threading")(
        delayed(runSingleTrial)(data,response,model_opts,train_opts,None,i,verbose) for i in xrange(1,train_opts['trials']+1))

    #####################
    ## Process results ##
    #####################

    # Select the trial with the best lower bound
    lb = map(lambda x: x.getTrainingStats()["elbo"][-1], trials)
    best_model = trials[s.argmax(lb)]

    # Save the results
    print "\nSaving model in %s..." % train_opts['outfile']
    sample_names = data[0].index.tolist()
    feature_names = [  data[m].columns.values.tolist() for m in xrange(len(data)) ]
    saveModel(best_model, outfile=train_opts['outfile'], view_names=data_opts['view_names'], 
        sample_names=sample_names, feature_names=feature_names)



if __name__ == '__main__':

    #############################
    ## Define the data options ##
    #############################

    data_opts = {}

    if 'mac-huber12' in gethostname():
        base_folder = "/Users/bvelten/Documents/LassoVariants/DataCompendium/data/simulation"
    else:
        print "Computer not recognised"
        exit()

    #data_opts['view_names'] = ( "expr",  "met_3utr")#,"met_5utr","met_cds_genebody","met_noncds_genebody","met_intergenic","met_prom2k")
    #data_opts['input_files'] = [ "%s/%s.txt" % (base_folder,m) for m in data_opts['view_names'] ]
    
    viewnames=list()
    for i in range(10):
        viewnames.append("view_" + str(i+1))
    data_opts['view_names'] = viewnames
    data_opts['input_files'] = [ "%s/%s.txt" % (base_folder,m) for m in data_opts['view_names'] ]
    M = len(data_opts['input_files'])
    data_opts['center'] = [True]*M
    data_opts['rownames'] = 0
    data_opts['colnames'] = 0
    data_opts['delimiter'] = "\t"
    

    #for response data seperate:
    data_opts['center_Response'] = True
    #data_opts['inputfile_Response'] = "%s/%s.txt" %(base_folder,"nutlin-3_45")
    data_opts['inputfile_Response'] = "%s/%s.txt" %(base_folder,"response")

    # pprint(data_opts)
    # print "\n"

    ##############################
    ## Define the model options ##
    ##############################

    model_opts = {}
    model_opts["liklihood"]="gaussian"
    

    # Define priors
    #use low values =uniformtive prior
    model_opts["priorgamma"] = { 'a':[1e-5]*M, 'b':[1e-5]*M }
    model_opts["priorTau"] = { 'a':[1e-5], 'b':[1e-5]}
    #need to fix prior variance of coefficients, no prior -->  conditional
    model_opts["priorCoefficient"] = {'mean':s.nan, 'var':s.nan}

    # Define initialisation options
    #set inital expected value to high values (=low penalization)
    model_opts["initgamma"] = { 'a':[s.nan]*M, 'b':[s.nan]*M, 'E':[100.]*M }
    model_opts["initCoefficient"] = {'mean':s.nan, 'var': s.nan, 'E': [0]}
    #set initial expected value to high value (=low noise level)
    model_opts["initTau"] = { 'a':s.nan, 'b':s.nan, 'E': 100. }

    # Define schedule of updates
    model_opts['schedule'] = ("Coefficient","gamma","Tau", "Response", "Covariate")

    # pprint(model_opts)
    # print "\n"

    #################################
    ## Define the training options ##
    #################################

    train_opts = {}
    train_opts['maxiter'] = 300
    train_opts['elbofreq'] = 1
    if 'mac-huber12' in gethostname():
        train_opts['outfile'] = "/Users/bvelten/Documents/LassoVariants/BayesNetPython/outBayesNet/model.hdf5"
    else:
        print "Computer not recognised"
        exit()
    train_opts['savefreq'] = s.nan
    train_opts['savefolder'] = s.nan
    train_opts['verbosity'] = 2
    train_opts['dropK'] = {}
    train_opts['dropK']['by_norm'] = None
    train_opts['dropK']['by_pvar'] = None
    train_opts['dropK']['by_cor'] = None
    train_opts['forceiter'] = True
    train_opts['tolerance'] = 0.01


    # Define the number of trials and cores
    train_opts['trials'] = 1
    cores = 1

    # pprint(data_opts)
    # print "\n"

    # print "Cores: %d " % cores
    # print "\n"

    # Go!
    runMultipleTrials(data_opts, model_opts, train_opts, cores)
