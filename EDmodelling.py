# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:13:33 2024

@author: flavio
"""

import numpy as np
import pandas as pd

from EDtools import applyHysteresis, performance

from CustomFitnessTrees import DecisionTree, RandomForest


def strategy(TPR, FDR, LT, FWT):
    return F1(TPR, FDR, LT, FWT)


def PrecisionRecallCurve(model, X, y):
    
    y_prob = y.copy()
    y_prob.iloc[:] = model.predict_proba(X.values)
    
    PR = []
    
    for th in np.linspace(0, 1.0, 15):
        y_pred = y_prob.copy()
        
        y_pred[y_prob>=th] = 1
        y_pred[y_prob<th] = 0
        
        result = allscores(y_pred, y, model.hysteresis)

        precision = 1-result['FDR']
        recall = result['TPR']    
        
        if not np.isnan(precision*recall):
            PR.append([precision, recall])
        
    return np.unique(np.array(PR), axis=0)

def MakePrecisionRecallCurve(X,y, model_params, nperiods=4):
    
    Tsteps = pd.date_range(start=X.index[0],  end=X.index[-1], periods=(nperiods+2))
       
    PR = None
    # outer split
    for istep in range(1,nperiods+1):
        
        x_train = X.loc[X.index.values<Tsteps[istep]].copy()
        y_train = y.loc[X.index.values<Tsteps[istep]].copy()
        x_test = X.loc[(X.index.values>=Tsteps[istep]) & (X.index.values<Tsteps[istep+1])].copy()
        y_test = y.loc[(X.index.values>=Tsteps[istep]) & (X.index.values<Tsteps[istep+1])].copy()

        model = EDRandomForest(**model_params, bootstrap_sample_fraction=0.8)
        model = model.train(x_train.values, np.array(y_train))
        
        pr = PrecisionRecallCurve(model, x_test, y_test)
        
        PR = pr if PR is None else np.append(PR, pr, axis = 0)
        
    return np.unique(PR, axis=0)
        

def F1(TPR, FDR, LT, FWT):
    
    tpr = TPR*100
    fdr = 12*FDR
    lt = 20*(1-np.exp(-LT/3)) if LT>0 else -100
    fwt = FWT*150
    return -(tpr - fdr + lt - fwt)

def F2(TPR, FDR, LT, FWT):
    
    tpr = TPR*100
    fdr = 70*FDR
    lt = 26*(1-np.exp(-LT/3)) if LT>0 else -100
    fwt = FWT*100
    return -(tpr - fdr + lt - fwt)


def allscores(Y, y, hysteresis, **_):
    X = Y.to_frame().copy()
    y = y.copy()
    X['WS'] = Y        
    X['WS'] = applyHysteresis(X['WS'], hysteresis) 
    X['Target'] = y       
    perf = performance(X[['WS','Target']].copy())    
    return perf


def EvaluatePotentialModel(X, y, CV_param_dist, fitness=strategy):
    
    # return np.random.randn()*100
       
    TPRs, FDRs, deltaTs, activefracs = TimeSeriesNestedCrossValidation(X,y,CV_param_dist)
    
    val = fitness(np.nanmean(TPRs), np.nanmean(FDRs), np.nanmean(deltaTs), np.nanmean(activefracs))
    return val



def evalModelAgainstCrush(chromosomefeatures, X, y, CV_param_dist, nsets=3, disruptpercent=5):
    
    iok = np.array(chromosomefeatures).astype(bool)
    
    if any(iok):

        localX = X.loc[:,iok.astype(bool)].copy()
        
        vals = [EvaluatePotentialModel(localX, y.copy(), CV_param_dist)]
        
        for i in range(nsets-1):
            localX = X.loc[:,iok.astype(bool)].copy()
            
            for col in localX:
                idxmissing = np.random.choice(localX.index,  \
                                              int(localX[col].shape[0]*disruptpercent/100), \
                                              replace=False)
                    
                # impute the injected missing data
                localX.loc[idxmissing, col] = -1
            
            vals.append(EvaluatePotentialModel(localX, y.copy(), CV_param_dist))
        
        
        return np.NaN if (len(vals) == 0) or np.isnan(vals).all() else np.nanmean(vals) 
    
    else:
        return 1e9



class EDDecisionTree(DecisionTree):
           
    def __init__(self, max_depth=6, min_samples_leaf=1, 
                 min_misfit_gain=0.0, numb_of_features_splitting='sqrt', hysteresis = 1) -> None:
        super().__init__(max_depth, min_samples_leaf, 
                     min_misfit_gain, numb_of_features_splitting)
        
        self.hysteresis = hysteresis
        self.fitness = strategy
        
    
    def _misfit(self, data: np.array, originaltarget: np.array) -> float:            
        
        if len(data.shape)==1:
            detects = data.copy()
        else:
            detects = data[:,-1].copy()
        detects = pd.DataFrame(detects)
        X = applyHysteresis(detects, self.hysteresis)
        X.columns = ['WS']
        X['Target'] = originaltarget
        perf = performance(X)    
        
        if perf['activefraction']>0.5:
            return 1e10
        else:        
           return  self.fitness(perf['TPR'], perf['FDR'], perf['deltaT'], perf['activefraction'])



class EDRandomForest(RandomForest):
           
    def __init__(self, n_estimators=10, max_depth=7, min_samples_leaf=1, min_misfit_gain=0.0, \
                 numb_of_features_splitting='sqrt', bootstrap_sample_fraction=None, hysteresis=1) -> None:
        super().__init__(n_estimators, max_depth, min_samples_leaf, min_misfit_gain, \
                 numb_of_features_splitting, bootstrap_sample_fraction)
            
        self.hysteresis = hysteresis

    def create_tree(self):        
        return EDDecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, \
                                    min_misfit_gain=self.min_misfit_gain, 
                                    numb_of_features_splitting=self.numb_of_features_splitting,
                                    hysteresis=self.hysteresis)


def TimeSeriesNestedCrossValidation(X,y, CV_param_dist, minperiod=None, nperiods=4, nsplits=3, nparamsamples=3):
    
    
    TPRs = []   
    FDRs = []
    deltaTs = []
    activefracs = []
    
    if minperiod is not None:
        Tsteps = pd.date_range(start=X.index[0] + minperiod,  end=X.index[-1], periods=nperiods)
    else:
        Tsteps = pd.date_range(start=X.index[0],  end=X.index[-1], periods=(nperiods+1))
        Tsteps = Tsteps[1:]        
       
    # outer split
    for istep, tstep in enumerate(Tsteps):
        
        Tsplits = pd.date_range(start=X.index[0], end=tstep, periods=(nsplits+2))
        Tsplits = Tsplits[1:-1]
               
        
        # randomly sampling hyperparameters
        cvhyperparams = dict()
        for param in CV_param_dist:            
            cvhyperparams[param] = CV_param_dist[param].rvs(nparamsamples)
                        
        cvparamfitness = []
               
        # inner split hyperparameter tuning
        for icvparamsample in range(nparamsamples):
            
            valid_fitnesses = []
            
            hyperparams = dict()
            for param in cvhyperparams:            
                hyperparams[param] = cvhyperparams[param][icvparamsample]
            
            for tsplit0, tsplit1 in list(zip(Tsplits,Tsplits[1:])) :
            
            
                x_train = X.loc[X.index.values<tsplit0].copy()
                y_train = y.loc[X.index.values<tsplit0].copy()
                x_valid = X.loc[(X.index.values>=tsplit0) & (X.index.values<tsplit1)].copy()
                y_valid = y.loc[(X.index.values>=tsplit0) & (X.index.values<tsplit1)].copy()
           
                inan = y_train.isna()        
                x_train = x_train.loc[~inan]
                y_train = y_train.loc[~inan]
                       
                clf = EDRandomForest(**hyperparams, bootstrap_sample_fraction=0.8)
                clf = clf.train(x_train.values, np.array(y_train))
                
                y_valid_predicted = y_valid.copy()
                y_valid_predicted.iloc[:] = clf.predict(x_valid.values)
                                            
                valid_results = allscores(y_valid_predicted, y_valid, **hyperparams)
                
                valid_fitnesses.append(clf.base_learner_list[0].fitness(valid_results['TPR'], 
                                                                valid_results['FDR'], 
                                                                valid_results['deltaT'], 
                                                                valid_results['activefraction']))              
                
            
            cvparamfitness.append(np.nanmean(valid_fitnesses))
            
        ibestparams = np.argmin(cvparamfitness)
        besthyperparams = dict()
        for param in cvhyperparams:            
            besthyperparams[param] = cvhyperparams[param][ibestparams]
            
        x_train = X.loc[X.index.values<Tsplits[-1]].copy()
        y_train = y.loc[X.index.values<Tsplits[-1]].copy()
        x_test = X.loc[(X.index.values>=Tsplits[-1]) & (X.index.values<tstep)].copy()
        y_test = y.loc[(X.index.values>=Tsplits[-1]) & (X.index.values<tstep)].copy()

        clf = EDRandomForest(**besthyperparams, bootstrap_sample_fraction=0.8)
        clf = clf.train(x_train.values, np.array(y_train))
        
        y_test_predicted = y_test.copy()
        y_test_predicted.iloc[:] = clf.predict(x_test.values)
                            
        test_results = allscores(y_test_predicted, y_test, **besthyperparams)
        
        TPRs.append(test_results['TPR'])
        FDRs.append(test_results['FDR'])
        deltaTs.append(test_results['deltaT'])
        activefracs.append(test_results['activefraction'])
        
        
        # log results for each step
        line = pd.DataFrame.from_dict([besthyperparams])
        line['TPR'] = test_results['TPR']
        line['FDR'] = test_results['FDR']
        line['deltaT'] = test_results['deltaT']
        line['activefraction'] = test_results['activefraction']
                            
        with open("TSNCV_steps_F1_2011_2021_full.txt", "a") as myfile:
            myfile.write(f"{str(istep)} | {' '.join(X.columns.values)} | {line.to_string(header=False, index=False, index_names=False).replace(' ',' | ')} \r\n")

          

    # print(f'{TPRs}       {FDRs}         {deltaTs}      {activefracs}')
    return TPRs, FDRs, deltaTs, activefracs


def TimeSeriesCrossValidation(X,y, CV_param_dist, nperiods=4, nparamsamples=3):
    
    from tqdm import tqdm

        
    Tsteps = pd.date_range(start=X.index[0],  end=X.index[-1], periods=(nperiods+2))
       
    # randomly sampling hyperparameters
    cvhyperparams = dict()
    for param in CV_param_dist:            
        cvhyperparams[param] = CV_param_dist[param].rvs(nparamsamples)
                    
    cvparamfitness = []

    for istep in tqdm(range(1,nperiods+1)):
        
        x_train = X.loc[X.index.values<Tsteps[istep]].copy()
        y_train = y.loc[X.index.values<Tsteps[istep]].copy()
        x_test = X.loc[(X.index.values>=Tsteps[istep]) & (X.index.values<Tsteps[istep+1])].copy()
        y_test = y.loc[(X.index.values>=Tsteps[istep]) & (X.index.values<Tsteps[istep+1])].copy()
 
        paramfitness = []

        # inner split hyperparameter tuning
        for icvparamsample in range(nparamsamples):
                       
            hyperparams = dict()
            for param in cvhyperparams:            
                hyperparams[param] = cvhyperparams[param][icvparamsample]


            model = EDRandomForest(**hyperparams, bootstrap_sample_fraction=0.8)
            model = model.train(x_train.values, np.array(y_train))
    
            y_test_predicted = y_test.copy()
            y_test_predicted.iloc[:] = model.predict(x_test)
            
            test_results = allscores(y_test_predicted, y_test, **hyperparams)
            
            paramfitness.append(strategy(test_results['TPR'], test_results['FDR'], test_results['deltaT'],
                                           test_results['activefraction']))
            
        cvparamfitness.append(paramfitness)


          
    ibestparams = pd.DataFrame(cvparamfitness).mean().argmin().tolist()
    
    besthyperparams = dict()
    for param in cvhyperparams:            
        besthyperparams[param] = cvhyperparams[param][ibestparams]
    
    
    # print(f'{TPRs}       {FDRs}         {deltaTs}      {activefracs}')
    return besthyperparams

