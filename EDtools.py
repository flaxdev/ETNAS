# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:04:35 2024

@author: flavio

Early Detection Decision Tree

"""

import numpy as np
import pandas as pd


def applyHysteresis(WS, nhyst):       
    newWS = WS.copy()
    newWS = newWS.astype('float')
    inan = newWS.isna()
    newWS = newWS.fillna(0)
    newWS = newWS.rolling(nhyst+1).max()
    newWS.loc[(inan.values) & (newWS.values<1)] = np.nan
    return newWS

def applyBackHysteresis(Y, nhyst):       
    newY = Y.copy()
    newY = newY.iloc[::-1]
    inan = newY.isna()
    newY = newY.fillna(0)
    newY = newY.rolling(nhyst+1).max()
    newY.loc[(inan) & (newY<1)] = np.nan
    newY = newY.iloc[::-1]
    return newY



def findBeginsandEnds(Target):
       
    X = pd.concat([pd.DataFrame(data=[0]), Target], ignore_index=True)
    X[X==-1] = np.nan
    X = X.ffill()
    diff = X.iloc[1:].values.T[0] - X.iloc[:-1].values.T[0]
    iBegins = np.where(diff==1)[0]
    
    X = pd.concat([Target, pd.DataFrame(data=[0])], ignore_index=True)
    X[X==-1] = np.nan
    X = X.ffill()    
    diff = X.iloc[1:].values.T[0] - X.iloc[:-1].values.T[0]
    iEnds = np.where(diff==-1)[0]

    return iBegins, iEnds


def calculateSuccess(WS_Target_original):
    
    WS_Target = WS_Target_original.copy()    
    
    iBegins, iEnds = findBeginsandEnds(WS_Target['Target'])
    
    WS_Target.loc[WS_Target['WS']<0, 'WS'] = np.nan

    Ntot = len(iBegins)
    Nok = 0
    Ndiscarded = 0
    dTs = []
    tMissed = []
    tFound = []
    tDiscarded = []

    for i0, i1 in zip(iBegins, iEnds):
        
        if all(WS_Target['WS'].iloc[i0:(i1+1)].isna()):
           Ndiscarded = Ndiscarded + 1
           tDiscarded.append([i0,i1])
           Ntot = Ntot - 1
       
        elif all((WS_Target['WS'].iloc[i0:(i1+1)]<1) | (WS_Target['WS'].iloc[i0:(i1+1)].isna())):
            tMissed.append([i0,i1])
        
        else:
            Nok = Nok + 1
            tFound.append([i0,i1])
            
            if WS_Target['WS'].iloc[i0] > 0:
                istart = np.where(WS_Target['WS'].iloc[:i0] < 1)[0] 
                if len(istart) == 0:
                    istart = [-1]
                dTs.append(i0-istart[-1]-1)
            else:
                istart = np.where(WS_Target['WS'].iloc[i0:(i1+1)] > 0)[0]
                
                dTs.append(-istart[0])
            
    results = {'Nok':Nok, 'Ntot':Ntot, 'dTs':dTs, 'Ndiscarded':Ndiscarded, \
               'tMissed':tMissed, 'tFound':tFound, 'tDiscarded':tDiscarded}
    return results
    
def calculateFalse(WS_Target_original):
    
    WS_Target = WS_Target_original.copy()    

    iBegins, iEnds = findBeginsandEnds(WS_Target['WS'])
    
    WS_Target.loc[WS_Target['WS']<0,'WS'] = np.nan

    Nfalse = 0
    Ndiscarded = 0
    tFalse = []
    tDiscarded = []
    Ntot = len(iBegins)

    for i0, i1 in zip(iBegins, iEnds):
    
        if all(WS_Target['Target'].iloc[i0:(i1+1)].isna()):
           Ndiscarded = Ndiscarded + 1
           tDiscarded.append([i0,i1])
           Ntot = Ntot - 1
         
        elif all(WS_Target['Target'].iloc[i0:(i1+1)]<1):
            Nfalse = Nfalse + 1
            tFalse.append([i0,i1])
            
            
    tModelStrikes = list(zip(iBegins, iEnds))
    
    results = {'Nfalse':Nfalse, 'tFalse':tFalse, \
               'Ntot':Ntot, 'Ndiscarded':Ndiscarded, \
               'tDiscarded':tDiscarded, 'tModelStrikes':tModelStrikes}
    return results
    
    
def performance(WS_Target):
    
    success = calculateSuccess(WS_Target)
    false = calculateFalse(WS_Target)

    if success['Ntot']>0:
        TPR = success['Nok']/success['Ntot']
    else:
        TPR = np.nan
        
    deltaT = np.round(np.median(success['dTs'])) if len(success['dTs'])>0 else np.nan
    
    if false['Ntot']>0:
        FDR = false['Nfalse']/false['Ntot']
    else:
        FDR = np.nan
    
    activefraction = np.nan if (na:=WS_Target['WS'].notna().sum()) == 0 else len(WS_Target[WS_Target['WS']>0])/na 
    
    results = {'TPR':TPR, 'FDR':FDR, \
               'deltaT':deltaT, 'activefraction':activefraction, 'Nevents':success['Ntot'], 'Ndetectedevents':success['Nok'], 
               'Nalerts':false['Ntot'], 'Nfalsealerts': false['Nfalse']}
    return results


