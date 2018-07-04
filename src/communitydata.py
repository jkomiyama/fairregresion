# -*- coding: utf-8 -*-
"""
convert uci communities and crime dataset into S, X1, X2, y quadraple
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
import datetime
import sys
import os
import copy
import itertools
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd
#from statsmodels.discrete import discrete_model
import math
import random
from io import open
import conf

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

names=("state", "county", "community", "communityname", "fold", "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop", "ViolentCrimesPerPop")
numeric_values=("state", "county", "community", "fold", "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop", "ViolentCrimesPerPop")
string_values=("communityname")
missing_indices = set([1, 2, 30, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 121, 122, 123, 124,126])

def read_community(filename = os.path.join(conf.datadir, "communities.data"), use_explanatory=False, contS=False, return_all=False): #read community dataset file (numeric ver)
  global names, numeric_values, string_values
  lines = [line for line in open(filename, "r").readlines()]
  data = pd.read_csv(filename, sep=',', names=names)

  my_attrs = []
  for i,numeric_val in enumerate(names):
    if numeric_val not in string_values:
      if i not in missing_indices:
        my_attrs.append(data[numeric_val])
  for string_val in string_values:
    pass
    #my_attrs.append( pd.get_dummies(data[string_val], prefix=string_val, drop_first=True) )
  new_data = pd.concat(my_attrs, axis=1)
  new_data = new_data.dropna()
  if return_all:
    new_data["black"] = new_data["racepctblack"]>0.07
    return new_data
  new_data.insert(0, "intercept", 1)

  y = list(new_data["ViolentCrimesPerPop"])
 
  if not contS: #binary S
    #S_keys = ["racepctblack"]
    #S_keys = ["PctForeignBorn"]
    S_keys = ["racepctblack","PctForeignBorn"]
    S = np.transpose([list(new_data[i]) for i in S_keys])
    #print "cont S - corr between S[0] and S[1]=",np.corrcoef(map(lambda s:s[0],S), map(lambda s:s[1],S));sys.exit()
    #S = np.array(S>=0.07, dtype=np.int_) #cont->binary
  else: #cont S
    S_keys = ["medIncome"]
    S = np.transpose([list(new_data[i]) for i in S_keys])
    #sys.exit(0)
  #sys.exit(0)
  #S = np.array(S, dtype=np.int_)*2-1
  X_keys = set(new_data.keys()).difference([]+S_keys)
  print("X_keys=",len(X_keys),X_keys)
  #print list(race.keys())
  #X2_keys = set()
  if use_explanatory:
    X2_keys = set(["intercept","FemalePctDiv","PctImmigRecent"]).intersection(X_keys)
  else:
    X2_keys = set(["intercept"]).intersection(X_keys)
  print("X2 keys=",X2_keys)
  X2 = np.transpose([list(new_data[i]) for i in X2_keys])
  #print("X2=",str(X2))
  X2 = np.array(X2).reshape([len(new_data),len(X2_keys)])
  #print "X2=",X2.shape
  #print "X2=",X2
  X1_keys = X_keys.difference(X2_keys.union(set(["two_year_recid_1"]))).difference(set(["ViolentCrimesPerPop"]))
  X1 = np.transpose([list(new_data[i]) for i in X1_keys])
  print("X1 keys=",X1_keys)
  #print "S=",S[:10]

  return np.array(S), np.array(X1), np.array(X2), np.array(y)

if __name__ == '__main__':
 read_community("../dataset/communities.data")

