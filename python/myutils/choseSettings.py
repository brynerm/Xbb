import pickle
import numpy as np
import pandas as pd
import argparse
import os 

parser = argparse.ArgumentParser(description='read file')
parser.add_argument('filename', metavar='file1', type=str, nargs='+',
                           help='pickle file')
#parser.add_argument('--showBins',action='store_const', const=True, default=False, dest='showBins', help='show all bin entries')
#parser.add_argument('--total', action='store_const', const=True, default=False, dest='total', help='print total number of entries in regions')
args=parser.parse_args()
pd.set_option('display.width', 1000)
settings=[]
methods=[]
dictPerformance={"ROCint_test":[],"ES_test":[],"ES_diffRel":[], "ROCint_diffRel":[], "KS_S":[] }
dictAll={}
for filename in args.filename:
    print filename
    dictAll.update(pickle.load(open(filename,"rb")))

for method, parms in dictAll.items():
    for key in dictPerformance:
        dictPerformance[key] += [parms[key]]
    settings += [parms["settings"]]
    methods += [method]

print ""
#print df.sort_values(by=["rank","ROCint_test"])
df = pd.DataFrame(dictPerformance)
n = df.shape[0]
r = 0.4*n
df["rank_Oes"] = df.ES_diffRel.rank(ascending=False)
df["rank_Roc"] = df.ROCint_test.rank()
df["rank_Es"] = df.ES_test.rank()
df["rank_ORoc"] = df.ROCint_diffRel.rank(ascending=False)
df["rank_KS"] = df.KS_S.rank()
#df = df[(df.rank_Oes > r) & (df.rank_ORoc > r)]
df = df[(df.ES_diffRel < 7.0) & (df.KS_S > 0.1) & (df.ROCint_diffRel < 4.2)]
df["rank"] = 2*df.rank_Roc + 2*df.rank_Es + df.rank_Oes + df.rank_ORoc + df.rank_KS
#winner = df.sort_values(by=["rank","ROCint_test"],ascending=False)[:10]
winner = df.sort_values(by=["ROCint_test"],ascending=False)[:10]
idxes = winner.index.values
print winner        
for idx in idxes:
    print(str(idx)+" "+settings[idx])
#print dictAll[methods[idx]]
