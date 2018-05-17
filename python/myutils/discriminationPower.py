import ROOT
import numpy as np
import pandas as pd

import argparse
import os 

parser = argparse.ArgumentParser(description='read file')
parser.add_argument('filename', metavar='file1', type=str, nargs='+',
                           help='root file')
parser.add_argument('--showBins',action='store_const', const=True, default=False, dest='showBins', help='show all bin entries')
parser.add_argument('--showLogs', action='store_const', const=True, default=False, dest='showLogs', help='show full Logs')
parser.add_argument('--total', action='store_const', const=True, default=False, dest='total', help='print total number of entries in regions')
parser.add_argument('--range', action='store_const', const=True, default=False, dest='range', help='set range around peak')
args=parser.parse_args()

def binToArray(tree,name = "tree"):
    n = 120
    offset = 1
    array = np.zeros(n)
    if args.showLogs:
        print "read "+name
    for b in range(0,n):
        array[b] = tree.GetBinContent(b+offset)
    return array

trees = ["ggZH_hbb", "ZH_hbb", "Zj0b", "Zj1b", "Zj2b", "TT", "VVLF", "s_Top", "VVHF"]
trees_dict = {
        "ZH_hbb": "",
        "ggZH_hbb": "ZHbb",
        "Zj0b": "Z_udscg",
        "Zj1b": "Zb",
        "Zj2b": "Zbb",
        "TT":"TT",
        "VVLF": "VVlight",
        "VVHF": "VV2b",
        "s_Top":"ST"
        }
dfs = {}
for filename in args.filename:

    f = os.getcwd() + "/" + filename
    directory = filename[8:-5]+"/"
    if args.showLogs:
        print "file: " + filename
        print "directory: " + directory
    dc = ROOT.TFile.Open(f, 'read')

    df = pd.DataFrame()
    for tree in trees:
        t = dc.Get(directory + tree)
        if t:
    #        t.Print()
            df[tree] = binToArray(t,tree)
        else:
            t = dc.Get(trees_dict[tree])
            if t:
                df[tree] = binToArray(t,tree)
            else:
                print "missing hist: "+tree
                df[tree] = np.zeros(len(df.index))


    df["signal"] = df.ggZH_hbb + df.ZH_hbb
    df["background"] = df.Zj0b + df.Zj1b + df.Zj2b + df.TT + df.VVLF + df.s_Top + df.VVHF
    dfs[filename] = df

zll_list = {}
for filename in dfs:
    name = filename.split("BDT_")[1]
    if name in zll_list:
        zll_list[name].append(filename)
        if len(zll_list[name])>2:
            print "ERROR: more than 2 DC called"+name
    else:
        zll_list[name] = [filename]
zll = {}
for name, pair in zll_list.iteritems():
    df = sum(dfs[i] for i in pair) 
    print name
    if args.showBins or args.showLogs:
        print df
    zll[name] = df
    if args.total or args.showLogs:
        print "total number:"
        print df.sum(axis=0)
    df["disc"] = np.sqrt(sum(dfs[i].signal**2 / (dfs[i].signal + dfs[i].background) for i in pair))
    if args.range:
        i_max = df.signal.idxmax()
        bounds = np.array([(i_max-15),(i_max+15)])
        cutted = np.arange(*bounds)
    else:
        cutted = df.index
    print "discrimination power: " + str(np.sqrt(np.nansum(df.disc[cutted] ** 2)))
    print ""

panel = pd.Panel(zll)
#print panel
disc = panel.minor_xs("disc")
#if args.showLogs:
#    print disc
power = np.sqrt(np.nansum(disc**2))
print "total discrimination power: " + str(power)
