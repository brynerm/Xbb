#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import ROOT
ROOT.gROOT.SetBatch(True)
from myutils import BetterConfigParser
from myutils.pandasConverter import SampleTreesToDataFrameConverter
import resource
from myutils.NewRebinner import Rebinner
import os
import sys
import pickle
import glob
import shutil
import numpy as np

# read arguments
# TODO: option for singe region
argv = sys.argv
parser = OptionParser()
parser.add_option("-C", "--config", dest="config", default=[], action="append",
                      help="configuration file")
parser.add_option("-T", "--tag", dest="tag", default='',
                      help="configuration tag")
parser.add_option("-t","--trainingRegions", dest="trainingRegions", default='',
                      help="cut region identifier")
parser.add_option("-l","--load", dest="loadHDF", action="store_const", default=False, const=True,
                      help="load hdf files")
parser.add_option("-p","--plot", dest="plot", action="store_const", default=False, const=True,
                      help="plot fit curves")
parser.add_option("--force", dest="force", action="store_const", default=False, const=True,
                      help="force reloading samples (use it when cuts have changed)")
(opts, args) = parser.parse_args(argv)
if opts.config =="":
        opts.config = ["config"]

# Import after configure to get help message
if len(opts.tag.strip()) > 1:
    config = BetterConfigParser()
    config.read("{tag}config/paths.ini".format(tag=opts.tag))
    configFiles = config.get("Configuration", "List").split(' ')
    opts.config = ["{tag}config/{file}".format(tag=opts.tag, file=x.strip()) for x in configFiles]
    print("reading config files:", opts.config)

# load config
config = BetterConfigParser()
config.read(opts.config)

converter = SampleTreesToDataFrameConverter(config,config_name=opts.tag) 


if not opts.loadHDF:
    converter.loadSamples(safe_hdf=True,force=opts.force)

dfs = converter.getDataFrames()

config_updates = {}
sig = 0
sig_old = 0

for region,df_dict in dfs.iteritems():
    print("\n_______________________________________________________\nComputing bin edges for region: %s"%region)
    bin_list = None
    dc = converter.dcMakers[region]
    mva_min = dc.binning["minX"]
    mva_max = dc.binning["maxX"]
    n_bins = dc.binning["nBinsX"]    
    binner = Rebinner(mva_min,mva_max,mva_var=dc.treevar,make_plots=opts.plot,plot_dir="binner_plots/%s_%s/"%(region,dc.treevar.split(".")[0]))

    try:
        if opts.loadHDF:
            filename = "dumps/%s_%s_%s.hdf"%(opts.tag,region,dc.treevar.split(".")[0])
            print("INFO: Load HDF file: %s"%filename)
            binner.load_hdf(filename)
        else:
            binner.prepare(df_dict['SIG'],df_dict['BKG'])
        bin_list = binner.getBinEdges(n_bins)
        sig += binner.EvalSign(bin_list)**2
        sig_old += binner.EvalSign(np.linspace(mva_min,mva_max,n_bins))**2
    except:
        print ("ERROR: Could not fit ROC curve. Set rebin method to default")

    if bin_list is None:
        print ("ERROR: Binning fit did not converge. Set rebin method to default")
        rebin_method = "default"
        rebin_list = "[]"
    else:
        rebin_list = "[" + ", ".join(str(np.round(x,4)) for x in bin_list) + "]"
        rebin_method = "fixed"
        print ("rebin_list set to " + rebin_list + ", rebin_method set to 'fixed'")
    
    config_updates[region] = {"rebin_list":rebin_list, "rebin_method":rebin_method}

print("\n *** Significance of equally spaced bins: %s ***"%str(np.round(np.sqrt(sig_old),3)))
print(" *** Significance of new bins: %s ***"%str(np.round(np.sqrt(sig),3)))
print(" *** relative difference: %s %% ***"%str(np.round(100*(np.sqrt(sig)-np.sqrt(sig_old))/np.sqrt(sig_old),2)))

#TODO: automatic update
print("\n================================\n\nupdate your config!\n")
for region, update_dict in config_updates.iteritems():
    print("[dc:%s]"%region)
    for key, val in update_dict.iteritems():
        print("%s: %s"%(key,val))
    print("")

