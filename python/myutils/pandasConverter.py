#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import ROOT
ROOT.gROOT.SetBatch(True)
import NewTreeCache as TreeCache
from sampleTree import SampleTree as SampleTree
from Datacard import Datacard
from sample_parser import ParseInfo
import resource
import sys
import pickle
import glob
import shutil
import numpy as np
import pandas as pd

class SampleTreesToDataFrameConverter(object):

    def __init__(self, config):
        self.config = config
        self.regions = [x.strip() for x in config.get('LimitGeneral', 'List_for_rebinner').split(',') if len(x.strip()) > 0]
        self.dcMakers = {region: Datacard(config=self.config, region=region) for region in self.regions}
        self.dfs = {region: {} for region in self.regions} 



    def loadSamples(self):
        for region in self.regions:
            dcMaker = self.dcMakers[region]
            for sampleType in ['SIG','BKG']:
                samples = dcMaker.samples[sampleType]
                treevar = dcMaker.treevar
                weightF = dcMaker.weightF
                EvalCut = dcMaker.EvalCut
                df = pd.DataFrame({treevar: [], 'weight': []})         
                for i, sample in enumerate(samples): 
                    print("INFO: Add ",sampleType," sample ", i, " of ", len(samples))
                    # get sample tree from cache
                    tc = TreeCache.TreeCache(
                            sample=sample,
                            cutList=dcMaker.getCacheCut(sample),
                            inputFolder=dcMaker.path,
                            config=self.config,
                            debug=False
                        )
                    if not tc.isCached():
                        print("\x1b[31m:ERROR not cached! run cachedc step again\x1b[0m")
                        raise Exception("NotCached")
                    sampleTree = tc.getTree()
           
                    if sampleTree:
                        treeScale = sampleTree.getScale(sample) * 2.0
                        print ('scale:', treeScale)
                        
                        # initialize numpy array
                        nSamples = sampleTree.GetEntries()
                        input_treevar = []
                        input_weight = []
                        sampleTree.addFormula(treevar)
                        sampleTree.addFormula(weightF)
                        sampleTree.addFormula(EvalCut)
                        
                        # fill numpy array from ROOT tree
                        for i, event in enumerate(sampleTree):
                            if sampleTree.evaluate(EvalCut):
                                input_treevar.append(sampleTree.evaluate(treevar))
                                input_weight.append(sampleTree.evaluate(weightF)*treeScale)
                                # total weight comes from weightF (btag, lepton sf, ...) and treeScale to scale MC to x-section

                        sampleDf = pd.DataFrame({treevar: input_treevar, 'weight': input_weight})
                        df = df.append(sampleDf,ignore_index=True)

                    else:
                        print ("\x1b[31mERROR: TREE NOT FOUND:", sample.name, " -> not cached??\x1b[0m")
                        raise Exception("CachedTreeMissing")
        

                self.dfs[region][sampleType] = df
    def makeHDF(self,path="./"):
        for region, dataframes in self.dfs.iteritems():
            for dataType, df in dataframes.iteritems():
                df.to_hdf(path+"/"+region+".hdf",dataType)



