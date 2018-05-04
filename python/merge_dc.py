#!/usr/bin/env python
from __future__ import print_function
import os, sys, ROOT, warnings, pickle
ROOT.gROOT.SetBatch(True)
from array import array
from math import sqrt
from copy import copy, deepcopy
#suppres the EvalInstace conversion warning bug
warnings.filterwarnings( action='ignore', category=RuntimeWarning, message='creating converter.*' )
from optparse import OptionParser
from myutils import BetterConfigParser, Sample, progbar, printc, ParseInfo, Rebinner, HistoMaker
import re
import json
import glob
from myutils import NewTreeCache as TreeCache
from myutils.sampleTree import SampleTree as SampleTree
from myutils.Datacard import Datacard
from myutils.BranchList import BranchList
from myutils.FileList import FileList

class MergeDatacards(object):

    # initialize datacard config
    def __init__(self, config, region):
        self.config = config
        self.region = region
        self.dcMaker = Datacard(config=self.config, region=region)
   
    def getTextFileName(self, sampleIdentifier):
        return self.dcMaker.getDatacardBaseName(sampleIdentifier)+'.txt'

    def getHistogramFileName(self, sampleIdentifier):
        return self.dcMaker.getDatacardBaseName(sampleIdentifier)+'.root'
    
    def checkFiles(self, fileNames):
        filesComplete = True
        for textFilePath in fileNames:
            textFileName = textFilePath.split('/')[-1]
            if os.path.isfile(textFilePath):
                print("[\x1b[42m\x1b[97mOK\x1b[0m]", textFileName)
            else:
                print("[\x1b[41m\x1b[97mNOT FOUND\x1b[0m]", textFilePath)
                filesComplete = False
        return filesComplete

    # check if all the split datacard .txt and .root files are existing
    def prepare(self):
        samples = Datacard.getSamples(config=self.config, regions=[self.region])
        sampleIdentifiers = sorted(list(set([sample.identifier for sample in samples])))
        self.histogramFilePaths = [self.getHistogramFileName(sampleIdentifier) for sampleIdentifier in sampleIdentifiers]
        filesComplete = self.checkFiles(fileNames=self.histogramFilePaths)
        return self if filesComplete else None
    
    # load all datacard histograms from rundc step and write output histograms+txt files
    def run(self):
        self.dcMaker.load()
        self.dcMaker.writeDatacards()

if __name__ == "__main__":
    # read arguments
    argv = sys.argv
    parser = OptionParser()
    parser.add_option("-t", "--regions", dest="regions", default='',
                          help="cut regions identifier")
    parser.add_option("-C", "--config", dest="config", default=[], action="append",
                          help="configuration file")
    (opts, args) = parser.parse_args(argv)
    if opts.config == "":
            opts.config = "config"

    # Import after configure to get help message
    from myutils import BetterConfigParser, mvainfo, ParseInfo

    # load config
    config = BetterConfigParser()
    config.read(opts.config)

    # if no region is given in argument, run it for all of them
    regionsListString = opts.regions if len(opts.regions.strip())>0 else config.get('LimitGeneral', 'List')
    regions = [x.strip() for x in regionsListString.split(',') if len(x.strip()) > 0]
    for region in regions:
        mergeDC = MergeDatacards(config=config, region=region)
        isComplete = mergeDC.prepare()
        if isComplete:
            mergeDC.run()
        else:
            raise Exception("InputDcIncomplete")
