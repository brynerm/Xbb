#!/usr/bin/env python
import sys
import os
import ROOT
ROOT.gROOT.SetBatch(True)
from optparse import OptionParser
from myutils.FileList import FileList
from myutils import BetterConfigParser, ParseInfo, LeptonSF
from myutils.FileLocator import FileLocator
import importlib
from myutils.sampleTree import SampleTree
from myutils.VtypeCorrector import VtypeCorrector
from myutils.AdditionalJetIndex import AdditionalJetIndex
from myutils.TTWeights import TTWeights
from myutils.EWKweights import EWKweights
from myutils.BTagWeights import BTagWeights
from myutils.LeptonWeights import LeptonWeights
from myutils.JetEnergySystematics import JetEnergySystematics
from myutils.WPtReweight import WPtReweight
import resource
from myutils.DYspecialWeight import DYspecialWeight
from myutils.PerSampleWeight import PerSampleWeight
from myutils.Skim import Skim

argv = sys.argv
parser = OptionParser()
parser.add_option("-S", "--sampleIdentifier", dest="sampleIdentifier", default="", 
                      help="samples you want to run on")
parser.add_option("-C", "--config", dest="config", default=[], action="append",
                      help="configuration defining the plots to make")
parser.add_option("-f", "--fileList", dest="fileList", default="",
                              help="list of files you want to run on")
parser.add_option("-b", "--addCollections", dest="addCollections", default="",
                              help="collections to add: vtype")
parser.add_option("-F", "--force", dest="force", action="store_true", help="overwrite existing files", default=False)
(opts, args) = parser.parse_args(argv)
if opts.config == "":
        opts.config = "config"

filelist = FileList.decompress(opts.fileList) if len(opts.fileList) > 0 else None
print "len(filelist)",len(filelist),
if len(filelist) > 0:
    print "filelist[0]:", filelist[0]
else:
    print ''

debug = 'XBBDEBUG' in os.environ
config = BetterConfigParser()
config.read(opts.config)
anaTag = config.get("Analysis","tag")
TrainFlag = eval(config.get('Analysis','TrainFlag'))
btagLibrary = config.get('BTagReshaping','library')
samplesinfo=config.get('Directories','samplesinfo')
channel=config.get('Configuration','channel')
VHbbNameSpace=config.get('VHbbNameSpace','library')
ROOT.gSystem.Load(VHbbNameSpace)
pathIN = config.get('Directories','SYSin')
pathOUT = config.get('Directories','SYSout')
tmpDir = config.get('Directories','scratch')
print 'INput samples:\t%s'%pathIN
print 'OUTput samples:\t%s'%pathOUT

fileLocator = FileLocator(config=config)

# samples
info = ParseInfo(samplesinfo, pathIN)
matchingSamples = [x for x in info if x.identifier==opts.sampleIdentifier and not x.subsample]
if len(matchingSamples) != 1:
    print "need exactly 1 sample identifier as input with -S !!"
    print matchingSamples
    exit(1)
sample = matchingSamples[0]

# TODO: 
collections = [x.strip() for x in opts.addCollections.split(',') if len(x.strip()) > 0] if len(opts.addCollections.strip())>0  else []
if len(collections) < 1:
    print "\x1b[31mWARNING: no collections added! Specify the collections to add with the --addCollections option!\x1b[0m"
print 'collections to add:', collections


for fileName in filelist:
    localFileName = fileLocator.getFilenameAfterPrep(fileName)
    inputFileName = "{path}/{subfolder}/{filename}".format(path=pathIN, subfolder=sample.identifier, filename=localFileName)
    outputFileName = "{path}/{subfolder}/{filename}".format(path=pathOUT, subfolder=sample.identifier, filename=localFileName)
    tmpFileName = "{path}/{subfolder}/{filename}".format(path=tmpDir, subfolder=sample.identifier, filename=localFileName)
    outputFolder = '/'.join(outputFileName.split('/')[:-1])
    tmpFolder = '/'.join(tmpFileName.split('/')[:-1])
    fileLocator.makedirs(tmpFolder)
    fileLocator.makedirs(outputFolder)

    if not fileLocator.exists(outputFileName) or opts.force:
        # load sample tree and initialize vtype corrector
        sampleTree = SampleTree([inputFileName], config=config)
        if not sampleTree.tree:
            # try original naming scheme if reading directly from Heppy/Nano ntuples (without prep)
            fileNameOriginal = pathIN + '/' + fileName
            print "FO:", fileNameOriginal
            xrootdRedirector = fileLocator.getRedirector(fileNameOriginal)
            sampleTree = SampleTree([fileNameOriginal], config=config, xrootdRedirector=xrootdRedirector)
            if not sampleTree.tree:
                print "\x1b[31mERROR: file does not exist or is broken, will be SKIPPED!\x1b[0m"
                continue

        # lists of single modules can be given instead of a module, "--addCollections Sys.all"
        # [Sys]
        # all = ['Sys.Vtype', 'Sys.Leptons', ...]
        collectionsListsReplaced = []
        for collection in collections:
            if '.' in collection:
                section = collection.split('.')[0]
                key = collection.split('.')[1]
                listExpression = config.get(section, key).strip()
                if listExpression.startswith('[') and listExpression.endswith(']'):
                    listParsed = eval(listExpression)
                    for i in listParsed:
                        collectionsListsReplaced.append(i)
                else:
                    collectionsListsReplaced.append(collection)
            else:
                collectionsListsReplaced.append(collection)
        collections = collectionsListsReplaced

        # to use this syntax, use "--addCollections Sys.Vtype" for a config file entry like this:
        # [Sys]
        # Vtype = VtypeCorrector.VtypeCorrector(channel='Zll')
        # (instead of passing the tree in the constructor, the setTree method can be used)
        for collection in collections:
            if '.' in collection:
                section = collection.split('.')[0]
                key = collection.split('.')[1]
                pyCode = config.get(section, key)

                # import module from myutils
                moduleName = pyCode.split('(')[0].split('.')[0].strip()
                if debug:
                    print "DEBUG: import module:", moduleName
                    print("\x1b[33mDEBUG: " + collection + ": run PYTHON code:\n"+pyCode+"\x1b[0m")
                globals()[moduleName] = importlib.import_module(".{module}".format(module=moduleName), package="myutils")

                # get object
                wObject = eval(pyCode)

                # pass the tree and other variables if needed to finalize initialization
                if hasattr(wObject, "customInit") and callable(getattr(wObject, "customInit")):
                    wObject.customInit({'config': config,
                                        'sampleTree': sampleTree,
                                        'tree': sampleTree.tree,
                                        'sample': sample,
                                        'channel': channel,
                                        })

                # add callbacks if the objects provides any
                if hasattr(wObject, "processEvent") and callable(getattr(wObject, "processEvent")):
                    sampleTree.addCallback('event', wObject.processEvent)

                # add branches
                if hasattr(wObject, "getBranches") and callable(getattr(wObject, "getBranches")):
                    sampleTree.addOutputBranches(wObject.getBranches())

        # TODO: this can also be made a separate module
        if 'addbranches' in collections:
            writeNewVariables = eval(config.get("Regression", "writeNewVariablesDict"))
            sampleTree.addOutputBranches(writeNewVariables)

        if 'removebranches' in collections:
            bl_branch = eval(config.get('Branches', 'useless_branch'))
            for br in bl_branch:
                sampleTree.addBranchToBlacklist(br)
            bl_branch = eval(config.get('Branches', 'useless_after_sys'))
            for br in bl_branch:
                sampleTree.addBranchToBlacklist(br)

        # define output file 
        sampleTree.addOutputTree(tmpFileName, cut='1', branches='*')
        sampleTree.process()

        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # copy temporary file to output folder
        if opts.force and fileLocator.exists(outputFileName):
            fileLocator.rm(outputFileName)

        try:
            fileLocator.cp(tmpFileName, outputFileName)
        except Exception as e:
            print "\x1b[31mERROR: copy from scratch to final destination failed!!\x1b[0m"
            print e

        try:
            fileLocator.rm(tmpFileName)
        except Exception as e:
            print "ERROR: could not delete file on scratch!"
            print e

        print 'copy ', tmpFileName, outputFileName

    else:
        print 'SKIP:', localFileName

