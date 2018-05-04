#!/usr/bin/env python
from __future__ import print_function
import sys
import ROOT
#suppres the EvalInstace conversion warning bug
import warnings
warnings.filterwarnings( action='ignore', category=RuntimeWarning, message='creating converter.*' )
from optparse import OptionParser
import pickle
from myutils.FileList import FileList
from myutils.FileLocator import FileLocator
from myutils.sampleTree import SampleTree as SampleTree

#CONFIGURE
ROOT.gROOT.SetBatch(True)
argv = sys.argv
parser = OptionParser()
parser.add_option("-D", "--discr", dest="discr", default="",
                      help="discriminators to be added")
parser.add_option("-S", "--sampleIdentifier", dest="sampleIdentifier", default="",
                      help="samples you want to run on")
parser.add_option("-C", "--config", dest="config", default=[], action="append",
                      help="configuration file")
parser.add_option("-W", "--weight", dest="weight", default='',
                      help="list of weights, used when performing the optimisation")
parser.add_option("-f", "--fileList", dest="fileList", default="",
                              help="list of files you want to run on")
parser.add_option("-o","--force", action="store_true", dest="force", default=False,
                      help="force overwriting of already cached files")
(opts, args) = parser.parse_args(argv)

if opts.config == "":
        opts.config = "config"

weight = opts.weight
evaluate_optimisation = False
if weight != '':
    evaluate_optimisation = True

#Import after configure to get help message
from myutils import BetterConfigParser, ParseInfo, MvaEvaluator

config = BetterConfigParser()
config.read(opts.config)
anaTag = config.get("Analysis", "tag")

fileLocator = FileLocator(config=config)
print ("OPTS", opts)
if len(opts.fileList) > 0:
    filelist = FileList.decompress(opts.fileList) if len(opts.fileList) > 0 else None
    print ("len(filelist)", len(filelist))
    if len(filelist) > 0:
        print ("filelist[0]:", filelist[0])
else:
    filelist = SampleTree({'name': opts.sampleIdentifier, 'folder': config.get('Directories', 'MVAin')}, countOnly=True, splitFilesChunkSize=-1, config=config).getSampleFileNameChunks()[0]
    print ("INFO: no file list given, use all files!")
    print (len(filelist), filelist)

#get locations:
Wdir = config.get('Directories', 'Wdir')
samplesinfo = config.get('Directories', 'samplesinfo')

#read shape systematics
systematics = config.get('systematics', 'systematics')

INpath = config.get('Directories', 'MVAin')
OUTpath = config.get('Directories', 'MVAout')
tmpDir = config.get('Directories', 'scratch')

info = ParseInfo(samplesinfo,INpath)

arglist = ''

if not evaluate_optimisation:
    arglist = opts.discr #RTight_blavla,bsbsb
else:
#    print '@INFO: Evaluating bdt for optimisation'
    arglist = weight

namelistIN = opts.sampleIdentifier
namelist = namelistIN.split(',')
print ('namelist', namelist)

#doinfo=bool(int(opts.update))

MVAlist = arglist.split(',')

#CONFIG
#factory
factoryname = config.get('factory','factoryname')

#load the namespace
VHbbNameSpace = config.get('VHbbNameSpace','library')
ROOT.gSystem.Load(VHbbNameSpace)

#MVA
MVAinfos = []
MVAdir = config.get('Directories','vhbbpath')
for MVAname in MVAlist:
    MVAinfofile = open(MVAdir+'/python/weights/'+factoryname+'_'+MVAname+'.info', 'r')
    MVAinfos.append(pickle.load(MVAinfofile))
    MVAinfofile.close()

#Workdir
workdir = ROOT.gDirectory.GetPath()

theMVAs = []
for mva in MVAinfos:
    theMVAs.append(MvaEvaluator(config,mva))

# samples
matchingSamples = [x for x in info if x.identifier==opts.sampleIdentifier and not x.subsample]
if len(matchingSamples) != 1:
    print ("need exactly 1 sample identifier as input with -S !!", matchingSamples)
    exit(1)
sample = matchingSamples[0]

for fileName in filelist:
    localFileName = fileName.split('/')[-1] #TODO! # fileLocator.getFilenameAfterPrep(fileName)
    inputFileName = "{path}/{subfolder}/{filename}".format(path=INpath, subfolder=sample.identifier, filename=localFileName)
    outputFileName = "{path}/{subfolder}/{filename}".format(path=OUTpath, subfolder=sample.identifier, filename=localFileName)
    tmpFileName = "{path}/{subfolder}/{filename}".format(path=tmpDir, subfolder=sample.identifier, filename=localFileName)
    outputFolder = '/'.join(outputFileName.split('/')[:-1])
    tmpFolder = '/'.join(tmpFileName.split('/')[:-1])
    fileLocator.makedirs(tmpFolder)
    fileLocator.makedirs(outputFolder)
    if not fileLocator.isValidRootFile(outputFileName) or opts.force:
        # load sample tree
        sampleTree = SampleTree([inputFileName], config=config)
        if not sampleTree.tree:
            print ("\x1b[31mERROR: file does not exist or is broken, will be SKIPPED!\x1b[0m")
            continue
        # Set branch adress for all vars
        for i in range(0, len(theMVAs)):
            theMVAs[i].setVariables(sampleTree.tree, sample)
        mvaBranches = []
        for i in range(0, len(theMVAs)):
            mvaBranches.append({
                    'name': MVAinfos[i].MVAname,
                    'length': len(systematics.split()),
                    'formula': theMVAs[i].evaluate,
                    'leaflist': ':'.join(systematics.split())+'/F',
                    # force 'srray-style' filling = passing the pointer to the array to the function instead of using the return value, even when the branch is a scalar, e.g. when only nominal systematic is selected
                    'arrayStyle': True,
                })

        sampleTree.addOutputBranches(mvaBranches)
        print("branches:", mvaBranches)

        # define output file
        sampleTree.addOutputTree(tmpFileName, cut='1', branches='*')
        sampleTree.process()

        # copy temporary file to output folder
        if opts.force and fileLocator.exists(outputFileName):
            fileLocator.rm(outputFileName)

        if fileLocator.isValidRootFile(tmpFileName):
            fileLocator.cp(tmpFileName, outputFileName)
            #fileLocator.rm(tmpFileName)
            print ("copy: ", tmpFileName, " --> ", outputFileName)
            if fileLocator.isValidRootFile(outputFileName):
                fileLocator.rm(tmpFileName)
            else:
                print("TMP:", tmpFileName)
                print("OUT:", outputFileName)
                print("\x1b[31mERROR: copy from tmp to output failed!\x1b[0m")
        else:
            print("\x1b[31mERROR: temporary file is zombie or disappeared!", tmpFileName ,"\x1b[0m")
    else:
        print("SKIP: ", localFileName)
