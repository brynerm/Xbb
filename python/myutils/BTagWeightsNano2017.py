#!/usr/bin/env python
import ROOT
import numpy as np
import array
import os

class Jet:
    def __init__(self, pt, eta, fl, csv) :
        self.pt = pt
        self.eta = eta
        self.hadronFlavour = fl
        self.csv = csv

# example how to included it in config for sys step:
# BTagWeights = BTagWeightsNano2017.BTagWeights(calibName='DeepCSV',calibFile='DeepCSV_94XSF_V1_B_F.csv')

class BTagWeights(object):

    def __init__(self, calibName, calibFile, method="iterativefit", branchName=None, jetBtagBranchName="Jet_btagDeepB", includeFixPtEtaBins=False, jetPtBranchName="Jet_Pt"):
        self.jetPtBranchName = jetPtBranchName
        self.method = method
        self.calibName = calibName
        self.includeFixPtEtaBins = includeFixPtEtaBins
        self.branchBaseName = branchName if branchName else "bTagWeight"+calibName
        self.jetBtagBranchName = jetBtagBranchName
        self.lastEntry = -1
        self.branchBuffers = {}
        self.branches = []
        self.systematics = ["central", "up_jes", "down_jes", "up_lf", "down_lf", "up_hf", "down_hf", "up_hfstats1", "down_hfstats1", "up_hfstats2", "down_hfstats2", "up_lfstats1", "down_lfstats1", "up_lfstats2", "down_lfstats2", "up_cferr1", "down_cferr1", "up_cferr2", "down_cferr2"]

        print 'load BTagCalibrationStandalone...'
        self.btagCalibratorFileName = "../interface/BTagCalibrationStandalone_cpp.so"
        if os.path.isfile(self.btagCalibratorFileName):
            ROOT.gSystem.Load(self.btagCalibratorFileName)
        else:
            print "\x1b[31m:ERROR: BTagCalibrationStandalone not found! Go to Xbb directory and run 'make'!\x1b[0m"
            raise Exception("BTagCalibrationStandaloneNotFound")
        
        print 'load bTag CSV files...'
        calib = ROOT.BTagCalibration(calibName, calibFile)
        self.btag_calibrators = {}
        for syst in self.systematics:
            print "[btagSF]: Loading calibrator for algo:", self.calibName, "systematic:", syst
            self.btag_calibrators[self.calibName+"_iterative_"+syst] = ROOT.BTagCalibrationReader(3, syst)
            for fl in range(3):
                self.btag_calibrators[self.calibName+"_iterative_"+syst].load(calib, fl, self.method)

        # map of calibrators. E.g. btag_calibrators["CSVM_nominal_bc"], btag_calibrators["CSVM_up_l"], ...
        self.sysMap = {
                    "JESUp": "up_jes",
                    "JESDown": "down_jes",
                    "LFUp": "up_lf",
                    "LFDown": "down_lf",
                    "HFUp": "up_hf",
                    "HFDown": "down_hf",
                    "HFStats1Up": "up_hfstats1",
                    "HFStats1Down": "down_hfstats1",
                    "HFStats2Up": "up_hfstats2",
                    "HFStats2Down": "down_hfstats2",
                    "LFStats1Up": "up_lfstats1",
                    "LFStats1Down": "down_lfstats1",
                    "LFStats2Up": "up_lfstats2",
                    "LFStats2Down": "down_lfstats2",
                    "cErr1Up": "up_cferr1",
                    "cErr1Down": "down_cferr1",
                    "cErr2Up": "up_cferr2",
                    "cErr2Down": "down_cferr2",
                }
        print 'INFO: bTag initialization done.'

        # initialize buffers for new branches 
        self.branchBuffers[self.branchBaseName] = array.array('f', [0])
        self.branches.append({'name': self.branchBaseName, 'formula': self.getBranch, 'arguments': self.branchBaseName})
        for syst in ["JES", "LF", "HF", "LFStats1", "LFStats2", "HFStats1", "HFStats2", "cErr1", "cErr2"]:
            for sdir in ["Up", "Down"]:
                branchName = self.branchBaseName + "_" + syst + sdir
                self.branchBuffers[branchName] = array.array('f', [0])
                self.branches.append({'name': branchName, 'formula': self.getBranch, 'arguments': branchName})

                if self.includeFixPtEtaBins:
                    for ipt in range(0,5):
                        for ieta in range(1,4):
                            branchName = self.branchBaseName + "_"+syst+"_pt"+str(ipt)+"_eta"+str(ieta)+sdir 
                            self.branchBuffers[branchName] = array.array('f', [0])
                            self.branches.append({'name': branchName, 'formula': self.getBranch, 'arguments': branchName})

    def customInit(self, initVars):
        sample = initVars['sample']
        self.isData = sample.type == 'DATA'
        if self.isData:
            self.branches = []

    def getBranches(self):
        return self.branches

    # read from buffers which have been filled in processEvent()    
    def getBranch(self, event, arguments=None):
        self.processEvent(event)
        if arguments:
            return self.branchBuffers[arguments][0]

    # read from buffers which have been filled in processEvent()    
    def getVectorBranch(self, event, arguments=None, destinationArray=None):
        self.processEvent(event)
        for i in range(arguments['length']):
            destinationArray[i] =  self.branchBuffers[arguments['branch']][i]

    # depending on flavour, only a sample of systematics matter
    def applies(self, flavour, syst):
        if flavour==5 and syst not in ["central", "up_jes", "down_jes",  "up_lf", "down_lf",  "up_hfstats1", "down_hfstats1", "up_hfstats2", "down_hfstats2"]:
            return False
        elif flavour==4 and syst not in ["central", "up_cferr1", "down_cferr1", "up_cferr2", "down_cferr2" ]:
            return False
        elif flavour==0 and syst not in ["central", "up_jes", "down_jes", "up_hf", "down_hf",  "up_lfstats1", "down_lfstats1", "up_lfstats2", "down_lfstats2" ]:
            return False
        return True

    # function that reads the SF
    def get_SF(self, pt=30., eta=0.0, fl=5, val=0.0, syst="central", algo="CSV", wp="M", shape_corr=False):
        
        #if eta<0:
        #    eta=-eta

        # no SF for pT<20 GeV or pt>1000 or abs(eta)>2.4
        if abs(eta)>2.4 or pt>1000. or pt<20.:
            return 1.0

        # the .csv files use the convention: b=0, c=1, l=2. Convert into hadronFlavour convention: b=5, c=4, f=0
        fl_index = min(-fl+5,2)
        # no fl=1 in .csv for CMVAv2 (a bug???)
        #if not shape_corr and "CMVAV2" in algo and fl==4:
        #    fl_index = 0

        if shape_corr:
            if self.applies(fl,syst):
                sf = self.btag_calibrators[algo+"_iterative_"+syst].eval(fl_index ,eta, pt, val)
                #print 'shape_corr SF:',fl_index ,eta, pt, val, "=>", sf
                if sf < 0.01 and fl_index==0:
                    print 'sf is 0 for:', fl_index ,eta, pt, val, algo+"_iterative_"+syst

                return sf
            else:
                sf = self.btag_calibrators[algo+"_iterative_central"].eval(fl_index ,eta, pt, val)
                #print 'shape_corr for central SF:', fl_index ,eta, pt, val, "=>", sf
                return sf


        # pt ranges for bc SF: needed to avoid out_of_range exceptions
        pt_range_high_bc = 10000 #670.-1e-02 if "CSV" in algo else 320.-1e-02
        pt_range_low_bc = 20 #30.+1e-02

        # b or c jets
        if fl>=4:
            # use end_of_range values for pt in [20,30] or pt in [670,1000], with double error
            out_of_range = False
            if pt>pt_range_high_bc or pt<pt_range_low_bc:
                out_of_range = True
            pt = min(pt, pt_range_high_bc)
            pt = max(pt, pt_range_low_bc)
            sf = self.btag_calibrators[algo+wp+"_"+syst+"_bc"].eval(fl_index ,eta, pt)
            # double the error for pt out-of-range
            if out_of_range and syst in ["up","down"]:
                sf = max(2*sf - self.btag_calibrators[algo+wp+"_central_bc"].eval(fl_index ,eta, pt), 0.)
            return sf
        # light jets
        else:
            sf = self.btag_calibrators[algo+wp+"_"+syst+"_l"].eval( fl_index ,eta, pt)
            return  sf

    def get_event_SF(self, ptmin, ptmax, etamin, etamax, jets=[], syst="central", algo="CSV"):
        weight = 1.0

        #print 'gonna add the jet SF'
        for jet in jets:
            #print 'ptmin', ptmin, 'ptmax', ptmax, 'etamin', etamin, 'etamax', etamax
            if (jet.pt > ptmin and jet.pt < ptmax and abs(jet.eta) > etamin and abs(jet.eta) < etamax):
                #print syst, '!', self.get_SF(pt=jet.pt, eta=jet.eta, fl=jet.hadronFlavour, val=jet.csv, syst=syst, algo=algo, wp="", shape_corr=True), jet.pt, 'jet: pt', jet.pt, 'eta', jet.eta, 'fl', jet.hadronFlavour
                weight *= self.get_SF(pt=jet.pt, eta=jet.eta, fl=jet.hadronFlavour, val=jet.csv, syst=syst, algo=algo, wp="", shape_corr=True)
            else:
                #print 'central !',self.get_SF(pt=jet.pt, eta=jet.eta, fl=jet.hadronFlavour, val=jet.csv, syst="central", algo=algo, wp="", shape_corr=True), jet.pt
                weight *= self.get_SF(pt=jet.pt, eta=jet.eta, fl=jet.hadronFlavour, val=jet.csv, syst="central", algo=algo, wp="", shape_corr=True)
        #print '--->',weight
        return weight
    
    # compute all the btag weights
    def processEvent(self, tree):
        isGoodEvent = True
        currentEntry = tree.GetReadEntry()
        # if current entry has not been processed yet
        if currentEntry != self.lastEntry and not self.isData:
            self.lastEntry = currentEntry

            jets_cmva = []
            treeJet_Pt = getattr(tree, self.jetPtBranchName)
            for i in range(tree.nJet):
                if (tree.Jet_bReg[i]*treeJet_Pt[i]/tree.Jet_pt[i] > 20 and abs(tree.Jet_eta[i]) < 2.4 and tree.Jet_lepFilter[i] > 0):
                    jet_cmva = Jet(tree.Jet_bReg[i]*treeJet_Pt[i]/tree.Jet_pt[i], tree.Jet_eta[i], tree.Jet_hadronFlavour[i], getattr(tree,self.jetBtagBranchName)[i])
                    jets_cmva.append(jet_cmva)

            ptmin = 20.
            ptmax = 1000.
            etamin = 0.
            etamax = 2.4

            central_SF = self.get_event_SF(ptmin, ptmax, etamin, etamax, jets_cmva, "central", self.calibName)
            self.branchBuffers[self.branchBaseName][0] = central_SF

            for syst in ["JES", "LF", "HF", "LFStats1", "LFStats2", "HFStats1", "HFStats2", "cErr1", "cErr2"]:
                for sdir in ["Up", "Down"]:
                    branchName = self.branchBaseName + "_" + syst + sdir
                    self.branchBuffers[branchName][0] = self.get_event_SF( ptmin, ptmax, etamin, etamax, jets_cmva, self.sysMap[syst+sdir], self.calibName)

                    if self.includeFixPtEtaBins:
                        for ipt in range(0,5):

                            ptmin = 20.
                            ptmax = 1000.
                            etamin = 0.
                            etamax = 2.4

                            if ipt == 0:
                                ptmin = 20.
                                ptmax = 30.
                            elif ipt == 1:
                                ptmin = 30.
                                ptmax = 40.
                            elif ipt ==2:
                                ptmin = 40.
                                ptmax = 60.
                            elif ipt ==3:
                                ptmin = 60.
                                ptmax = 100.
                            elif ipt ==4:
                                ptmin = 100.
                                ptmax = 1000.

                            for ieta in range(1,4):
                                if ieta ==1:
                                    etamin = 0.
                                    etamax = 0.8
                                elif ieta ==2:
                                    etamin = 0.8
                                    etamax = 1.6
                                elif ieta ==3:
                                    etamin = 1.6
                                    etamax = 2.4

                                branchName = self.branchBaseName+"_"+syst+"_pt"+str(ipt)+"_eta"+str(ieta)+sdir
                                self.branchBuffers[branchName][0] = self.get_event_SF(ptmin, ptmax, etamin, etamax, jets_cmva, self.sysMap[syst+sdir], self.calibName)
        return isGoodEvent
    
# if btag SF per Jet has been already computed
class BTagEventWeightFromJetSF(object):
    def __init__(self, nano=True):
        self.nano = nano

    def customInit(self, initVars):
        self.sample = initVars['sample']
        self.sampleTree = initVars['sampleTree']
        if self.sample.isMC():
            self.branches = [{'name': 'bTagEventWeight', 'formula': self.getBTagEventWeight}]
        else:
            self.branches = []
    
    def getBranches(self):
        return self.branches
    
    def getBTagEventWeight(self, tree):
        weight = 1.0
        treeJet_Pt = getattr(tree, self.jetPtBranchName)
        for i in range(tree.nJet):
            if tree.Jet_bReg[i]*treeJet_Pt[i]/tree.Jet_pt[i] > 20 and abs(tree.Jet_eta[i]) < 2.4 and tree.Jet_lepFilter[i]:
                weight *= tree.Jet_btagSF[i]
        return weight

