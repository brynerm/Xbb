from __future__ import print_function
import os
import array
import ROOT
import random

# Rochester Muon momentum SF
# see https://twiki.cern.ch/twiki/bin/viewauth/CMS/RochcorMuon
class Rochester2016(object):

    def __init__(self):
        self.config = None
        self.branches = []
        self.rc = None
        self.isData = False
        self.debug = 'XBBDEBUG' in os.environ
        self.lastEntry = -1
        self.nMuons = 0
    
    def customInit(self, initVars):
        self.config = initVars['config']
        self.isData = initVars['sample'].type == 'DATA'
        status = ROOT.gSystem.Load(self.config.get('Directories','vhbbpath') + "/interface/Rochester2016_h.so")
        rcPath = self.config.get('Directories','vhbbpath') + "/interface/Rochester2016/rcdata.2016.v3"
        self.rc = ROOT.RoccoR(rcPath)
        self.maxNmuons = 16
        self.branchBuffers = {'Muon_rochesterSF': array.array('f', [1.0] * self.maxNmuons)}
        self.branches.append({'name': 'Muon_rochesterSF', 'formula': self.getVectorBranch, 'arguments': {'branch': 'Muon_rochesterSF'}, 'length': self.maxNmuons, 'leaflist': 'Muon_rochesterSF[nMuon]/F'})

    # return list of branches to be filled 
    def getBranches(self):
        return self.branches

    # read from buffers which have been filled in processEvent()    
    def getVectorBranch(self, event, arguments=None, destinationArray=None):
        self.processEvent(event)
        for i in range(self.nMuons):
            destinationArray[i] =  self.branchBuffers[arguments['branch']][i]

    # compute SF for all muons in the event
    def processEvent(self, tree):
        currentEntry = tree.GetReadEntry()
        # if current entry has not been processed yet
        if currentEntry != self.lastEntry:
            self.lastEntry = currentEntry
            self.nMuons = tree.nMuon
            for i in range(self.nMuons):    
                sf = 1.0
                if self.isData:
                    sf = self.rc.kScaleDT(tree.Muon_charge[i], tree.Muon_pt[i], tree.Muon_eta[i], tree.Muon_phi[i], 0, 0)
                else:
                    u1 = random.uniform(0.0, 1.0) 
                    u2 = random.uniform(0.0, 1.0) 
                    sf = self.rc.kScaleAndSmearMC(int(tree.Muon_charge[i]), tree.Muon_pt[i], tree.Muon_eta[i], tree.Muon_phi[i], int(tree.Muon_nTrackerLayers[i]), u1, u2, 0, 0)
                self.branchBuffers['Muon_rochesterSF'][i] = sf
                if self.debug:
                    print('DEBUG: muon', i, tree.Muon_charge[i], tree.Muon_pt[i], tree.Muon_eta[i], tree.Muon_phi[i], tree.Muon_nTrackerLayers[i], ' --> ', sf) 
        return True
