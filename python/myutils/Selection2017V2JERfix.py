#!/usr/bin/env python
import ROOT
import array
from Jet import Jet

class Selection2017V2JERfix(object):

    def __init__(self, addSystematics=True):
        self.debug = False
        self.lastEntry = -1
        self.branches = []
        self.branchBuffers = {}
        self.nJetMax = 100
        
        for branchName in ['hJidx']:
            self.branchBuffers[branchName] = array.array('i', [0, 0])
            self.branches.append({'name': branchName, 'formula': self.getVectorBranch, 'arguments': {'branch': branchName, 'length': 2}, 'type': 'i', 'length': 2})
        
        for branchName in ['Jet_PtFixed']:
            self.branchBuffers[branchName] = array.array('f', [0.0]*self.nJetMax)
            self.branches.append({'name': branchName, 'formula': self.getVectorBranch, 'arguments': {'branch': branchName, 'size': 'nJet', 'length': self.nJetMax}, 'length': self.nJetMax, 'leaflist': branchName+'[nJet]/F'})
    
    def customInit(self, initVars):
        self.sampleTree = initVars['sampleTree']
        self.sample = initVars['sample']
        self.sampleTree.addFormula('skimming', self.sample.addtreecut)

    # return list of all branches to add
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
        if 'size' in arguments:
            length = getattr(self.sampleTree.tree, arguments['size'])
            if 'length' in arguments:
                if length > arguments['length']:
                    length = arguments['length']
        else:
            length = arguments['length']

        for i in range(length):
            destinationArray[i] = self.branchBuffers[arguments['branch']][i]

    def processEvent(self, tree):
        currentEntry = tree.GetReadEntry()
        # if current entry has not been processed yet
        if currentEntry != self.lastEntry:
            self.lastEntry = currentEntry

            # apply standard skimming cuts
            if not self.sampleTree.evaluate('skimming'):
                #print "skipped due to skimming cut:", currentEntry
                return False
            
            # select higgs candidate jets
            nJet = tree.nJet if tree.nJet < self.nJetMax else self.nJetMax
            higgsJets = []
            for i in range(nJet):
                #[x for x in jets if x.lepFilter and x.puId>0 and x.jetId>0 and x.pt>20 and abs(x.eta)<2.5]

                # calculate ~fixed smeared by unsmearing unmatched jets 
                # Alt$(Jet_genJetIdx[hJidx[1]],0) >=0 ? Jet_Pt[hJidx[1]] : Jet_pt[hJidx[1]]
                self.branchBuffers['Jet_PtFixed'][i] = tree.Jet_Pt[i] if self.sample.type == 'DATA' or tree.Jet_genJetIdx[i] >= 0 else tree.Jet_pt[i]

                if tree.Jet_lepFilter[i] and tree.Jet_puId[i]>0 and tree.Jet_jetId[i]>0 and abs(tree.Jet_eta[i])<2.5 and self.branchBuffers['Jet_PtFixed'][i] > 20:
                    higgsJets.append(Jet(self.branchBuffers['Jet_PtFixed'][i], tree.Jet_eta[i], tree.Jet_phi[i], tree.Jet_mass[i], btag=tree.Jet_btagDeepB[i], index=i))
            
            # skip event if less than 2 candidate jets found
            if len(higgsJets) < 2:
                #print "skipped, no 2 candidate jets:", currentEntry
                return False

            higgsJets = sorted(higgsJets, key=lambda x: x.btag, reverse=True)[0:2]
            self.branchBuffers['hJidx'][0] = higgsJets[0].index
            self.branchBuffers['hJidx'][1] = higgsJets[1].index
            #print "selected ", higgsJets[0].index, higgsJets[0].pt, higgsJets[0].btag, " and ", higgsJets[1].index, higgsJets[1].pt, higgsJets[1].btag


        return True


