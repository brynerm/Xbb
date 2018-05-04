import json
import os
import array
from JsonTable import JsonTable
from vLeptons import vLeptonSelector

class MuonSFfromJSON(object):

    def __init__(self, jsonFiles=None):
        self.jsonFiles = jsonFiles
        self.debug = 'XBBDEBUG' in os.environ

        # load JOSN files
        self.jsonTable = JsonTable(jsonFiles)
        self.idSf = self.jsonTable.getEtaPtTable('NUM_LooseID_DEN_genTracks', 'abseta_pt')
        self.isoSf = self.jsonTable.getEtaPtTable('NUM_LooseRelIso_DEN_LooseID', 'abseta_pt')

    def customInit(self, initVars):
        sample = initVars['sample']
        self.isData = sample.type == 'DATA'

        # prepare buffers for new branches to add
        self.branches = []
        self.branchBuffers = {}
        self.lastEntry = -1
        if not self.isData:
            for branchName in ['muonSF', 'muonSF_IdIso', 'muonSF_trigger']:
                self.branchBuffers[branchName] = array.array('f', [0])
                self.branches.append({'name': branchName, 'formula': self.getBranch, 'arguments': branchName}) 

    def getBranches(self):
        return self.branches

    # read from buffers which have been filled in processEvent()    
    def getBranch(self, event, arguments=None):
        self.processEvent(event)
        if arguments:
            return self.branchBuffers[arguments][0]

    def processEvent(self, tree):
        currentEntry = tree.GetReadEntry()
        # if current entry has not been processed yet
        if currentEntry != self.lastEntry and not self.isData:
            self.lastEntry = currentEntry

            vLeptons = vLeptonSelector(tree)
            zMuons = vLeptons.getZmuons()

            # require two muons
            if len(zMuons) == 2:
                sfIdIso = self.getIdIsoSf(eta=zMuons[0].eta, pt=zMuons[0].pt) * self.getIdIsoSf(eta=zMuons[1].eta, pt=zMuons[1].pt) 
                sfTrigger = self.getTriggerSf(eta1=zMuons[0].eta, pt1=zMuons[0].pt, eta2=zMuons[1].eta, pt2=zMuons[1].pt)
                self.branchBuffers['muonSF'][0] = sfIdIso * sfTrigger
                self.branchBuffers['muonSF_IdIso'][0] = sfIdIso
                self.branchBuffers['muonSF_trigger'][0] = sfTrigger
            else:
                self.branchBuffers['muonSF'][0] = 1.0
                self.branchBuffers['muonSF_IdIso'][0] = 1.0
                self.branchBuffers['muonSF_trigger'][0] = 1.0
        return True

    def getIdIsoSf(self, eta, pt):
        sfId = self.jsonTable.find(self.idSf, eta=eta, pt=pt)
        sfIso = self.jsonTable.find(self.isoSf, eta=eta, pt=pt)
        sf = sfId * sfIso
        if self.debug:
            print "id/iso eta:", eta, "pt:", pt, "->", sf
        return sf
    
    def getTriggerSf(self, eta1, pt1, eta2, pt2):
        leg1 = 1.0 #not implemented yet
        leg2 = 1.0 
        if self.debug:
            print "leg1: eta:", eta1, " pt:", pt1, "->", leg1
            print "leg2: eta:", eta2, " pt:", pt2, "->", leg2
            print "-->", leg1*leg2
        return leg1*leg2

if __name__ == "__main__":
    sfObject = MuonSFfromJSON([
            'weights/Zll/Muons/RunBCDEF_SF_ID.json',
            'weights/Zll/Muons/RunBCDEF_SF_ISO.json',
        ])
    print sfObject.getIdIsoSf(0.5, 42)
    print sfObject.getIdIsoSf(-0.5, 42)
    print sfObject.getIdIsoSf(1.5, 21)
    print sfObject.getIdIsoSf(-1.5, 21)
