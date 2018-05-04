#!/usr/bin/env python
import array

class lepton(object):
    def __init__(self, pt, eta, phi, mass, charge):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass
        self.charge = charge

class vLeptonSelector(object):

    def __init__(self, tree):
        zMuons = []
        zElectrons = []
        wElectrons = []
        wMuons = []

        self.vMuonIdx = []
        self.vElectronIdx = []
        for i in range(tree.nMuon):
            if tree.Muon_pt[i]>20 and tree.Muon_pfRelIso04_all[i]<0.25 and tree.Muon_dxy[i]<0.05 and tree.Muon_dz[i]<0.2:
                zMuons.append(lepton(pt=tree.Muon_pt[i], eta=tree.Muon_eta[i], phi=tree.Muon_phi[i], mass=tree.Muon_mass[i], charge=tree.Muon_charge[i]))
                self.vMuonIdx.append(i)
        for i in range(tree.nElectron):
            if tree.Electron_pt[i]>20 and tree.Electron_mvaSpring16GP_WP90[i] and tree.Electron_pfRelIso03_all[i]<0.15:
                zElectrons.append(lepton(pt=tree.Electron_pt[i], eta=tree.Electron_eta[i], phi=tree.Electron_phi[i], mass=tree.Electron_mass[i], charge=tree.Electron_charge[i]))
                self.vElectronIdx.append(i)
        for i in range(tree.nMuon):
            if tree.Muon_pt[i]>25 and tree.Muon_tightId[i] >= 1 and tree.Muon_pfRelIso04_all[i]<0.15 and tree.Muon_dxy[i]<0.05 and tree.Muon_dz[i]<0.2:
                wMuons.append(lepton(pt=tree.Muon_pt[i], eta=tree.Muon_eta[i], phi=tree.Muon_phi[i], mass=tree.Muon_mass[i], charge=tree.Muon_charge[i]))
                self.vMuonIdx.append(i)
        for i in range(tree.nElectron):
            if tree.Electron_pt[i]>25 and tree.Electron_mvaSpring16GP_WP90[i] and tree.Electron_pfRelIso03_all[i]<0.12:
                wElectrons.append(lepton(pt=tree.Electron_pt[i], eta=tree.Electron_eta[i], phi=tree.Electron_phi[i], mass=tree.Electron_mass[i], charge=tree.Electron_charge[i]))
                self.vElectronIdx.append(i)

        vLeptons = [] 
        Vtype = 0 
        if len(zMuons) >= 2:
            if zMuons[0].pt > 20:
                for i in xrange(1,len(zMuons)):
                    if zMuons[0].charge * zMuons[i].charge < 0:
                        Vtype = 0
                        vLeptons = [zMuons[0],zMuons[i]]
                        self.vMuonIdx = [self.vMuonIdx[0], self.vMuonIdx[i]]
                        break
        elif len(zElectrons) >= 2:
            if zElectrons[0].pt > 20:
                for i in xrange(1,len(zElectrons)):
                    if zElectrons[0].charge * zElectrons[i].charge < 0:
                        Vtype = 1
                        vLeptons = [zElectrons[0],zElectrons[i]]
                        self.vElectronIdx = [self.vElectronIdx[0], self.vElectronIdx[i]]
                        break
        elif len(wElectrons) + len(wMuons) == 1:
            if len(wMuons) == 1:
                Vtype = 2
                vLeptons = [wMuons[0]]
            if len(wElectrons) == 1:
                Vtype = 3
                vLeptons = [wElectrons[0]]
        elif len(zElectrons) + len(zMuons) > 0:
            Vtype = 5
        else:
            Vtype = 4
            if tree.MET_pt < 150:
                Vtype = -1
        self.Vtype = Vtype
        self.vLeptons = vLeptons
        self.zMuons = zMuons
        self.zElectrons = zElectrons
        self.vMuonIdx = sorted(list(set(self.vMuonIdx)))
        self.vElectronIdx = sorted(list(set(self.vElectronIdx)))

    def getVleptons(self):
        return self.vLeptons
    
    def getVtype(self):
        return self.Vtype
    
    def getZelectrons(self):
        return self.zElectrons

    def getZmuons(self):
        return self.zMuons

class vLeptons(object):
    def __init__(self):
        self.lastEntry = -1
        self.branches = []
        self.branchBuffers = {}

        self.branchBuffers['nVMuonIdx'] = array.array('i', [0])
        self.branchBuffers['nVElectronIdx'] = array.array('i', [0])
        self.branches.append({'name': 'nVMuonIdx', 'formula': self.getBranch, 'arguments': 'nVMuonIdx', 'type': 'i'})
        self.branches.append({'name': 'nVElectronIdx', 'formula': self.getBranch, 'arguments': 'nVElectronIdx', 'type': 'i'})

        self.branchBuffers['VMuonIdx'] = array.array('i', [-1, -1])
        self.branchBuffers['VElectronIdx'] = array.array('i', [-1, -1])
        self.branches.append({'name': 'VMuonIdx', 'formula': self.getVMuonIdx, 'length':2, 'type': 'i', 'leaflist': 'VMuonIdx[nVMuonIdx]/I'})
        self.branches.append({'name': 'VElectronIdx', 'formula': self.getVElectronIdx, 'length':2, 'type': 'i', 'leaflist': 'VelectronIdx[nVElectronIdx]/I'})
        self.selector = None

    def customInit(self, initVars):
        pass

    def getBranches(self):
        return self.branches

    def processEvent(self, tree):
        currentEntry = tree.GetReadEntry()
        # if current entry has not been processed yet
        if currentEntry != self.lastEntry:
            self.lastEntry = currentEntry
            self.selector = vLeptonSelector(tree)

            self.branchBuffers['VMuonIdx'][0] = self.selector.vMuonIdx[0] if len(self.selector.vMuonIdx) > 0 else -2 
            self.branchBuffers['VMuonIdx'][1] = self.selector.vMuonIdx[1] if len(self.selector.vMuonIdx) > 1 else -2 
            self.branchBuffers['VElectronIdx'][0] = self.selector.vElectronIdx[0] if len(self.selector.vElectronIdx) > 0 else -2 
            self.branchBuffers['VElectronIdx'][1] = self.selector.vElectronIdx[1] if len(self.selector.vElectronIdx) > 1 else -2 
            self.branchBuffers['nVMuonIdx'][0] = len(self.selector.vMuonIdx)
            self.branchBuffers['nVElectronIdx'][0] = len(self.selector.vElectronIdx)
            #print self.branchBuffers['VMuonIdx'][0],self.branchBuffers['VMuonIdx'][1],self.branchBuffers['VElectronIdx'][0],self.branchBuffers['VElectronIdx'][1],self.branchBuffers['nVMuonIdx'][0],self.branchBuffers['nVElectronIdx'][0]

        return True


    # read from buffers which have been filled in processEvent()    
    def getBranch(self, event, arguments=None):
        self.processEvent(event)
        if arguments:
            return self.branchBuffers[arguments][0]

    def getVMuonIdx(self, event, arguments=None, destinationArray=None):
        self.processEvent(event)
        for i in range(len(self.selector.vMuonIdx)):
            destinationArray[i] = self.branchBuffers['VMuonIdx'][i]
    
    def getVElectronIdx(self, event, arguments=None, destinationArray=None):
        self.processEvent(event)
        for i in range(len(self.selector.vElectronIdx)):
            destinationArray[i] = self.branchBuffers['VElectronIdx'][i]
