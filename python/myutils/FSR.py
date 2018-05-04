#!/usr/bin/env python
import ROOT
import array

class Jet(object):
    def __init__(self, pt, eta, phi, mass):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass

class FSR(object):
    
    def deltaR(self, jet1, jet2):
        deltaPhi = ROOT.TVector2.Phi_mpi_pi(jet1.phi-jet2.phi)
        deltaEta = jet1.eta-jet2.eta
        return ROOT.TMath.Sqrt(deltaEta*deltaEta + deltaPhi*deltaPhi)

    def __init__(self, nano=False):
        self.nano = nano
        self.debug = False
        self.lastEntry = -1
        self.branches = []
        self.branchBuffers = {}

        # FSR jet candidates
        self.branchBuffers['nfsr_Jet'] = array.array('i', [0])
        self.branches.append({'name': 'nfsr_Jet', 'formula': self.getBranch, 'arguments': 'nfsr_Jet', 'type': 'i'})
        self.fsrJetProperties = ['fsrJet_pt', 'fsrJet_eta', 'fsrJet_phi', 'fsrJet_mass', 'fsrJet_deltaR']
        for fsrJetProperty in self.fsrJetProperties:
            self.branchBuffers[fsrJetProperty] = array.array('f', [0.0, 0.0, 0.0, 0.0])
            self.branches.append({'name': fsrJetProperty, 'formula': self.getVectorBranch, 'arguments': {'branch': fsrJetProperty, 'length':4}, 'length': 4, 'leaflist': fsrJetProperty + '[nfsr_Jet]/F'})

        # corrected Higgs properties
        self.higgsProperties = ['HCMVAV2_reg_fsrCorr_pt', 'HCMVAV2_reg_fsrCorr_eta', 'HCMVAV2_reg_fsrCorr_phi', 'HCMVAV2_reg_fsrCorr_mass']
        for higgsProperty in self.higgsProperties: 
            self.branchBuffers[higgsProperty] = array.array('f', [0.0])
            self.branches.append({'name': higgsProperty, 'formula': self.getBranch, 'arguments': higgsProperty})
        
        self.branchBuffers['nisr_Jet'] = array.array('i', [0])
        self.branches.append({'name': 'nisr_Jet', 'formula': self.getBranch, 'arguments': 'nisr_Jet', 'type': 'i'})
        self.isrJetProperties = ['isrJet_pt', 'isrJet_eta', 'isrJet_phi', 'isrJet_mass', 'isrJet_deltaR']
        for isrJetProperty in self.isrJetProperties:
            self.branchBuffers[isrJetProperty] = array.array('f', [0.0, 0.0, 0.0, 0.0])
            self.branches.append({'name': isrJetProperty, 'formula': self.getVectorBranch, 'arguments': {'branch': isrJetProperty, 'length':4}, 'length': 4, 'leaflist': isrJetProperty + '[nisr_Jet]/F'})

    def customInit(self, initVars):
        self.sample = initVars['sample']

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

    def processEvent(self, tree):
        currentEntry = tree.GetReadEntry()
        # if current entry has not been processed yet
        if currentEntry != self.lastEntry:
            self.lastEntry = currentEntry

            higgsCandidateJets = []
            hJCMVAV2idx = [tree.hJCMVAV2idx[0], tree.hJCMVAV2idx[1]]
            for i in range(2):
                higgsCandidateJets.append(Jet(pt=tree.Jet_pt_reg[hJCMVAV2idx[i]], eta=tree.Jet_eta[hJCMVAV2idx[i]], phi=tree.Jet_phi[hJCMVAV2idx[i]], mass=tree.Jet_mass[hJCMVAV2idx[i]]))

            # find FSR/ISR candidates and sort by pT
            fsrCandidateJets = []
            isrCandidateJets = []
            for i in range(tree.nJet):
                additionalJet = Jet(pt=tree.Jet_pt[i], eta=tree.Jet_eta[i], phi=tree.Jet_phi[i], mass=tree.Jet_mass[i])
                if tree.Jet_pt[i] > 30:
                    if tree.Jet_puId[i] == 7 and i not in hJCMVAV2idx and min(self.deltaR(additionalJet, higgsCandidateJets[0]), self.deltaR(additionalJet, higgsCandidateJets[1])) < 0.8:
                        fsrCandidateJets.append(additionalJet)
                    elif tree.Jet_puId[i] == 7 and i not in hJCMVAV2idx and abs(tree.Jet_eta[i]) < 2.4:
                        isrCandidateJets.append(additionalJet)
            fsrCandidateJets.sort(key=lambda jet: jet.pt, reverse=True)
            isrCandidateJets.sort(key=lambda jet: jet.pt, reverse=True)

            # save up to 4 candidate jets
            self.branchBuffers['nfsr_Jet'][0] = min(len(fsrCandidateJets), 3)
            for i in range(self.branchBuffers['nfsr_Jet'][0]):
                self.branchBuffers['fsrJet_pt'][i] = fsrCandidateJets[i].pt
                self.branchBuffers['fsrJet_eta'][i] = fsrCandidateJets[i].eta
                self.branchBuffers['fsrJet_phi'][i] = fsrCandidateJets[i].phi
                self.branchBuffers['fsrJet_mass'][i] = fsrCandidateJets[i].mass
                self.branchBuffers['fsrJet_deltaR'][i] = min(self.deltaR(fsrCandidateJets[i], higgsCandidateJets[0]), self.deltaR(fsrCandidateJets[i], higgsCandidateJets[1]))

            self.branchBuffers['nisr_Jet'][0] = min(len(isrCandidateJets), 3)
            for i in range(self.branchBuffers['nisr_Jet'][0]):
                self.branchBuffers['isrJet_pt'][i] = isrCandidateJets[i].pt
                self.branchBuffers['isrJet_eta'][i] = isrCandidateJets[i].eta
                self.branchBuffers['isrJet_phi'][i] = isrCandidateJets[i].phi
                self.branchBuffers['isrJet_mass'][i] = isrCandidateJets[i].mass
                self.branchBuffers['isrJet_deltaR'][i] = min(self.deltaR(isrCandidateJets[i], higgsCandidateJets[0]), self.deltaR(isrCandidateJets[i], higgsCandidateJets[1]))

            # correct higgs by highest FSR jet
            higgs = ROOT.TLorentzVector()
            higgs.SetPtEtaPhiM(tree.HCMVAV2_reg_pt, tree.HCMVAV2_reg_eta, tree.HCMVAV2_reg_phi, tree.HCMVAV2_reg_mass)

            if len(fsrCandidateJets) > 0:
                fsr = ROOT.TLorentzVector()
                fsr.SetPtEtaPhiM(fsrCandidateJets[0].pt, fsrCandidateJets[0].eta, fsrCandidateJets[0].phi, fsrCandidateJets[0].mass)
                if self.debug:
                    print "Higgs: :", tree.HCMVAV2_reg_pt, tree.HCMVAV2_reg_eta, tree.HCMVAV2_reg_phi, tree.HCMVAV2_reg_mass
                    print " +FSR:", fsrCandidateJets[0].pt, fsrCandidateJets[0].eta, fsrCandidateJets[0].phi, fsrCandidateJets[0].mass
                    print " deltaR:", self.deltaR(fsrCandidateJets[0], higgsCandidateJets[0]), " / ", self.deltaR(fsrCandidateJets[0], higgsCandidateJets[1])
                    print " nFSR:", len(fsrCandidateJets)
                    
                oldMass = higgs.M()
                higgs = higgs + fsr
                if self.debug:
                    if abs(125-higgs.M()) < abs(125-oldMass):
                        print "\x1b[32m",
                    else:
                        print "\x1b[31m", 
                    print " -> ", higgs.Pt(), higgs.Eta(), higgs.Phi(), higgs.M(), "\x1b[0m"

            self.branchBuffers['HCMVAV2_reg_fsrCorr_pt'][0]   = higgs.Pt()
            self.branchBuffers['HCMVAV2_reg_fsrCorr_eta'][0]  = higgs.Eta()
            self.branchBuffers['HCMVAV2_reg_fsrCorr_phi'][0]  = higgs.Phi()
            self.branchBuffers['HCMVAV2_reg_fsrCorr_mass'][0] = higgs.M()

        return True

