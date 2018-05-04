#!/usr/bin/env python
import ROOT
import array
from Jet import Jet
from BranchTools import Collection
from BranchTools import AddCollectionsModule

class FSR(AddCollectionsModule):

    def __init__(self, nano=False):
        self.nano = nano
        self.debug = False
        super(FSR, self).__init__()

        # corrected Higgs properties
        self.addCollection(Collection('HCMVAV2_reg_fsrCorr', ['pt','eta','phi','mass']))
        self.addCollection(Collection('fsrJet', ['pt','eta','phi','mass','deltaR'], maxSize=4))
        self.addCollection(Collection('isrJet', ['pt','eta','phi','mass','deltaR'], maxSize=4))

        # test of new syntax
        self.addCollection(Collection('hJetFSRcorr', ['pt','eta','phi','mass'], maxSize=2))

    def processEvent(self, tree):

        if not self.hasBeenProcessed(tree):
            self.markProcessed(tree)

            higgsCandidateJets = []
            hJCMVAV2idx = [tree.hJCMVAV2idx[0], tree.hJCMVAV2idx[1]]
            nJetReg = len(tree.Jet_pt_reg)
            for i in range(2):
                if hJCMVAV2idx[i] < nJetReg:
                    higgsCandidateJets.append(Jet(pt=tree.Jet_pt_reg[hJCMVAV2idx[i]], eta=tree.Jet_eta[hJCMVAV2idx[i]], phi=tree.Jet_phi[hJCMVAV2idx[i]], mass=tree.Jet_mass[hJCMVAV2idx[i]]))

            # find FSR/ISR candidates and sort by pT
            fsrCandidateJets = []
            isrCandidateJets = []
            if len(higgsCandidateJets) >=2:
                for i in range(tree.nJet):
                    additionalJet = Jet(pt=tree.Jet_pt[i], eta=tree.Jet_eta[i], phi=tree.Jet_phi[i], mass=tree.Jet_mass[i])

                    # SELECTION for ISR/FSR jets
                    if tree.Jet_puId[i] == 7 and i not in hJCMVAV2idx and min(Jet.deltaR(additionalJet, higgsCandidateJets[0]), Jet.deltaR(additionalJet, higgsCandidateJets[1])) < 0.8:
                        fsrCandidateJets.append(additionalJet)
                    elif tree.Jet_pt[i] > 30.0 and tree.Jet_puId[i] == 7 and i not in hJCMVAV2idx and abs(tree.Jet_eta[i]) < 5.2:
                        isrCandidateJets.append(additionalJet)

                # sort descending by pT
                fsrCandidateJets.sort(key=lambda jet: jet.pt, reverse=True)
                isrCandidateJets.sort(key=lambda jet: jet.pt, reverse=True)

            # save up to 4 candidate jets
            fsrCollection = self.collections['fsrJet']
            fsrCollection.setSize(min(len(fsrCandidateJets), 3))
            for i in range(fsrCollection.getSize()):
                fsrCollection['pt'][i] = fsrCandidateJets[i].pt
                fsrCollection['eta'][i] = fsrCandidateJets[i].eta
                fsrCollection['phi'][i] = fsrCandidateJets[i].phi
                fsrCollection['mass'][i] = fsrCandidateJets[i].mass
                fsrCollection['deltaR'][i] = min(Jet.deltaR(fsrCandidateJets[i], higgsCandidateJets[0]), Jet.deltaR(fsrCandidateJets[i], higgsCandidateJets[1]))

            isrCollection = self.collections['isrJet']
            isrCollection.setSize(min(len(isrCandidateJets), 3))
            for i in range(isrCollection.getSize()):
                isrCollection['pt'][i] = isrCandidateJets[i].pt
                isrCollection['eta'][i] = isrCandidateJets[i].eta
                isrCollection['phi'][i] = isrCandidateJets[i].phi
                isrCollection['mass'][i] = isrCandidateJets[i].mass
                isrCollection['deltaR'][i] = min(Jet.deltaR(isrCandidateJets[i], higgsCandidateJets[0]), Jet.deltaR(isrCandidateJets[i], higgsCandidateJets[1]))

            # correct higgs by highest FSR jet
            if len(higgsCandidateJets) >= 2:

                # jets for dijet
                hJ = [ROOT.TLorentzVector(), ROOT.TLorentzVector()]
                for i in range(2):
                    hJ[i].SetPtEtaPhiM(higgsCandidateJets[i].pt, higgsCandidateJets[i].eta, higgsCandidateJets[i].phi, higgsCandidateJets[i].mass)

                # FSR correction
                if len(fsrCandidateJets) > 0:
                    fsr = ROOT.TLorentzVector()
                    fsr.SetPtEtaPhiM(fsrCandidateJets[0].pt, fsrCandidateJets[0].eta, fsrCandidateJets[0].phi, fsrCandidateJets[0].mass)

                    # find nearest jet
                    if Jet.deltaR(higgsCandidateJets[0], fsrCandidateJets[0]) < Jet.deltaR(higgsCandidateJets[1], fsrCandidateJets[0]):
                        hJ[0] = hJ[0] + fsr
                    else:
                        hJ[1] = hJ[1] + fsr

                # save corrected jets
                correctedJets = self.collections['hJetFSRcorr']
                correctedJets.setSize(2)
                for i in range(2):
                    correctedJets['pt'][i] = hJ[i].Pt()
                    correctedJets['eta'][i] = hJ[i].Eta()
                    correctedJets['phi'][i] = hJ[i].Phi()
                    correctedJets['mass'][i] = hJ[i].M()

                # save corrected dijet
                higgs = hJ[0] + hJ[1]
                higgsCollection = self.collections['HCMVAV2_reg_fsrCorr']
                higgsCollection['pt'][0] = higgs.Pt()
                higgsCollection['eta'][0] = higgs.Eta()
                higgsCollection['phi'][0] = higgs.Phi()
                higgsCollection['mass'][0] = higgs.M()
            else:
                self.collections['hJetFSRcorr'].setSize(0)

        return True

