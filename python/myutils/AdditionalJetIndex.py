#!/usr/bin/env python
import ROOT
import numpy as np
import array

class AdditionalJetIndex(object):

    def __init__(self):
        self.HV_treeIdx = None
        self.AJ_treeIdx = None
        self.HF_treeIdx = None
        self.PS_treeIdx = None
        self.HjF_treeIdx = None
        self.nhj_treeIdx = None
        self.perpV_treeIdx = None
        self.process_treeIdx = None
        self.Saj_treeIdx = None
        self.Maj_treeIdx = None
        self.HVpp_treeIdx = None
        self.ISR_treeIdx = {}
        self.Sisr_treeIdx = {}
        self.Sfsr_treeIdx = {}
        self.Misr_treeIdx = {}
        self.Mfsr_treeIdx = {}

        #Tree Buffers
        self.AJidx = None
        self.PtSL = None
        self.HV = None
        self.HF = None
        self.HjF = None
        self.nhj_idx = None
        self.nhj_dR = None
        self.HVpp = None
        self.perpV = None
        self.Saj = None
        self.Maj = None
        self.StdAj = None
        self.nAJ = None
        self.ISRidx = {}
        self.FSRidx = {}
        self.Sisr = {}
        self.Sfsr = {}
        self.Misr = {}
        self.Mfsr = {}
        self.StdIsr = {}
        self.StdFsr = {}
        self.nISR = {}
        self.nFSR = {}
        self.cuts = ['30','excl','08']
        for cut in self.cuts:
            self.ISR_treeIdx[cut] = None
            self.Sisr_treeIdx[cut] = None
            self.Sfsr_treeIdx[cut] = None
            self.Misr_treeIdx[cut] = None
            self.Mfsr_treeIdx[cut] = None
            self.ISRidx[cut] = None
            self.FSRidx[cut] = None
            self.Sisr[cut] = None
            self.Sfsr[cut] = None
            self.Misr[cut] = None
            self.Mfsr[cut] = None
            self.StdIsr[cut] = None
            self.StdFsr[cut] = None
            self.nISR[cut] = None
            self.nFSR[cut] = None
        self.maxnAJ = 8
        self.nJet = 12

        #Branches
        self.branches = []
        self.branchBuffers = {}
        #self.branches.append({'name': 'HVcombined_pt', 'formula': lambda tree: self.getHV(tree).Pt(), 'type': 'f'})
        #self.branches.append({'name': 'HVcombined_eta', 'formula': lambda tree: self.getHV(tree).Eta(), 'type': 'f'})
        #self.branches.append({'name': 'HVcombined_phi', 'formula': lambda tree: self.getHV(tree).Phi(), 'type': 'f'})
        #self.branches.append({'name': 'HVcombined_mass', 'formula': lambda tree: self.getHV(tree).M(), 'type': 'f'})
        self.branches.append({'name': 'HCMVAV2_reg_wFSR_pt', 'formula': lambda tree: self.getHF(tree).Pt(), 'type': 'f'})
        self.branches.append({'name': 'HCMVAV2_reg_wFSR_eta', 'formula': lambda tree: self.getHF(tree).Eta(), 'type': 'f'})
        self.branches.append({'name': 'HCMVAV2_reg_wFSR_phi', 'formula': lambda tree: self.getHF(tree).Phi(), 'type': 'f'})
        self.branches.append({'name': 'HCMVAV2_reg_wFSR_mass', 'formula': lambda tree: self.getHF(tree).M(), 'type': 'f'})
        self.branches.append({'name': 'hJetCMVAV2_pt_reg_wFSR_0', 'formula': lambda tree: self.getHjF(tree)[0], 'type': 'f'})
        self.branches.append({'name': 'hJetCMVAV2_pt_reg_wFSR_1', 'formula': lambda tree: self.getHjF(tree)[2], 'type': 'f'})
        self.branches.append({'name': 'hJetCMVAV2_eta_reg_wFSR_0', 'formula': lambda tree: self.getHjF(tree)[1], 'type': 'f'})
        self.branches.append({'name': 'hJetCMVAV2_eta_reg_wFSR_1', 'formula': lambda tree: self.getHjF(tree)[3], 'type': 'f'})
        self.branches.append({'name': 'SumAddJet_pt', 'formula': lambda tree: self.getSumAJ(tree).Pt(), 'type': 'f'})
        self.branches.append({'name': 'SumAddJet_eta', 'formula': lambda tree: self.getSumAJ(tree).Eta(), 'type': 'f'})
        self.branches.append({'name': 'SumAddJet_phi', 'formula': lambda tree: self.getSumAJ(tree).Phi(), 'type': 'f'})
        self.branches.append({'name': 'SumAddJet_mass', 'formula': lambda tree: self.getSumAJ(tree).M(), 'type': 'f'})
        #self.branches.append({'name': 'AddJet_meanPt', 'formula': lambda tree: self.getMeanAJ(tree), 'type': 'f'})
        #self.branches.append({'name': 'AddJet_stdPt', 'formula': lambda tree: self.getStdAJ(tree), 'type': 'f'})
        self.branches.append({'name': 'nAddJet', 'formula': self.getNAJ, 'type': 'i'})
        self.branches.append({'name': 'rAddJet_idx', 'formula': self.getrAJ, 'type': 'i'})
        wName = 'AddJet_idx'
        self.branchBuffers[wName] = np.zeros(self.maxnAJ, dtype=np.int16)
        self.branches.append({'name': wName, 'formula': self.getVectorBranch, 'arguments': {'branch': wName, 'length': self.maxnAJ}, 'length': self.maxnAJ, 'type': 'i'})
        wName = 'Jet_nhJidx'
        self.branchBuffers[wName] = np.zeros(self.maxnAJ, dtype=np.int16)
        self.branches.append({'name': wName, 'formula': self.getVectorBranch, 'arguments': {'branch': wName, 'length': self.nJet}, 'length': self.nJet , 'type': 'i'})
        #for wName in ['Jet_nhJdR','Jet_perpV']:
        for wName in ['Jet_nhJdR']:
            self.branchBuffers[wName] = np.zeros(self.maxnAJ, dtype=np.int16)
            self.branches.append({'name': wName, 'formula': self.getVectorBranch, 'arguments': {'branch': wName, 'length': self.nJet}, 'length': self.nJet , 'type': 'f'})
        
        for definition in self.cuts:
            for wName in ['SumISR'+definition+'_pt','SumISR'+definition+'_eta','SumISR'+definition+'_phi','SumISR'+definition+'_mass','SumFSR'+definition+'_pt','SumFSR'+definition+'_eta','SumFSR'+definition+'_phi','SumFSR'+definition+'_mass']:
                self.branchBuffers[wName] = 0.0
                self.branches.append({'name': wName, 'formula': self.getCutBranch, 'arguments': {'branch': wName}, 'type': 'f'})

            for wName in ['nISR'+definition,'nFSR'+definition,'rISR'+definition+'_idx','rFSR'+definition+'_idx']:
                self.branchBuffers[wName] = 0
                self.branches.append({'name': wName, 'formula': self.getCutBranch, 'arguments': {'branch': wName}, 'type': 'i'})

            for wName in ['ISR'+definition+'_idx', 'FSR'+definition+'_idx']:
                self.branchBuffers[wName] = np.zeros(self.maxnAJ, dtype=np.int16)
                self.branches.append({'name': wName, 'formula': self.getVectorBranch, 'arguments': {'branch': wName, 'length': self.maxnAJ}, 'length': self.maxnAJ, 'type': 'i'})

    # read from buffers which have been filled in processEvent()    
    def getVectorBranch(self, event, arguments=None, destinationArray=None):
        self.processEvent(event)
        for i in range(arguments['length']):
            destinationArray[i] =  self.branchBuffers[arguments['branch']][i]

    def getCutBranch(self, event, arguments=None, destination=None):
        self.processEvent(event)
        return self.branchBuffers[arguments['branch']]

    def processEvent(self, tree):
        if self.process_treeIdx != tree.GetReadEntry():
            self.process_treeIdx = tree.GetReadEntry()

            self.branchBuffers['AddJet_idx'] = self.getAJidx(tree)
            self.branchBuffers['Jet_nhJidx'] = self.getJet_nhJ_idx(tree)
            self.branchBuffers['Jet_nhJdR'] = self.getJet_nhJ_dR(tree)
            #self.branchBuffers['Jet_perpV'] = self.getJet_perpV(tree)

            for definition in self.cuts:
                self.branchBuffers['ISR'+definition+'_idx'] = self.getISRidx(tree,definition)
                self.branchBuffers['FSR'+definition+'_idx'] = self.getFSRidx(tree,definition)
                #self.branchBuffers['ISR'+definition+'_meanPt'] = self.getMeanISR(tree,definition)
                #self.branchBuffers['FSR'+definition+'_meanPt'] = self.getMeanFSR(tree,definition)
                #self.branchBuffers['ISR'+definition+'_stdPt'] = self.getStdISR(tree,definition)
                #self.branchBuffers['FSR'+definition+'_stdPt'] = self.getStdFSR(tree,definition)
                self.branchBuffers['SumISR'+definition+'_pt'] = self.getSumISR(tree,definition).Pt()
                self.branchBuffers['SumISR'+definition+'_eta'] = self.getSumISR(tree,definition).Eta()
                self.branchBuffers['SumISR'+definition+'_phi'] = self.getSumISR(tree,definition).Phi()
                self.branchBuffers['SumISR'+definition+'_mass'] = self.getSumISR(tree,definition).M()
                self.branchBuffers['SumFSR'+definition+'_pt'] = self.getSumFSR(tree,definition).Pt()
                self.branchBuffers['SumFSR'+definition+'_eta'] = self.getSumFSR(tree,definition).Eta()
                self.branchBuffers['SumFSR'+definition+'_phi'] = self.getSumFSR(tree,definition).Phi()
                self.branchBuffers['SumFSR'+definition+'_mass'] = self.getSumFSR(tree,definition).M()
                self.branchBuffers['nISR'+definition] = self.getNISR(tree,definition)
                self.branchBuffers['nFSR'+definition] = self.getNFSR(tree,definition)
                self.branchBuffers['rISR'+definition+'_idx'] = self.getrISR(tree,definition)
                self.branchBuffers['rFSR'+definition+'_idx'] = self.getrFSR(tree,definition)
            
    def getBranches(self):
        return self.branches

    def getPtSortedList(self,tree):
        '''returns a list of jet indices orderd by pt'''
        if tree.GetReadEntry() == self.PS_treeIdx:
            return self.PtSL
        pt = np.array(tree.Jet_pt)
        self.mean_pt = np.mean(pt)
        self.std_pt = np.std(pt, ddof = 0)
        self.PtSL = np.argsort(-pt)
        self.PS_treeIdx = tree.GetReadEntry()
        return self.PtSL

    def getAJidx(self, tree):
        '''returns a list of additional jet indices'''
        if tree.GetReadEntry() == self.AJ_treeIdx:
            if self.AJidx is not None:
                return self.AJidx
        self.AJidx = []
        self.AJ_treeIdx = tree.GetReadEntry()
        j = 0
        for jet_idx in self.getPtSortedList(tree):
            if jet_idx < tree.nJet and abs(tree.Jet_eta[jet_idx]) < 5.2 and tree.Jet_puId[jet_idx] == 7 and jet_idx != tree.hJCMVAV2idx[0] and jet_idx != tree.hJCMVAV2idx[1]:
                self.AJidx.append(jet_idx)
                j += 1
        self.nAJ = j
        for i in range(j,self.maxnAJ):
            self.AJidx.append(-1)
        return self.AJidx
        
    def getHV(self,tree):
        '''not used'''
        if tree.GetReadEntry() == self.HV_treeIdx:
            if self.HV is not None:
                return self.HV
        self.HV_treeIdx = tree.GetReadEntry()
        h = ROOT.TLorentzVector()
        h.SetPtEtaPhiM(tree.HCMVAV2_pt,tree.HCMVAV2_eta,tree.HCMVAV2_phi,tree.HCMVAV2_mass)
        v = ROOT.TLorentzVector()
        v.SetPtEtaPhiM(tree.V_new_pt,tree.V_new_eta,tree.V_new_phi,tree.V_new_mass)
        self.HV = h+v
        return self.HV
    
    def getHF(self,tree):
        '''returns the four-vector of the FSR corrected Higgs'''
        if tree.GetReadEntry() == self.HF_treeIdx:
            if self.HF is not None:
                return self.HF
        self.HF_treeIdx = tree.GetReadEntry()
        h = ROOT.TLorentzVector()
        h.SetPtEtaPhiM(tree.HCMVAV2_reg_pt,tree.HCMVAV2_reg_eta,tree.HCMVAV2_reg_phi,tree.HCMVAV2_reg_mass)
        f = ROOT.TLorentzVector()
        fsridx = self.getFSRidx(tree,'08')[0]
        if fsridx < 0:
            self.HF = h
            return h
        f.SetPtEtaPhiM(tree.Jet_pt[fsridx],tree.Jet_eta[fsridx],tree.Jet_phi[fsridx],tree.Jet_mass[fsridx])
        self.HF = h+f
        return self.HF

    def getHjF(self,tree):
        '''returns the pt and eta of the FSR corrected b candidates'''
        if tree.GetReadEntry() == self.HjF_treeIdx:
            if self.HjF is not None:
                return self.HjF
        self.HjF_treeIdx = tree.GetReadEntry()
        fsridx = self.getFSRidx(tree,'08')[0]
        if fsridx < 0:
            self.HjF = (tree.hJetCMVAV2_pt_reg_0, tree.Jet_eta[tree.hJCMVAV2idx[0]], tree.hJetCMVAV2_pt_reg_1, tree.Jet_eta[tree.hJCMVAV2idx[1]])
            return self.HjF
        f = ROOT.TLorentzVector()
        f.SetPtEtaPhiM(tree.Jet_pt[fsridx],tree.Jet_eta[fsridx],tree.Jet_phi[fsridx],tree.Jet_mass[fsridx])
        nhJ_idx = self.getJet_nhJ_idx(tree)[fsridx]
        if nhJ_idx == tree.hJCMVAV2idx[0]:
            h0 = ROOT.TLorentzVector()
            h0.SetPtEtaPhiM(tree.hJetCMVAV2_pt_reg_0,tree.Jet_eta[tree.hJCMVAV2idx[0]],tree.Jet_phi[tree.hJCMVAV2idx[0]],tree.Jet_mass[tree.hJCMVAV2idx[0]]*tree.hJetCMVAV2_pt_reg_0/tree.Jet_pt[tree.hJCMVAV2idx[0]])
            h0pt = (h0+f).Pt()
            h0eta = (h0+f).Eta()
            h1pt = tree.hJetCMVAV2_pt_reg_1
            h1eta = tree.Jet_eta[tree.hJCMVAV2idx[1]]

        if nhJ_idx == tree.hJCMVAV2idx[1]:
            h1 = ROOT.TLorentzVector()
            h1.SetPtEtaPhiM(tree.hJetCMVAV2_pt_reg_1,tree.Jet_eta[tree.hJCMVAV2idx[1]],tree.Jet_phi[tree.hJCMVAV2idx[1]],tree.Jet_mass[tree.hJCMVAV2idx[1]]*tree.hJetCMVAV2_pt_reg_1/tree.Jet_pt[tree.hJCMVAV2idx[1]])
            h1pt = (h1+f).Pt()
            h1eta = (h1+f).Eta()
            h0pt = tree.hJetCMVAV2_pt_reg_0
            h0eta = tree.Jet_eta[tree.hJCMVAV2idx[0]]

        self.HjF = (h0pt,h0eta,h1pt,h1eta)

        return self.HjF

    def getNISR(self,tree,cut):
        '''returns the number of ISR jets'''
        self.getISRidx(tree,cut)
        return self.nISR[cut]

    def getNFSR(self,tree,cut):
        '''returns the number of FSR jets'''
        self.getISRidx(tree,cut)
        return self.nFSR[cut]

    def getNAJ(self,tree):
        '''returns the number of additional jets'''
        return self.nAJ

    def getrISR(self,tree,cut):
        ''' returns the jet index of a random ISR jet'''
        n = self.getNISR(tree,cut)
        if not n:
            return -1
        return self.getISRidx(tree,cut)[np.random.randint(n)]

    def getrFSR(self,tree,cut):
        ''' returns the jet index of a random FSR jet'''
        n = self.getNFSR(tree,cut)
        if not n:
            return -1
        return self.getFSRidx(tree,cut)[np.random.randint(n)]

    def getrAJ(self,tree):
        ''' returns the jet index of a random additional jet'''
        n = self.getNAJ(tree)
        if not n:
            return -1
        return self.getAJidx(tree)[np.random.randint(n)]

    def getSumISR(self,tree,cut):
        '''returns the four-vector sum of all ISR jets'''
        if tree.GetReadEntry() == self.Sisr_treeIdx[cut]:
             return self.Sisr[cut]
        v = ROOT.TLorentzVector()
        for idx in self.getISRidx(tree,cut):
            if idx == -1:
                break
            j = ROOT.TLorentzVector()
            j.SetPtEtaPhiM(tree.Jet_pt[idx],tree.Jet_eta[idx],tree.Jet_phi[idx],tree.Jet_mass[idx])
            v += j
        self.Sisr_treeIdx[cut]=tree.GetReadEntry() 
        self.Sisr[cut] = v
        return v

    def getSumAJ(self,tree):
        '''returns the four-vector sum of all additional jets'''
        if tree.GetReadEntry() == self.Saj_treeIdx:
             return self.Saj
        v = ROOT.TLorentzVector()
        for idx in self.getAJidx(tree):
            if idx == -1:
                break
            j = ROOT.TLorentzVector()
            j.SetPtEtaPhiM(tree.Jet_pt[idx],tree.Jet_eta[idx],tree.Jet_phi[idx],tree.Jet_mass[idx])
            v += j
        self.Saj_treeIdx = tree.GetReadEntry() 
        self.Saj = v
        return v

    def getSumFSR(self,tree,cut):
        '''returns the four-vector sum of all FSR'''
        if tree.GetReadEntry() == self.Sfsr_treeIdx[cut]:
             return self.Sfsr[cut]
        v = ROOT.TLorentzVector()
        for idx in self.getFSRidx(tree,cut):
            if idx == -1:
                break
            j = ROOT.TLorentzVector()
            j.SetPtEtaPhiM(tree.Jet_pt[idx],tree.Jet_eta[idx],tree.Jet_phi[idx],tree.Jet_mass[idx])
            v += j
        self.Sfsr_treeIdx[cut] = tree.GetReadEntry() 
        self.Sfsr[cut] = v
        return v

    def getMeanAJ(self,tree):
        '''not used'''
        if tree.GetReadEntry() == self.Maj_treeIdx:
             return self.Maj
        aj = np.array(self.getAJidx(tree))
        nans = aj < 0
        pt = np.ones_like(aj,dtype=np.float)
        for key,val in enumerate(aj):
            if val < 0:
                pt[key] = np.nan
            else:
                pt[key] = tree.Jet_pt[val]
        if np.all(nans):
            self.Maj = 0
            self.StdAj = 0
        else:
            self.StdAj = np.nanstd(pt)
            self.Maj = np.nanmean(pt)
        return self.Maj

    def getMeanISR(self,tree,cut):
        '''not used'''
        if tree.GetReadEntry() == self.Misr_treeIdx[cut]:
             return self.Misr[cut]
        jets = np.array(self.getISRidx(tree,cut))
        nans = jets < 0
        pt = np.ones_like(jets,dtype=np.float)
        for key,val in enumerate(jets):
            if val < 0:
                pt[key] = np.nan
            else:
                pt[key] = tree.Jet_pt[val]
        if np.all(nans):
            self.Misr[cut] = 0
            self.StdIsr[cut] = 0
        else:
            self.StdIsr[cut] = np.nanstd(pt)
            self.Misr[cut] = np.nanmean(pt)
        return self.Misr[cut]

    def getMeanFSR(self,tree,cut):
        '''not used'''
        if tree.GetReadEntry() == self.Mfsr_treeIdx[cut]:
             return self.Mfsr[cut]
        jets = np.array(self.getFSRidx(tree,cut))
        nans = jets < 0
        pt = np.ones_like(jets,dtype=np.float)
        for key,val in enumerate(jets):
            if val < 0:
                pt[key] = np.nan
            else:
                pt[key] = tree.Jet_pt[val]
        if np.all(nans):
            self.Mfsr[cut] = 0
            self.StdFsr[cut] = 0
        else:
            self.StdFsr[cut] = np.nanstd(pt)
            self.Mfsr[cut] = np.nanmean(pt)
        return self.Mfsr[cut]

    def getStdAJ(self,tree):
        '''not used'''
        self.getMeanAJ(tree)
        return self.StdAj

    def getStdISR(self,tree,cut):
        '''not used'''
        self.getMeanISR(tree,cut)
        return self.StdIsr[cut]

    def getStdFSR(self,tree,cut):
        '''not used'''
        self.getMeanFSR(tree,cut)
        return self.StdFsr[cut]

    def dR(self,eta1,eta2,phi1,phi2):
        '''delta R'''
        return np.sqrt((eta1 - eta2)**2 + self.dPhi(phi1, phi2)**2)

    def dPhi(self,phi1,phi2):
        '''delta phi'''
        dp = phi1 - phi2
        if dp > np.pi:
            dp -= 2*np.pi
        if dp < -np.pi:
            dp += 2*np.pi
        return dp

    def dR_(self,eta1,eta2,phi1,phi2):
        '''not used'''
        return self.dR(eta1,-eta2,phi1,phi2+np.pi)
    
    def getAJ0_nhJ_dPhi2(self,tree):
        '''not used'''
        idx = self.getAJidx(tree)[0]
        nhJ_idx = self.getJet_nhJ_idx(tree)[idx]
        if idx>0 and nhJ_idx > 0:
            return self.dPhi(tree.Jet_phi[idx],tree.Jet_phi[nhJ_idx])**2
        return -1

    def getAJ0_nhJ_dEta2(self,tree):
        '''not used'''
        idx = self.getAJidx(tree)[0]
        nhJ_idx = self.getJet_nhJ_idx(tree)[idx]
        if idx>0 and nhJ_idx > 0:
            return (tree.Jet_eta[idx]-tree.Jet_eta[nhJ_idx])**2
        return -1

    def getJet_nhJ_idx(self,tree):
        '''returns the jet index of the nearest b candidate of the jet'''
        if tree.GetReadEntry() == self.nhj_treeIdx:
            return self.nhj_idx
        self.nhj_treeIdx = tree.GetReadEntry()
        self.nhj_idx = []
        self.nhj_dR = []
        h0 = tree.hJCMVAV2idx[0]
        h1 = tree.hJCMVAV2idx[1]
        if 0 <= h0 < tree.nJet and 0 <= h1 < tree.nJet:
            for idx in range(self.nJet):
                if idx < tree.nJet:
                    dRh1 = self.dR(tree.Jet_eta[idx],tree.Jet_eta[h1],tree.Jet_phi[idx],tree.Jet_phi[h1])
                    dRh0 = self.dR(tree.Jet_eta[idx],tree.Jet_eta[h0],tree.Jet_phi[idx],tree.Jet_phi[h0])
                    if dRh0 < dRh1:
                        self.nhj_idx.append(h0)
                        self.nhj_dR.append(dRh0)
                    else:
                        self.nhj_idx.append(h1)
                        self.nhj_dR.append(dRh1)
                else:
                    self.nhj_idx.append(-1)
                    self.nhj_dR.append(-1)
        else:
            for i in range(self.nJet):
                self.nhj_idx.append(-1)
                self.nhj_dR.append(-1)
        return self.nhj_idx

    def getJet_perpV(self,tree):
        '''not used'''
        if tree.GetReadEntry() == self.perpV_treeIdx:
            return self.perpV
        self.perpV_treeIdx = tree.GetReadEntry()
        self.perpV = []
        for idx in range(self.nJet):
            if idx < tree.nJet:
                Jet = ROOT.TVector3()
                V = ROOT.TVector3()
                Jet.SetPtEtaPhi(1,tree.Jet_eta[idx],tree.Jet_phi[idx])
                V.SetPtEtaPhi(1,tree.V_new_eta,tree.V_new_phi)
                self.perpV.append(Jet.Pt(V))
            else:
                self.perpV.append(-1)
        return self.perpV

    def getJet_nhJ_dR(self,tree):
        '''returns the delta R from the jet to the nearest b candidate'''
        self.getJet_nhJ_idx(tree)
        return self.nhj_dR

    def getJet_HV_pp(self,tree):
        '''not used'''
        if tree.GetReadEntry() == self.HVpp_treeIdx:
            return self.HVpp
        self.HVpp_treeIdx = tree.GetReadEntry()
        HV = self.getHV(tree)
        self.HVpp = []
        Jet = ROOT.TLorentzVector()
        for idx in range(self.nJet):
            if idx < tree.nJet:
                Jet.SetPtEtaPhiM(tree.Jet_pt[idx],tree.Jet_eta[idx],tree.Jet_phi[idx],tree.Jet_mass[idx])
                self.HVpp.append((HV * Jet) / (Jet.Pt() * HV.M()))
            else:
                self.HVpp.append(-1)
        return self.HVpp

    def getISRidx(self,tree,cut):
        '''returns a list of ISR indices, also saves a list of FSR indices to the buffer'''
        if tree.GetReadEntry() == self.ISR_treeIdx[cut] and self.ISRidx[cut] is not None:
            return self.ISRidx[cut]
        
        self.ISRidx[cut] = []
        self.FSRidx[cut] = []
        self.ISR_treeIdx[cut] = tree.GetReadEntry()
        nhJidx = self.getJet_nhJ_idx(tree)
        j = 0
        l = 0

        for idx in self.getAJidx(tree):
            if idx < 0 or self.nJet <= idx or nhJidx[idx] < 0:
                break
            
#            if cut == 'advCut':
#                HV = self.getHV(tree)
#                dRh = self.getJet_nhJ_dR(tree)[idx]
#                dRHV_ = self.dR_(tree.Jet_eta[idx],HV.Eta(),tree.Jet_phi[idx],HV.Phi())
#                dRHV = self.dR(tree.Jet_eta[idx],HV.Eta(),tree.Jet_phi[idx],HV.Phi())
#                fsr1 = dRh < 1.1 and (dRHV < 2.7 or 1.7 < dRHV_ < 3.2)
#                fsr2 = dRh < 0.7 
#                isr_cut = not fsr1 and not fsr2
#                fsr_cut = not isr_cut
#
#            if cut == 'ptCut':
#                isr_cut = tree.Jet_pt[idx] > 30
#                fsr_cut = not isr_cut
#
#            if cut == 'pt25':
#                isr_cut = tree.Jet_pt[idx] > 25
#                fsr_cut = not isr_cut
#
#            if cut == 'pt35':
#                isr_cut = tree.Jet_pt[idx] > 35
#                fsr_cut = not isr_cut
#
#            if cut == 'perpVCut2':
#                dRh = self.getJet_nhJ_dR(tree)[idx]
#                perp = self.getJet_perpV(tree)[idx]
#                isr_cut = dRh >= 0.9
#                #fsr_cut = (dRh-0.6)**2 / 0.3**2 + (np.log10(perp)-2.4)**2 < 1
#                fsr_cut = dRh < 0.9 and (np.log10(perp)/0.15)**2 + (dRh-1.75)**2 > 1 
#
#            if cut == 'perpVCut':
#                dRh = self.getJet_nhJ_dR(tree)[idx]
#                perp = self.getJet_perpV(tree)[idx]
#                isr_cut = dRh >= 0.9
#                #fsr_cut = (dRh-0.6)**2 / 0.3**2 + (np.log10(perp)-2.4)**2 < 1
#                fsr_cut = dRh < 0.9 and (np.log10(perp)/0.15)**2 + (dRh-1.6)**2 > 1 
#
#            if cut == 'dRCut':
#                dRh = self.getJet_nhJ_dR(tree)[idx]
#                isr_cut = dRh > 0.8
#                fsr_cut = not isr_cut
#
#            if cut == 'combo':
#                dRh = self.getJet_nhJ_dR(tree)[idx]
#                isr_cut = dRh > 0.8 and tree.Jet_pt[idx] > 30
#                fsr_cut = not isr_cut
#
#            if cut == 'dRppCut':
#                dRh = self.getJet_nhJ_dR(tree)[idx]
#                pp = self.getJet_HV_pp(tree)[idx]
#                isr_cut = (dRh-0.6)**2 + (np.log10(pp-0.6)/1.5 + 0.1)**2 > 0.3**2
#                fsr_cut = not isr_cut
            if cut == '30':
                isr_cut = tree.Jet_pt[idx] > 30
                fsr_cut = isr_cut and self.getJet_nhJ_dR(tree)[idx] < 0.8

            if cut == 'excl':
                dRh = self.getJet_nhJ_dR(tree)[idx]
                isr_cut = dRh > 0.8 and tree.Jet_pt[idx] > 30
                fsr_cut = not isr_cut

            if cut == '08':
                dRh = self.getJet_nhJ_dR(tree)[idx]
                isr_cut = dRh > 0.8
                fsr_cut = not isr_cut


            if isr_cut:
                self.ISRidx[cut].append(idx)
                j += 1
            if fsr_cut:
                self.FSRidx[cut].append(idx)
                l += 1
        
        self.nISR[cut] = j
        self.nFSR[cut] = l
        for i in range(j,self.maxnAJ):
            self.ISRidx[cut].append(-1)
        for i in range(l,self.maxnAJ):
            self.FSRidx[cut].append(-1)
        return self.ISRidx[cut]

    def getFSRidx(self,tree,cut):
        '''returns a list of FSR indices'''
        self.getISRidx(tree,cut)
        return self.FSRidx[cut]
