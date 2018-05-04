#!/usr/bin/env python
import ROOT
import numpy as np
import array
import os
import math

# jet regression and systematics
class JetEnergySystematics(object):

    def __init__(self, channel='', weights=None, nano=False):
        self.channel = channel
        self.regWeightFileName = weights
        self.nano = nano

    def customInit(self, initVars):
        self.config = initVars['config']
        self.sample = initVars['sample']
        self.lastEntry = -1
        self.branchBuffers = {}
        self.branches = []
        
        self.doGroup = False
        self.isVerbose = False
        
        #regWeight = './reg/ttbar-G25-500k-13d-300t.weights.xml'
        #regWeight = './csv/gravall-v25.weights.xml'
        regWeight = self.regWeightFileName if self.regWeightFileName else config.get("TrainRegression","regWeight")
        #regWeight = './reg/TMVARegression_BDTG.weights.xml'
        self.regVars = eval(config.get("TrainRegression","regVars")) #attention: the var names and assigned expressions are still partitally hard coded in processEvent() !!!!!!!!
        self.regDict = eval(config.get("TrainRegression","regDict"))
        self.useVarAbbr = int(config.get("TrainRegression","useVarAbbr"))
        self.regDef = eval(config.get("TrainRegression","regDef"))
#        self.regVars = ["Jet_pt",
#                   "nPVs",
#                   "Jet_eta",
#                   "Jet_mt",
#                   "Jet_leadTrackPt",
#                   "Jet_leptonPtRel",
#                   "Jet_leptonPt",
#                   "Jet_leptonDeltaR",
#                   "Jet_neHEF",
#                   "Jet_neEmEF",
#                   "Jet_vtxPt",
#          #         "Jet_vtxMass",
#                   "Jet_vtx3dL",
#                   "Jet_vtxNtrk",
#                   "Jet_vtx3deL"
#                   #"met_pt",
#                   #"Jet_met_proj"
#                   ]
#        regDict = {'Jet_leadTrackPt': 'Alt$(Jet_leadTrackPt[hJCidx],0)',
#                'Jet_vtxNtrk': 'Alt$(Jet_vtxNtracks[hJCidx],0)',
#                'Jet_eta': 'Alt$(Jet_eta[hJCidx],0)',
#                "Jet_vtx3deL": 'Alt$(Jet_vtx3DSig[hJCidx],0)',
#                "Jet_pt": 'Alt$(Jet_pt[hJCidx],0)',
#                "Jet_neEmEF": 'Alt$(Jet_neEmEF[hJCidx],0)',
#                "Jet_leptonPtRel": 'max(Alt$(Jet_leptonPtRel[hJCidx],0),0)',
#                "Jet_leptonPt": 'max(Alt$(Jet_leptonPt[hJCidx],0),0)',
#                "Jet_vtxPt": 'Alt$(Jet_vtxPt[hJCidx],0)'
#                "Jet_vtx3dL":'max(Alt$(Jet_vtx3DVal[hJCidx],0),0)',
#                "Jet_mt":'VHbb::evalMtFromPtEtaPhiM(Alt$(Jet_pt[hJCidx],0),Alt$(Jet_eta[hJCidx],0),Alt$(Jet_phi[hJCidx],0),Alt$(Jet_mass[hJCidx],0))'
#                "Jet_neHEF":'Alt$(Jet_neHEF[hJCidx],0)'
#                "Jet_leptonDeltaR": 'max(Alt$(Jet_leptonDeltaR[hJCidx],0),0)',
#                'nPVs': 'nPVs'}

        #        self.regDict = {"Jet_pt":"Jet_pt[hJCMVAV2idx[0]]",
#                   #"Jet_corr":"Jet_corr[hJCMVAV2idx[0]]"
#                   "nPVs":"nPVs",
#                   "Jet_eta":"Jet_eta[hJCMVAV2idx[0]]",
#                   "Jet_mt":"Jet_mt[hJCMVAV2idx[0]]",
#                   "Jet_leadTrackPt": "Jet_leadTrackPt[hJCMVAV2idx[0]]",
#                   "Jet_leptonPtRel":"Jet_leptonPtRel[hJCMVAV2idx[0]]",
#                   "Jet_leptonPt":"Jet_leptonPt[hJCMVAV2idx[0]]",
#                   "Jet_leptonDeltaR":"Jet_leptonDeltaR[hJCMVAV2idx[0]]",
#                   "Jet_neHEF":"Jet_neHEF[hJCMVAV2idx[0]]",
#                   "Jet_neEmEF":"Jet_neEmEF[hJCMVAV2idx[0]]",
#                   "Jet_vtxPt":"Jet_vtxPt[hJCMVAV2idx[0]]",
#         #          "Jet_vtxMass":"Jet_vtxMass[hJCMVAV2idx[0]]",
#                   "Jet_vtx3dL":"Jet_vtx3dl[hJCMVAV2idx[0]]",
#                   "Jet_vtxNtrk":"Jet_vtxNtrk[hJCMVAV2idx[0]]",
#                   "Jet_vtx3deL":"Jet_vtx3deL[hJCMVAV2idx[0]]"
#                   #"met_pt":"met_pt[hJCMVAV2idx[0]]",
#                   #"Jet_met_proj":"Jet_met_proj[hJCMVAV2idx[0]]"
#                   }
        if self.doGroup:
            self.JECsys = [
                "JER",
                "PileUp",
                "Relative",
                "AbsoluteMisc"
                ]
            self.JECsysGroupDict = {
                "PileUp": ["PileUpDataMC",
                           "PileUpPtRef",
                           "PileUpPtBB",
                           "PileUpPtEC1",
                           "PileUpPtEC2",
                           "PileUpPtHF"],
                "Relative": ["RelativeJEREC1",
                             "RelativeJEREC2",
                             "RelativeJERHF",
                             "RelativeFSR",
                             "RelativeStatFSR",
                             "RelativeStatEC",
                             "RelativeStatHF",
                             "RelativePtBB",
                             "RelativePtEC1",
                             "RelativePtEC2",
                             "RelativePtHF"],
                "AbsoluteMisc": [ "AbsoluteScale",
                                  "AbsoluteMPFBias",
                                  "AbsoluteStat",
                                  "SinglePionECAL",
                                  "SinglePionHCAL",
                                  "Fragmentation",
                                  "TimePtEta",
                                  "FlavorQCD"]
                }
        else:
            self.JECsys = [
                "JER",
                "PileUpDataMC",
                "PileUpPtRef",
                "PileUpPtBB",
                "PileUpPtEC1",
                #"PileUpPtEC2",
                #"PileUpPtHF",
                "RelativeJEREC1",
                #"RelativeJEREC2",
                #"RelativeJERHF",
                "RelativeFSR",
                "RelativeStatFSR",
                "RelativeStatEC",
                #"RelativeStatHF",
                "RelativePtBB",
                "RelativePtEC1",
                #"RelativePtEC2",
                #"RelativePtHF",
                "AbsoluteScale",
                "AbsoluteMPFBias",
                "AbsoluteStat",
                "SinglePionECAL",
                "SinglePionHCAL",
                "Fragmentation",
                "TimePtEta",
                "FlavorQCD"
                ]
        self.VarList = ['HCMVAV2_reg_mass','HCMVAV2_reg_pt','HCMVAV2_reg_eta','HCMVAV2_reg_phi','hJetCMVAV2_pt_reg_0','hJetCMVAV2_pt_reg_1','hJetCMVAV2_pt_reg']

        for var in self.VarList:
            if not var == 'hJetCMVAV2_pt_reg':
                self.branchBuffers[var] = array.array('f', [0])
                self.branches.append({'name': var, 'formula': self.getBranch, 'arguments': var})
            else:
                self.branchBuffers[var] = array.array('f', [0.0]*21)
                self.branches.append({'name': var, 'formula': self.getVectorBranch, 'arguments': {'branch': var, 'length':21}, 'length': 21})

        if self.sample.type != 'DATA':
            for syst in self.JECsys:
                for sdir in ["Up", "Down"]:
                    for var in self.VarList:
                        #if not 'hJet' in var:
                        if not var == 'hJetCMVAV2_pt_reg':
                            branchName = var+"_corr"+syst+sdir
                            self.branchBuffers[branchName] = array.array('f', [0])
                            self.branches.append({'name': branchName, 'formula': self.getBranch, 'arguments': branchName})
                        else:
                            branchName = var+"_corr"+syst+sdir
                            self.branchBuffers[branchName] = array.array('f', [0.0]*21)
                            self.branches.append({'name': branchName, 'formula': self.getVectorBranch, 'arguments': {'branch': branchName, 'length':21}, 'length': 21})
        ##Compute MinMax
        ###Those branches will Max/Minimise the systematic values. Used to speed-up th dc step
        JEC_systematicsMinMax = {}
        VarListMinMax = ['HCMVAV2_reg_mass','HCMVAV2_reg_pt','HCMVAV2_reg_eta','HCMVAV2_reg_phi','hJetCMVAV2_pt_reg_0','hJetCMVAV2_pt_reg_1']
        if self.sample.type != 'DATA':
            for bound in ["Min", "Max"]:
                for var in VarListMinMax:
                    branchName = var+"_corr_"+bound
                    self.branchBuffers[branchName] = array.array('f', [0.0]*21)
                    self.branches.append({'name': branchName, 'formula': self.getVectorBranch, 'arguments': {'branch': branchName, 'length':21}, 'length': 21})

        # define all the readers
        self.TMVA_reader = {}
        self.theVars = {}

        self.TMVA_reader['readerJet'] = ROOT.TMVA.Reader("!Color:!Silent" )

        #add a dictionary of containing array of the vars in the TMVAreader

        # Init the TMVA readers
        self.addVarsToReader(self.TMVA_reader['readerJet'], self.theVars)
        self.TMVA_reader['readerJet'].BookMVA("readerJet", regWeight)
        print '\n\t ----> Evaluating Regression on sample....'

    def addVarsToReader(self, reader,theVars):
        for key in self.regVars:
            if self.useVarAbbr == 2:
                var = key
            elif self.useVarAbbr == 1:
                var = "%s:=%s" % (key, self.regDict[key])
            else:
                var = self.regDict[key]
            #print key
            theVars[key] = array.array( 'f', [ 0 ] )
            reader.AddVariable(var,theVars[key])
        return

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

    # recompute Vtype, return false to skip the event if Vtype does not match channel
    def processEvent(self, tree):
        isGoodEvent = True
        currentEntry = tree.GetReadEntry()
        if currentEntry != self.lastEntry:
            self.lastEntry = currentEntry
            # do processing
            hJ = ROOT.TLorentzVector()
            hJ0 = ROOT.TLorentzVector()
            hJ1 = ROOT.TLorentzVector()
            Reg_var_list = []
            for j in xrange(min(tree.nJet,21)):
                reg_var_dic = {}
                Jet_pt = tree.Jet_pt[j]
                Jet_eta = tree.Jet_eta[j]
                Jet_m = tree.Jet_mass[j]
                Jet_phi = tree.Jet_phi[j]

                #fill definitions from regression.ini
                for var in self.regVars:
                    reg_var_dic[var] = self.regDef[var](tree,j)

                #these are the regVars of which the definition is not given in the config but is done here
                reg_var_dic['hJ'] = hJ.SetPtEtaPhiM(Jet_pt, Jet_eta, Jet_phi, Jet_m)
                #reg_var_dic['Jet_e'] = hJ.E()
                reg_var_dic['Jet_mt'] = hJ.Mt()
                #reg_var_dic['Jet_met_proj']=projectionMETOntoJet(tree.met_pt, tree.met_phi, Jet_pt, Jet_phi)
                
                #these variables are used for further calculations and therefore always need to be defined
                reg_var_dic['Jet_m']= Jet_m
                reg_var_dic['Jet_phi'] = Jet_phi
                reg_var_dic['Jet_pt'] = Jet_pt
                reg_var_dic['Jet_eta'] = Jet_eta

                Reg_var_list.append(reg_var_dic)

            # JEC factorized branches
            Jec_sys_list = []
            if self.sample.type != 'DATA':
                for j in xrange(min(tree.nJet,21)):
                    jec_sys_dic = {}
                    jec_sys_dic['Jet_corr'] =  getattr(tree,'Jet_corr')[j]
                    jec_sys_dic['Jet_corr_JER'] =  getattr(tree,'Jet_corr_JER')[j]
                    for jecsys in self.JECsys:
                        for ud in ['Up', 'Down']:
                            jec_sys_dic[jecsys+ud] =  getattr(tree,'Jet_corr_'+jecsys+ud)[j]
                    Jec_sys_list.append(jec_sys_dic)

                #Fill regression vars used in TMVA
                    #now loop over all the jets
            for j in xrange(min(tree.nJet,21)):
                for key in self.regVars:
                    self.theVars[key][0] = Reg_var_list[j][key]

                Pt = max(0.0001, self.TMVA_reader['readerJet'].EvaluateRegression("readerJet")[0])
                rPt = Reg_var_list[j]['Jet_pt']*Pt
                self.branchBuffers["hJetCMVAV2_pt_reg"][j] = Pt
                JetCMVAV2_regWeight = Pt/Reg_var_list[j]['Jet_pt']
                #print 'pt is', Reg_var_list[j]['Jet_pt']
                #print 'reg pt is', Pt

                #Fill the Higgs jet
                if j == tree.hJCMVAV2idx[0]:
                    hJ0.SetPtEtaPhiM(Pt, Reg_var_list[j]['Jet_eta'], Reg_var_list[j]['Jet_phi'], Reg_var_list[j]['Jet_m']*JetCMVAV2_regWeight)
                elif j == tree.hJCMVAV2idx[1]:
                    hJ1.SetPtEtaPhiM(Pt, Reg_var_list[j]['Jet_eta'], Reg_var_list[j]['Jet_phi'], Reg_var_list[j]['Jet_m']*JetCMVAV2_regWeight)

            #print 'mass is', (hJ0+hJ1).M()
            self.branchBuffers["HCMVAV2_reg_mass"][0] = (hJ0+hJ1).M()
            self.branchBuffers["HCMVAV2_reg_pt"][0]   = (hJ0+hJ1).Pt()
            self.branchBuffers["HCMVAV2_reg_eta"][0]  = (hJ0+hJ1).Eta()
            self.branchBuffers["HCMVAV2_reg_phi"][0]  = (hJ0+hJ1).Phi()
            self.branchBuffers["hJetCMVAV2_pt_reg_0"][0]  = hJ0.Pt()
            self.branchBuffers["hJetCMVAV2_pt_reg_1"][0]  = hJ1.Pt()
            
            if self.sample.type != 'DATA':
                #now loop over all the jets
                for syst in self.JECsys:
                    for sdir in ["Up", "Down"]:
                        for j in xrange(min(tree.nJet,21)):
                            for key in self.regVars:
                                self.theVars[key][0] = Reg_var_list[j][key]
                            self.theVars['Jet_pt'][0] = 0

                            if syst == "JER":
                                self.theVars['Jet_pt'][0] = Reg_var_list[j]['Jet_rawPt']*Jec_sys_list[j]['Jet_corr']*Jec_sys_list[j]['JER'+sdir]
                            else:
                                #theVars['Jet_pt'][0] = Reg_var_list[j]['Jet_rawPt']*Jec_sys_list[j][syst+sdir]*Jec_sys_list[j]['Jet_corr_JER']
                                self.theVars['Jet_pt'][0] = (Reg_var_list[j]['Jet_pt']/Jec_sys_list[j]['Jet_corr'])*Jec_sys_list[j][syst+sdir]

                            pt = max(0.0001, self.TMVA_reader['readerJet'].EvaluateRegression("readerJet")[0])

                            rPt = Reg_var_list[j]['Jet_pt']*pt
                            self.branchBuffers["hJetCMVAV2_pt_reg_corr"+syst+sdir][j] = pt
                            Jet_regWeight= pt/Reg_var_list[j]['Jet_pt']

                            if j == tree.hJCMVAV2idx[0]:
                                hJ0.SetPtEtaPhiM(pt, Reg_var_list[j]['Jet_eta'], Reg_var_list[j]['Jet_phi'], Reg_var_list[j]['Jet_m']*Jet_regWeight)
                            elif j == tree.hJCMVAV2idx[1]:
                                hJ1.SetPtEtaPhiM(pt, Reg_var_list[j]['Jet_eta'], Reg_var_list[j]['Jet_phi'], Reg_var_list[j]['Jet_m']*Jet_regWeight)

                        self.branchBuffers["HCMVAV2_reg_mass_corr"+syst+sdir][0] = (hJ0+hJ1).M()
                        self.branchBuffers["HCMVAV2_reg_pt_corr"+syst+sdir][0] = (hJ0+hJ1).Pt()
                        self.branchBuffers["HCMVAV2_reg_eta_corr"+syst+sdir][0] = (hJ0+hJ1).Eta()
                        self.branchBuffers["HCMVAV2_reg_phi_corr"+syst+sdir][0] = (hJ0+hJ1).Phi()
                        self.branchBuffers["hJetCMVAV2_pt_reg_0_corr"+syst+sdir][0] = hJ0.Pt()
                        self.branchBuffers["hJetCMVAV2_pt_reg_1_corr"+syst+sdir][0] = hJ1.Pt()

                #Compute Min/Max
                VarList = ['HCMVAV2_reg_mass','HCMVAV2_reg_pt','HCMVAV2_reg_eta','HCMVAV2_reg_phi','hJetCMVAV2_pt_reg_0','hJetCMVAV2_pt_reg_1']
                for var in VarList:
                    for bound in ['Min','Max']:
                        #intialise by using central value (no sys)
                        #new
                        val = self.branchBuffers[var][0]
                        #old
                        #val =  getattr(tree,var)
                        for syst in self.JECsys:
                            for sdir in ["Up", "Down"]:
                                #new
                                if bound == 'Min': val = min(val, self.branchBuffers[var+"_corr"+syst+sdir])
                                if bound == 'Max': val = max(val, self.branchBuffers[var+"_corr"+syst+sdir])
                                #old
                                #if bound == 'Min': val = min(val, getattr(tree,var+"_corr"+syst+sdir))
                                #if bound == 'Max': val = max(val, getattr(tree,var+"_corr"+syst+sdir))
                        if type(val) != float:
                            val = val[0]
                        self.branchBuffers[var+"_corr_"+bound][0] = val
                        #print 'val is', val
        return isGoodEvent
