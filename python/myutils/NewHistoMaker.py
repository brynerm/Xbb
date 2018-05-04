from __future__ import print_function
import ROOT
import TdrStyles
import os

from NewTreeCache import TreeCache as TreeCache
from sampleTree import SampleTree

class NewHistoMaker:

    instanceCounter = 0

    def __init__(self, config, sample, sampleTree, histogramOptions):
        NewHistoMaker.instanceCounter += 1
        self.debug = 'XBBDEBUG' in os.environ
        self.config = config
        self.sample = sample
        self.sampleTree = sampleTree
        self.histogramOptions = histogramOptions
        self.histogram = None
        self.EvalCut = config.get('Cuts', 'EvalCut')
        TdrStyles.tdrStyle()

    def initializeHistogram(self):
        self.histogramName = (self.histogramOptions['name']+'_') if 'name' in self.histogramOptions else ''
        self.histogramName += self.sample.name + ('_' + self.histogramOptions['var'] if 'var' in self.histogramOptions else '')
        # add unique instance counter to avoid same name for histogram and ROOT complaining
        if 'uniqueid' in self.histogramOptions and self.histogramOptions['uniqueid']:
            self.histogramName += '_instance%d'%NewHistoMaker.instanceCounter
        is2D = ':' in self.histogramOptions['treeVar'].replace('::', '')
        if is2D:
            self.histogram = ROOT.TH2F(self.histogramName, self.histogramName, self.histogramOptions['nBinsX'], self.histogramOptions['minX'], self.histogramOptions['maxX'], self.histogramOptions['nBinsY'], self.histogramOptions['minY'], self.histogramOptions['maxY'])
        else:
            self.histogram = ROOT.TH1F(self.histogramName, self.histogramName, self.histogramOptions['nBinsX'], self.histogramOptions['minX'], self.histogramOptions['maxX'])
        self.histogram.Sumw2()
        self.histogram.SetTitle(self.sample.name)
        return self.histogram

    def scaleHistogram(self):
        if self.sample.type != 'DATA':
            ScaleFactor = self.sampleTree.getScale(self.sample)
            if ScaleFactor != 0:
                self.histogram.Scale(ScaleFactor)
            else:
                print ("\x1b[31mWARNING: histogram scaling factor is 0!\x1b[0m")

    def getHistogram(self, cut='1'):
        if self.initializeHistogram():
            # apply weights only to MC and not to DATA
            if 'group' in self.histogramOptions and self.histogramOptions['group'] == 'DATA':
                weightF = '1'
            else:
                weightF = "({weight})".format(weight=self.histogramOptions['weight'] if ('weight' in self.histogramOptions and self.histogramOptions['weight']) else '1') 

            # drop training events and rescale MC by 2 for BDT plots
            if 'BDT' in self.histogramOptions['treeVar'] and self.sample.type != 'DATA':
                cut = '(({cut1})&&({cut2}))'.format(cut1=cut, cut2=self.EvalCut)
                weightF = '(({weight1})*({weight2}))'.format(weight1=weightF, weight2='2.0')
                print("INFO: training events removed for \x1b[32m", self.histogramOptions['treeVar'], "\x1b[0m plot with additional cut \x1b[35m", self.EvalCut, "\x1b[0m, MC rescaled by \x1b[36m2.0\x1b[0m")

            # per sample special weight
            if self.config.has_option('Weights', 'useSpecialWeight') and eval(self.config.get('Weights', 'useSpecialWeight')):
                specialweight = self.sample.specialweight
                weightF = "(({weight})*({specialweight}))".format(weight=weightF, specialweight=specialweight)
                print ("INFO: use specialweight: {specialweight}".format(specialweight=specialweight))

            # (deprecated) additional blinding cut applied to DATA only in BDT plots
            if 'BDT' in self.histogramOptions['treeVar'] and self.sample.type == 'DATA':
                if self.config.has_section('Blinding') and self.config.has_option('Blinding', 'BlindBDTinSR'):
                    self.blindingCut = self.config.get('Blinding', 'BlindBDTinSR')
                    cut = '(({cut1})&&({cut2}))'.format(cut1=cut, cut2=self.blindingCut)
                    print ("\x1b[31mUSE BLINDING CUT FOR DATA in BDT!!", self.blindingCut ,"\x1b[0m")

            # (deprecated) additional mass blinding
            if 'mass' in self.histogramOptions['treeVar'] and self.sample.type == 'DATA':
                if self.config.has_section('Blinding') and self.config.has_option('Blinding', 'BlindDataInMassPlots') and eval(self.config.get('Blinding', 'BlindDataInMassPlots')):
                    self.blindingCut = '0'
                    cut = '(({cut1})&&({cut2}))'.format(cut1=cut, cut2=self.blindingCut)
                    print ("\x1b[31mUSE BLINDING CUT FOR DATA in BDT!!", self.blindingCut ,"\x1b[0m")

            # region/var specific blinding cut
            # TODO: switch to this type only
            if 'blindCut' in self.histogramOptions and self.sample.type == 'DATA':
                cut = '(({cut1})&&({cut2}))'.format(cut1=cut, cut2=self.histogramOptions['blindCut'])
                print("INFO: found region/var specific blinding cut in histogramOptions:\n--->{cut}".format(cut=self.histogramOptions['blindCut']))

            # add tree cut 
            selection = "({weight})*({cut})".format(weight=weightF, cut=cut) 
            nEvents = self.sampleTree.tree.Draw('{var}>>{histogramName}'.format(var=self.histogramOptions['treeVar'], histogramName=self.histogramName), selection)
            if nEvents < 0:
                print ("\x1b[31mERROR: error in TTree:Draw! returned {nEvents}\x1b[0m".format(nEvents=nEvents))
            if self.debug:
                print(selection, " => # events:", nEvents, "weighted:", self.histogram.Integral())
            self.scaleHistogram()
        else:
            print ("ERROR: initialization of histogram failed!")
            raise Exception("HistoMakerInitializationError")

        if 'addOverFlow' in self.histogramOptions and self.histogramOptions['addOverFlow']:
            uFlow = self.histogram.GetBinContent(0)+self.histogram.GetBinContent(1)
            oFlow = self.histogram.GetBinContent(self.histogram.GetNbinsX()+1)+self.histogram.GetBinContent(self.histogram.GetNbinsX())
            uFlowErr = ROOT.TMath.Sqrt(ROOT.TMath.Power(self.histogram.GetBinError(0),2)+ROOT.TMath.Power(self.histogram.GetBinError(1),2))
            oFlowErr = ROOT.TMath.Sqrt(ROOT.TMath.Power(self.histogram.GetBinError(self.histogram.GetNbinsX()),2)+ROOT.TMath.Power(self.histogram.GetBinError(self.histogram.GetNbinsX()+1),2))
            self.histogram.SetBinContent(1,uFlow)
            self.histogram.SetBinContent(self.histogram.GetNbinsX(),oFlow)
            self.histogram.SetBinError(1,uFlowErr)
            self.histogram.SetBinError(self.histogram.GetNbinsX(),oFlowErr)

        return self.histogram

