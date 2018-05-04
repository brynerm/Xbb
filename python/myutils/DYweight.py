#!/usr/bin/env python
import ROOT

# DY weight
class DYweight(object):

    def __init__(self, sample, tree):
        self.branches = [{'name': 'DY_specialWeight', 'formula': self.getDYspecialWeight}]
        self.sample = sample
        self.specialWeightFormula =  ROOT.TTreeFormula('specialWeight', self.sample.specialweight, tree)

    def getBranches(self):
        return self.branches
    
    def getDYspecialWeight(self, tree=None):
        specialWeight = 1.0
        if self.sample.specialweight:
            specialWeight = self.specialWeightFormula.EvalInstance()
        return specialWeight

