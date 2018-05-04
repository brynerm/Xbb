#!/usr/bin/env python
import ROOT

class Drop(object):
    def __init__(self, dropBranches):
        self.dropBranches = dropBranches

    def customInit(self, initVars):
        self.sampleTree = initVars['sampleTree']
        for dropBranch in self.dropBranches:
            self.sampleTree.SetBranchStatus(dropBranch, 0)

    def getBranches(self):
        return []
