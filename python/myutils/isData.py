#!/usr/bin/env python

class isData(object):

    def __init__(self, nano=False):
        self.nano = nano

    def customInit(self, initVars):
        sample = initVars['sample']
        self.isData = sample.type == 'DATA'
        self.branches = [{'name': 'isData', 'formula': self.getBranch}]

    def getBranches(self):
        return self.branches

    def getBranch(self, tree):
        return self.isData



