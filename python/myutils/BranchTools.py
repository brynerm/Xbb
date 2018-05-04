#!/usr/bin/env python

# if a branch does not exist, add it to the tree with a default value
class DefaultIfNotExisting(object):
    
    def __init__(self, branchName, branchType='f', defaultValue=0.0):
        self.debug = False
        self.branches = []
        self.branchName = branchName
        self.branchType = branchType
        self.defaultValue = defaultValue

    def customInit(self, initVars):
        self.sample = initVars['sample']
        self.tree = initVars['tree']
        if not self.tree.GetListOfBranches().FindObject(self.branchName):
            self.branches.append({'name': self.branchName, 'formula': lambda x: self.defaultValue, 'type': self.branchType})

    def getBranches(self):
        return self.branches
