#!/usr/bin/env python
import array

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

class TreeFormula(object):

    def __init__(self, branchName, formula, branchType='f'):
        self.branchName = branchName
        self.branchType = branchType
        self.formula = formula

    def customInit(self, initVars):
        self.sampleTree = initVars['sampleTree']
        self.sampleTree.addFormula(formula)
        self.branches = [{'name': self.branchName, 'formula': lambda x: self.sampleTree.evaluate(formula), 'type': self.branchType}]

    def getBranches(self):
        return self.branches

class Copy(object):

    def __init__(self, branchName, formula, branchType='f'):
        self.branchName = branchName
        self.branchType = branchType
        self.formula = formula

    def customInit(self, initVars):
        self.branches = [{'name': self.branchName, 'formula': lambda x: getattr(x, self.formula), 'type': self.branchType}]

    def getBranches(self):
        return self.branches

class Collection(object):

    def __init__(self, name, properties, maxSize=1):
        self.name = name
        self.properties = properties
        self.maxSize = maxSize
        self.size = maxSize
        self.scalar = (self.maxSize < 2)
        self.branches = []
        self.branchBuffers = {}

        if not self.scalar:
            numBranchName = 'n' + self.name
            self.branchBuffers['n'] = array.array('i', [0])
            self.branches.append({'name': numBranchName, 'formula': self.getBranch, 'arguments': 'n', 'type': 'i'})

        for prop in self.properties:
            # properties given as list oif dicts: [{'name':'bla', 'type':'f'},...]
            if type(prop) == dict:
                propBranch = self.name + '_' + prop['name']
                propType = prop['type'] if 'type' in prop else 'f'
                bufferName = prop['name']
            # properties given as list of strings ['bla', ...]
            else:
                propType = 'f'
                propBranch = self.name + '_' + prop
                bufferName = prop
            self.branchBuffers[bufferName] = array.array(propType, [0.0] * self.maxSize)
            leaflist = '{branch}[{size}]/{type}'.format(branch=propBranch, size=numBranchName, type=propType.upper()) if not self.scalar else '{branch}/{type}'.format(branch=propBranch, type=propType.upper())
            if self.scalar:
                self.branches.append({
                        'name': propBranch,
                        'formula': self.getBranch,
                        'arguments': bufferName,
                        'type': propType,
                    })
            else:
                self.branches.append({
                        'name': propBranch,
                        'formula': self.fillVectorBranch,
                        'arguments': {'branch': bufferName, 'size': 'n'},
                        'length': self.maxSize,
                        'leaflist': leaflist,
                    })

    # direct access to branch arrays
    def __getitem__(self, key):
        return self.branchBuffers[key]

    # only for vectors, don't use with scalars
    def setSize(self, size):
        self.branchBuffers['n'][0] = size

    def getSize(self):
        return self.branchBuffers['n'][0]

    def getBranches(self):
        return self.branches

    def getBranch(self, event, arguments):
        return self.branchBuffers[arguments][0]

    def fillVectorBranch(self, event, arguments=None, destinationArray=None):
        size = 1
        if 'size' in arguments:
            if arguments['size'] in self.branchBuffers:
                size = self.branchBuffers[arguments['size']][0]
            elif type(arguments['size']) == int:
                size = arguments['size']
            elif type(arguments['size']) == str:
                size = getattr(event, arguments['size'])
        destinationArray[:size] = self.branchBuffers[arguments['branch']][:size]

class AddCollectionsModule(object):

    def __init__(self):
        self.lastEntry = -1
        self.branches = []
        self.branchBuffers = {}
        self.collections = {}

    def addCollection(self, collection):
        if collection.name not in self.collections:
            self.collections[collection.name] = collection
        else:
            raise Exception("CollectionAlreadyExists")
        self.branches += collection.getBranches()

    def getBranches(self):
        return self.branches

    def hasBeenProcessed(self, tree):
        return tree.GetReadEntry() == self.lastEntry

    def markProcessed(self, tree):
        self.lastEntry = tree.GetReadEntry()
