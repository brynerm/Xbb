from __future__ import print_function

'''Takes a string (or list of strings) as input (should be a cut/weigt/variable expression) and returns a list containing all the corresponding branches'''
class BranchList(object):
    def __init__(self, cuts=None):
        self.cuts = []
        if cuts:
            self.addCut(cuts)

    def flattenDict(self, cutDict):
        if type(cutDict) == str:
            return cutDict
        elif type(cutDict) == dict:
            if 'OR' in cutDict:
                return '||'.join(['(%s)'%self.flattenDict(x) for x in cutDict['OR']])
            elif 'AND' in cutDict:
                return '&&'.join(['(%s)'%self.flattenDict(x) for x in cutDict['AND']])
            else:
                raise Exception('BadTreeTypeCutDict')

    # any expression which can be used to construct a TTreeFormula can be added here as a string or list of strings
    def addCut(self, cuts):
        if type(cuts) == list:
            self.cuts += cuts
        elif type(cuts) == dict:
            self.cuts.append(self.flattenDict(cuts))
        else:
            self.cuts.append(cuts)
        return self

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # return list of branches (as string) used by any of the given cuts or weights. One still has to check if the branches
    #  really exist in the trees before calling SetBranchStatus(), otherwise it will print tons of warnings and will be slow.
    def getListOfBranches(self):
        operators = ['[', ']', '*', '/', '(', ')', '||','|', '<', '>', '=', '.', '&&', '&', '+', '-', ',', ' ', '!']
        separator = '??'
        branches = []
        for cut in self.cuts:
            for operator in operators:
                cut = cut.replace(operator, separator)
            leaves = cut.split(separator)
            for leaf in leaves:
                if len(leaf.strip()) > 0 and not self.isfloat(leaf) and not leaf.endswith('$') and '::' not in leaf:
                    branches.append(leaf)
        branchesUnique = list(set(branches))
        return branchesUnique

    # return an abbreviated list for print-out purpose
    def getShortRepresentation(self):
        listOfBranches = self.getListOfBranches()
        shortNames = {}
        for branch in listOfBranches:
            shortName = branch.split('_')[0]
            if shortName in shortNames:
                shortNames[shortName] += 1
            else:
                shortNames[shortName] = 1
        shortRepresentation = '{num} branches: '.format(num=len(listOfBranches))
        shortRepresentation += ', '.join(['\x1b[35m{name}\x1b[0m'.format(name=shortName) if num == 1 else '{num} like \x1b[34m{name}_*\x1b[0m'.format(num=num, name=shortName) for shortName, num in shortNames.iteritems()])
        return shortRepresentation

