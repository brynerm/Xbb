#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import hashlib
from optparse import OptionParser
from myutils.FileLocator import FileLocator
from myutils.FileList import FileList
from myutils.sampleTree import SampleTree

# merges files only partially to reduce IO overhead and balance job load
class PartialFileMerger(object):
    def __init__(self, fileNames, chunkNumber, submitTime='000000_000000', force=False, config=None, sampleIdentifier=None):
        self.fileNames = fileNames
        self.debug = 'XBBDEBUG' in os.environ
        self.submitTime = submitTime
        self.chunkNumber = chunkNumber
        self.config = config
        self.fileLocator = FileLocator(config=self.config)
        # -O option (reoptimizing baskets) leads to crashes...
        self.commandTemplate = "hadd -k  -ff {output} {inputs}"
        self.sampleIdentifier = sampleIdentifier
        self.force = force
        
        # use sampleTree class as replacement for hadd
        self.useChain = True

        treeHashes = []
        for fileName in self.fileNames: 
            treeHashes.append(hashlib.sha224(fileName).hexdigest())
        totalHash = hashlib.sha224('-'.join(sorted(treeHashes))).hexdigest()
        self.mergedFileName = '/'.join(self.fileNames[0].split('/')[:-4]) + '/' + totalHash + '/' + self.submitTime + '/0000/tree_%d.root'%chunkNumber

    # return a fake name which is written to sample list .txt files in order to keep compatibility to the method of converting file names in .txt
    # files to file names after prep step. This conversion applied to the fake name will give the real file name.
    def getMergedFakeFileName(self):
        return self.mergedFileName

    # real output file name where the file is stored
    def getOutputFileName(self):
        fakeFileName = self.getMergedFakeFileName()
        outputFileName = self.fileLocator.getFilenameAfterPrep(fakeFileName) 
        return "{path}/{sample}/{fileName}".format(path=self.config.get('Directories','HADDout'), sample=self.sampleIdentifier, fileName=outputFileName)
    
    def getTemporaryFileName(self):
        fakeFileName = self.getMergedFakeFileName()
        outputFileName = self.fileLocator.getFilenameAfterPrep(fakeFileName) 
        return "{path}/hadd/{sample}/{fileName}".format(path=self.config.get('Directories','scratch'), sample=self.sampleIdentifier, fileName=outputFileName)
    
    def run(self):
        inputFileNames = ["{path}/{sample}/{fileName}".format(path=self.config.get('Directories','HADDin'), sample=self.sampleIdentifier, fileName=self.fileLocator.getFilenameAfterPrep(fileName)) for fileName in self.fileNames]
        outputFileName = self.getTemporaryFileName()
        self.fileLocator.makedirs('/'.join(outputFileName.split('/')[:-1]))
        command = self.commandTemplate.format(output=outputFileName, inputs=' '.join(inputFileNames), f="-f" if self.force else "")
        if self.debug:
            print ("DEBUG: run \x1b[34m", command, "\x1b[0m")
        
        if self.useChain:
            # use sampleTree class (can e.g. drop branches at the same time)
            sampleTree = SampleTree(inputFileNames, config=self.config)

            try:
                removeBranches = eval(self.config.get('General', 'remove_branches'))
                for removeBranch in removeBranches:
                    sampleTree.addBranchToBlacklist(removeBranch)
                    print("DEBUG: disable branch ", removeBranch)
            except Exception as e:
                print("DEBUG: could not disable branch:", e)
            sampleTree.addOutputTree(outputFileName, cut='1', branches='*')
            sampleTree.process()
            result = 0
        else:
            # standard hadd
            result = self.fileLocator.runCommand(command)

        print ("INFO: hadd returned ", result)
        if result == 0:
            finalOutputFileName = self.getOutputFileName()
            print("move file to final destination: \x1b[34m", finalOutputFileName, "\x1b[0m")
            self.fileLocator.makedirs('/'.join(finalOutputFileName.split('/')[:-1]))
            resultCopy = self.fileLocator.cp(outputFileName, finalOutputFileName, self.force)
            if not resultCopy:
                print("\x1b[31mERROR: copy failed\n from:", outputFileName, "\n to:", finalOutputFileName, "\n force:", self.force, "\x1b[0m")
                raise Exception("FileCopyError")
            # try to delete temporary file
            try:
                self.fileLocator.rm(outputFileName)
            except Exception as e:
                print("ERROR: could not delete temporary file:", outputFileName, " => ", e)
            print("INFO: done.")
        else:
            raise Exception("HaddError")

if __name__ == "__main__":
    
    # read arguments
    argv = sys.argv
    parser = OptionParser()
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                              help="Verbose mode.")
    parser.add_option("-C", "--config", dest="config", default=[], action="append",
                          help="configuration file")
    parser.add_option("-s","--sampleIdentifier", dest="sampleIdentifier", default='',
                          help="sample identifier (no subsample!)")
    parser.add_option("-i","--chunkNumber", dest="chunkNumber", default='',
                          help="number of part to cache")
    parser.add_option("-f","--force", action="store_true", dest="force", default=False,
                          help="force overwriting of already cached files")
    parser.add_option("-l","--fileList", dest="fileList", default="",
                          help="file list")
    (opts, args) = parser.parse_args(argv)
    if opts.config == "":
            opts.config = "config"

    # Import after configure to get help message
    from myutils import BetterConfigParser, mvainfo, ParseInfo

    # load config
    config = BetterConfigParser()
    config.read(opts.config)

    partialFileMerger = PartialFileMerger(FileList.decompress(opts.fileList), int(opts.chunkNumber), config=config, sampleIdentifier=opts.sampleIdentifier, force=opts.force)
    partialFileMerger.run()

