from __future__ import print_function
import os
import shutil
import sys
import subprocess
import ROOT
import hashlib
from time import sleep

class FileLocator(object):

    def __init__(self, config=None, xrootdRedirector=None, usePythonXrootD=True):
        self.config = config
        self.debug = 'XBBDEBUG' in os.environ
        self.timeBetweenAttempts = 30
        try:
            self.xrootdRedirectors = [x.strip() for x in self.config.get('Configuration', 'xrootdRedirectors').split(',') if len(x.strip())>0]
            if len(self.xrootdRedirectors) < 1:
                print("WARNING: empty list of xrootd redirectors given!")
        except Exception as e:
            if config:
                print (e)
                print("WARNING: no xrootd redirector given!")
            self.xrootdRedirectors = None

        # needed for cases there are /store/ path accessible locally but with additional prefix path
        try:
            self.pnfsStoragePath = self.config.get('Configuration', 'pnfsStoragePath').strip()
        except:
            if config:
                print("WARNING: no pnfs storage path given!")
            self.pnfsStoragePath = None

        # if redirector is given as argument, use it as the main redirector
        if xrootdRedirector:
            if self.xrootdRedirectors:
                self.xrootdRedirectors = [xrootdRedirector]+self.xrootdRedirectors
            else:
                self.xrootdRedirectors = [xrootdRedirector]

        # use python bindings for xrootd (can be disabled setting the 'usePythonXrootD' option to False
        self.usePythonXrootD = eval(self.config.get('Configuration', 'usePythonXrootD')) if self.config and self.config.has_option('Configuration', 'usePythonXrootD') else True
        self.client = None
        self.server = None
        if self.usePythonXrootD or usePythonXrootD:
            try:
                from XRootD import client
                self.server = self.xrootdRedirectors[0].strip('/')
                self.client = client.FileSystem(self.server)
                if self.debug:
                    print('DEBUG: initialized xrootd client, server:', self.server)
                    print('DEBUG: client:', self.client)
            except:
                if self.debug:
                    print('DEBUG: xrootd could not be initialized, trying to use xrdfs as fallback. To use the faster Python bindings upgrade CMSSW to version 9.')

        # prefixes to distinguish remote file paths from local ones
        self.storagePathPrefix = '/store/'
        self.pnfsPrefix = '/pnfs/'
        self.remotePrefixes = [self.storagePathPrefix, self.pnfsPrefix]
        if self.config and self.config.has_option('Configuration', 'remotePrefixes'):
            self.remotePrefixes = self.config.get('Configuration', 'remotePrefixes').split(',')

        # TODO: use XrootD python bindings
        self.remoteStatDirectory = 'xrdfs {server} stat -q IsDir {path}'
        self.remoteStatFile = 'xrdfs {server} stat {path}'
        self.remoteMkdir = 'xrdfs {server} mkdir {path}'
        self.remoteRm = 'xrdfs {server} rm {path}'
        self.remoteCp = 'xrdcp -d 1 -f --posc --nopbar {source} {target}'
        self.makedirsMinLevel = 5   # don't even try to create/access the 5 lowest levels in the path

    # special Xbb function: get filename after prep step from original file name
    def getFilenameAfterPrep(self, inputFile):
        # this allows to have duplicate tree names, e.g. to import 2 trees which are both called tree_1.root
        if self.config and self.config.has_option('Configuration', 'AllowDuplicateTrees') and eval(self.config.get('Configuration', 'AllowDuplicateTrees')):
            subfolder = '_'.join(inputFile.split('/')[-4:-1])
        else:
            subfolder = inputFile.split('/')[-4]
        filename = inputFile.split('/')[-1]
        filename = filename.split('_')[0]+'_'+subfolder+'_'+filename.split('_')[1]
        hash = hashlib.sha224(filename).hexdigest()
        return filename.replace('.root','')+'_'+str(hash)+'.root'

    # check if path is relative to /store/
    def isStoragePath(self, path):
        return path.startswith(self.storagePathPrefix)

    # check if a path is a PNFS directory (with or without redirector)
    def isPnfs(self, path):
        return self.pnfsPrefix in path

    # check if the path is remote (= has to be accessed via xrootd)
    def isRemotePath(self, path):
        return self.isPnfs(path) or self.isStoragePath(path) or '://' in path

    def isValidRootFile(self, path):
        f = ROOT.TFile.Open(path, 'read')
        if f:
            isValid = not (f.IsZombie() or f.GetNkeys() == 0 or f.TestBit(ROOT.TFile.kRecovered))
            try:
                f.Close()
            except:
                pass
            return isValid
        else:
            return False

    def runCommand(self, command, forceDebug=False):
        if self.debug or forceDebug:
            print ("RUN: \x1b[32m",command,"\x1b[0m")
            return subprocess.call([command], shell=True)
        else:
            with open(os.devnull, 'w') as fp:
                result = subprocess.call([command], shell=True, stdout=fp, stderr=fp)
            return result 

    def getRemoteFileserver(self, path=None):
        if path and '://' in path:
            server = path.split('://')[1].split('/')[0].split(':')[0].strip()
        else:
            server = self.xrootdRedirectors[0].split('://')[1].split('/')[0].split(':')[0].strip()
        return server

    def getRedirector(self, path):
        if '://' in path:
            redirector = path.split('://')[0] + '://' + path.split('://')[1].split('/')[0]
        else:
            redirector = None
        return redirector

    def remoteDirectoryExists(self, path):
        statCommand = self.remoteStatDirectory.format(server=self.getRemoteFileserver(path), path=self.getLocalFileName(path))
        result = self.runCommand(statCommand)
        return result==0

    def remoteFileExists(self, path):
        if self.client:
            pathOnServer = self.removeRedirector(path.strip())
            response = self.client.stat(pathOnServer)
            existing = response[0].ok
            if self.debug:
                print('DEBUG: remoteFileExists("' + pathOnServer + '")')
                print('DEBUG: remoteFileExists() returned', response[0])
        else:
            statCommand = self.remoteStatFile.format(server=self.getRemoteFileserver(path), path=self.getLocalFileName(path))
            result = self.runCommand(statCommand)
            existing = result==0
        return existing

    def remoteFileRm(self, path):
        command = self.remoteRm.format(server=self.getRemoteFileserver(path), path=self.getLocalFileName(path))
        result = self.runCommand(command)
        return result==0

    def remoteCopy(self, source, target, force=False):
        if self.client:
            status,_ = self.client.copy(source, target, force=force)
            if self.debug:
                print('DEBUG:', status.message, 'copy from', source, 'to', target, " force=", force)
            result = 0 if status.ok else 5
        else:
            command = self.remoteCp.format(source=source, target=target)
            result = self.runCommand(command)
        return result==0

    def cp(self, source, target, force=False):
        if self.isRemotePath(source) or self.isRemotePath(target):
            return self.remoteCopy(source, target, force=force)
        else:
            shutil.copyfile(source, target)
            return True

    def rm(self, path):
        if self.isRemotePath(path):
            self.remoteFileRm(path)
        else:
            os.remove(path)

    def exists(self, path, attempts=1):
        attemptsLeft = attempts
        found = False
        while attemptsLeft > 0:
            attemptsLeft -= 1
            found = self.remoteFileExists(path) if self.isRemotePath(path) else os.path.exists(path)
            if found:
                return True
            if attemptsLeft > 0:
                print('INFO: file was not found:'+path+', trying %d more times...'%attemptsLeft)
                if self.timeBetweenAttempts:
                    print('INFO: wait 30 seconds before trying again')
                    sleep(self.timeBetweenAttempts)

        return found

    def mkdir(self, path):
        status = False
        if not self.isRemotePath(path):
            try:
                os.mkdir(path)
                status = True
                if self.debug:
                    print('DEBUG: created the local directory:', path)
            except:
                pass
        else:
            command = self.remoteMkdir.format(server=self.getRemoteFileserver(), path=self.getLocalFileName(path))
            status = self.runCommand(command)
        return status

    # create folder and parent folders 
    def makedirs(self, path):
        status = True
        # use os functionality for local files
        if not self.isRemotePath(path):
            try:
                os.makedirs(path)
            except:
                status = False
        else:
            if not self.remoteDirectoryExists(path):
                localName = self.getLocalFileName(path.strip('/'))
                directories = localName.split('/')
                i = len(directories)
                while i>self.makedirsMinLevel:
                    parentDirectory = self.getRemoteFileName('/'.join(directories[:i-1]))
                    if self.remoteDirectoryExists(parentDirectory):
                        break
                    i -= 1
                for j in range(i, len(directories) + 1):
                    remoteName = self.getRemoteFileName('/'.join(directories[:j]))
                    self.mkdir(remoteName)
        return status

    def fileExists(self, fileName):
        return os.path.isfile(self.getLocalFileName(fileName))

    def directoryExists(self, fileName):
        return os.path.isdir(self.getLocalFileName(fileName))

    def getRemoteFileName(self, path):
        # only xrootd protocol supported at the moment
        return self.getXrootdFileName(path)

    def removeRedirector(self, fileName):
        if '://' in fileName:
            fileName = '/'.join(fileName.split('://')[1].split('/')[1:])
        return fileName

    def addRedirector(self, redirector=None, fileName=None):
        if redirector is None:
            redirector = self.xrootdRedirectors[0]
        if fileName is None:
            fileName = ""
        return redirector + '/' + fileName

    # ------------------------------------------------------------------------------
    # get file name WITH redirector
    # ------------------------------------------------------------------------------
    def getXrootdFileName(self, rawFileName):
        xrootdFileName = rawFileName.strip()
        if self.isStoragePath(xrootdFileName):
            if self.pnfsStoragePath:
                xrootdFileName = self.pnfsStoragePath + xrootdFileName.strip()
            if self.xrootdRedirectors:
                xrootdFileName = self.addRedirector(fileName=self.removeRedirector(xrootdFileName), redirector=self.xrootdRedirectors[0])
            return xrootdFileName.strip()
        elif self.isPnfs(xrootdFileName):
            # replace already existing redirectors with primary one
            if self.xrootdRedirectors:
                xrootdFileName = self.addRedirector(fileName=self.removeRedirector(xrootdFileName), redirector=self.xrootdRedirectors[0])
            return xrootdFileName.strip()
        else:
            return xrootdFileName.strip()

    # ------------------------------------------------------------------------------
    # get file name WITHOUT redirector, but with complete PNFS path in case of a 
    # /store/ path.
    # ------------------------------------------------------------------------------
    def getLocalFileName(self, rawFileName):
        if rawFileName:
            localFileName = self.removeRedirector(rawFileName.strip())
            if localFileName.startswith('/store/') and self.pnfsStoragePath:
                localFileName = self.pnfsStoragePath + localFileName.strip()
            return localFileName
        else:
            print ("\x1b[31mERROR: invalid file name\x1b[0m")
            raise Exception("InvalidFileName")

    def getXrootdRedirector(self):
        return self.xrootdRedirectors[0] if self.xrootdRedirectors and len(self.xrootdRedirectors)>0 else None


