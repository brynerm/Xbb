#! /usr/bin/env python
from optparse import OptionParser
import sys
import time
import os
import shutil
import subprocess
import signal
import ROOT
import fnmatch
import hashlib
ROOT.gROOT.SetBatch(True)
from myutils.sampleTree import SampleTree as SampleTree
from myutils.FileList import FileList
from myutils.Datacard import Datacard
from myutils import BetterConfigParser, ParseInfo
from myutils.copytreePSI import filelist
from myutils.FileLocator import FileLocator

try:
    if sys.version_info[0] == 2 and sys.version_info[1] < 7:
        print "\x1b[31mWARNING: unsupported Python version! Python 2.7+ is needed!\x1b[0m"
except:
    print "unable to detect python version!"

parser = OptionParser()
parser.add_option("-b", "--addCollections", dest="addCollections", default=None, help="collections to add in sysnew step")
parser.add_option("-B", "--batch", dest="override_to_run_in_batch", action="store_true", default=False,
                      help="Override run_locally option to run in batch")
parser.add_option("-C", "--checkCached", dest="checkCached", action="store_true", default=False,
                      help="check if all cached trees exist before submitting the jobs")
parser.add_option("-c", "--condor-nobatch", dest="condorNobatch", action="store_true", default=False,
                      help="submit in a single submit file per job instead of using batches")
parser.add_option("-F", "--folderTag", dest="ftag", default="",
                      help="Creats a new folder structure for outputs or uses an existing one with the given name")
parser.add_option("-f", "--force", dest="force", action="store_true", default=False,
                      help="Force overwriting of files if they already exist")
parser.add_option("-i", "--interactive", dest="interactive", action="store_true", default=False, help="Interactive mode")
parser.add_option("-J", "--task", dest="task", default="",
                      help="Task to be done, i.e. 'dc' for Datacards, 'prep' for preparation of Trees, 'plot' to produce plots or 'eval' to write the MVA output or 'sys' to write regression and systematics (or 'syseval' for both). ")
parser.add_option("-k", "--skipExisting", dest="skipExisting", action="store_true", default=False,
                      help="don't submit jobs if output files already exist")
parser.add_option("-L", "--local", dest="override_to_run_locally", action="store_true", default=False,
                      help="Override run_locally option to run locally")
parser.add_option("-l", "--limit", dest="limit", default=None, help="max number of files to process per sample")
parser.add_option("-N", "--number-of-events-or-files", dest="nevents_split_nfiles_single", default=-1,
                      help="Number of events per file when splitting or number of files when using single file workflow.")
parser.add_option("-p", "--parallel", dest="parallel", default=None, help="Fine control for per job task parallelization. Higher values are usually faster and reduce IO and overhead, but also consume more memory. If number of running jobs is not limited, lower values could also increase performance. (Default: maximum per job parallelization).")
parser.add_option("-r", "--regions", dest="regions", default=None, help="regions to plot, can contain * as wildcard")
parser.add_option("-S","--samples",dest="samples",default="", help="samples you want to run on")
parser.add_option("-s","--folders",dest="folders",default="", help="folders to check, e.g. PREPout,SYSin")
parser.add_option("-T", "--tag", dest="tag", default="8TeV",
                      help="Tag to run the analysis with, example '8TeV' uses config8TeV and pathConfig8TeV to run the analysis")
                      help="Tag to run the analysis with, example '8TeV' uses config8TeV and pathConfig8TeV to run the analysis")
parser.add_option("-u","--samplesInfo",dest="samplesInfo", default="", help="path to directory containing the sample .txt files with the sample lists")
parser.add_option("-V", "--verbose", dest="verbose", action="store_true", default=False,
                      help="Activate verbose flag for debug printouts")
parser.add_option("-w", "--wait-for", dest="waitFor", default=None, help="wait for another job to finish")
parser.add_option("-m", "--scan", dest="mvaScan", default=None, help="Scan MVA Settings for runtraining with the given number as number of runs")

(opts, args) = parser.parse_args(sys.argv)

submitScriptRunAllLocally = False
submitScriptSubmitAll = False
debugPrintOUts = opts.verbose

if opts.tag == "":
    print "Please provide tag to run the analysis with, example '-T 8TeV' uses config8TeV and pathConfig8TeV to run the analysis."
    sys.exit(123)

if opts.task == "":
    print "Please provide a task.\n-J prep:\tpreparation of Trees\n-J sys:\t\twrite regression and systematics\n-J eval:\tcreate MVA output\n-J plot:\tproduce Plots\n-J dc:\t\twrite workspaces and datacards"
    sys.exit(123)


globalFilesSubmitted = 0
globalFilesSkipped = 0

def signal_handler(signal, frame):
    if (globalFilesSubmitted > 0 or globalFilesSkipped > 0):
        print('\n----------------------------\n')
        print('Files submitted:'+str(globalFilesSubmitted))
        print('Files skipped:'+str(globalFilesSkipped))
    print('You pressed Ctrl+C!')
    print('\n----------------------------\n')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# split list into sublists of given length
def partitionFileList(sampleFileList, chunkSize=1):
    return [sampleFileList[i:i+chunkSize] for i in range(0, len(sampleFileList), chunkSize)]

en = opts.tag

#create the list with the samples to run over
samplesList = [x.strip() for x in opts.samples.strip().split(',') if len(x.strip()) > 0]
timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")

if debugPrintOUts:
    print 'samplesList', samplesList
    print 'timestamp', timestamp

# ------------------------------------------------------------------------------
# the LIST OF CONFIG FILES is taken from the path config
# ------------------------------------------------------------------------------
pathconfig = BetterConfigParser()
pathconfig.read('%sconfig/paths.ini'%(opts.tag))
try:
    _configs = pathconfig.get('Configuration', 'List').split(" ")
    configs = ['%sconfig/'%(opts.tag) + c for c in _configs]
except:
    print("\x1b[31mERROR: configuration file not found. Check config-tag specified with -T and presence of '[Configuration] List' in .ini files.\x1b[0m")
    raise Exception("ConfigNotFound")

if debugPrintOUts:
    print 'configs', configs
    print 'opts.ftag', opts.ftag

# ------------------------------------------------------------------------------
# COPY CONFIG files to log output folder
# TODO: clean up
# ------------------------------------------------------------------------------
if not opts.ftag == '':
    tagDir = pathconfig.get('Directories', 'tagDir')
    if debugPrintOUts:
        print 'tagDir', tagDir
    DirStruct={
        'tagDir': tagDir,
        'ftagdir': '%s/%s/'%(tagDir, opts.ftag),
        'logpath': '%s/%s/%s/'%(tagDir, opts.ftag, 'Logs'),
        'plotpath': '%s/%s/%s/'%(tagDir, opts.ftag, 'Plots'),
        'limitpath': '%s/%s/%s/'%(tagDir, opts.ftag, 'Limits'),
        'confpath': '%s/%s/%s/'%(tagDir, opts.ftag, 'config')
    }

    if debugPrintOUts:
        print 'DirStruct', DirStruct

    for keys in ['tagDir', 'ftagdir', 'logpath', 'plotpath', 'limitpath', 'confpath']:
        try:
            os.stat(DirStruct[keys])
        except:
            os.mkdir(DirStruct[keys])

    # new behavior: write special .ini file, which is not under version control and is always recreated again
    with open("{tag}config/volatile.ini".format(tag=opts.tag), "w") as outputFile:
        outputFile.write('# this file has been created automatically and will be overwritten by submit.py!\n')
        outputFile.write('[Directories]\n')
        outputFile.write('plotpath: %s\n'%DirStruct['plotpath'])
        outputFile.write('logpath: %s\n'%DirStruct['logpath'])
        outputFile.write('limits: %s\n'%DirStruct['limitpath'])

    # old behavior: overwrite paths.ini (will only be done if one of the paths to modify has been found!)
    pathfile = open('%sconfig/paths.ini'%opts.tag)
    buffer = pathfile.readlines()
    pathfile.close()
    volatilePathsFound = False
    for line in buffer:
        if line.startswith('plotpath') or line.startswith('logpath') or line.startswith('limits'):
            volatilePathsFound = True
    if volatilePathsFound:
        os.rename('%sconfig/paths.ini'%opts.tag,'%sconfig/paths.ini.bkp'%opts.tag)
        pathfile = open('%sconfig/paths.ini'%opts.tag,'w')
        for line in buffer:
            if line.startswith('plotpath'):
                line = 'plotpath: %s\n'%DirStruct['plotpath']
            elif line.startswith('logpath'):
                line = 'logpath: %s\n'%DirStruct['logpath']
            elif line.startswith('limits'):
                line = 'limits: %s\n'%DirStruct['limitpath']
            pathfile.write(line)
        pathfile.close()

    # copy config files
    for item in configs:
        # if relative path to other config folder, just take file name
        destConfigFileName = item.split('/')[-1]
        shutil.copyfile(item, '%s/%s/%s'%(tagDir, opts.ftag, destConfigFileName))

if debugPrintOUts:
    print configs

config = BetterConfigParser()
config.read(configs)

# ------------------------------------------------------------------------------
# RETRIEVE RELEVANT VARIABLES FROM CONFIG FILES AND FROM COMMAND LINE OPTIONS
# TODO: clean up
# ------------------------------------------------------------------------------
counter = 0
logPath = config.get('Directories', 'logpath')
samplesinfo = config.get('Directories', 'samplesinfo')
whereToLaunch = config.get('Configuration', 'whereToLaunch')
run_locally = str(config.get('Configuration', 'run_locally'))

if opts.override_to_run_locally and opts.override_to_run_in_batch:
    print 'both override_to_run_locally and override_to_run_in_batch ativated, using str(config.get("Configuration","run_locally")) instead'
elif opts.override_to_run_locally:
    run_locally = 'True'
    submitScriptRunAllLocally = True
    print 'using override_to_run_locally to override str(config.get("Configuration","run_locally"))'
elif opts.override_to_run_in_batch:
    run_locally = 'False'
    print 'using override_to_run_in_batch to override str(config.get("Configuration","run_locally"))'

print 'whereToLaunch', whereToLaunch
print 'run_locally', run_locally

# ------------------------------------------------------------------------------
# CREATE DIRECTORIES
# ------------------------------------------------------------------------------
fileLocator = FileLocator(config=config)
if 'PSI' in whereToLaunch:
    for dirName in ['PREPout', 'SYSout', 'MVAout', 'tmpSamples']:
        if not fileLocator.exists(config.get('Directories', dirName)):
            fileLocator.makedirs(config.get('Directories', dirName))
if 'condor' in whereToLaunch:
    if 'X509_USER_PROXY' not in os.environ:
        print '\x1b[41m\x1b[97mX509 proxy certificate not set, run:\x1b[0m'
        print '--------------'
        print 'voms-proxy-init -voms cms -rfc -out ${HOME}/.x509up_${UID} -valid 192:00'
        print 'export X509_USER_PROXY=${HOME}/.x509up_${UID}'
        print '--------------'


def dump_config(configs, output_file):
    """
    Dump all the configs in a output file
    Args:
        output_file: the file where the log will be dumped
        configs: list of files (string) to be dumped
    Returns:
        nothing
    """
    outf = open(output_file, 'w')
    for i in configs:
        try:
            f = open(i, 'r')
            outf.write(f.read())
        except: print '@WARNING: Config' + i + ' not found. It will not be used.'

# check if the logPath exist. If not exit
if not os.path.isdir(logPath):
    print '@ERROR : ' + logPath + ': dir not found.'
    print '@ERROR : Create it before submitting '
    print 'Exit'
    sys.exit(-1)

# ------------------------------------------------------------------------------
# CREATE DICTIONARY TO BE USED AT JOB SUBMISSION TIME
#  ------------------------------------------------------------------------------
repDict = {
    'en': en,
    'logpath': logPath,
    'job': '',
    'task': opts.task,
    'queue': opts.queue,
    'timestamp': timestamp,
    'additional': '',
    'job_id': 'noid',
    'nprocesses': str(max(int(pathconfig.get('Configuration', 'nprocesses')), 1))
}

list_submitted_singlejobs = {}

# ------------------------------------------------------------------------------
# SUBMIT SCRIPT options defined in config general.ini [SubmitOptions]

# Template: 
#[SubmitOptions]
#
#submitScriptTemplate = qsub {options} -o {logfile} {runscript}
#submitScriptOptionsTemplate = -V -cwd -q %%(queue)s -N %%(name)s -j y -pe smp %%(nprocesses)s
#submitScriptSpecialOptions = {
#        'mergesyscachingdcsplit': ' -l h_vmem=6g ',
#        'singleeval': ' -l h_vmem=6g ',
#        'runtraining': ' -l h_vmem=6g ',
#        'eval': ' -l h_vmem=4g ',
#        'cachedc': ' -l h_vmem=6g ',
#        'cacheplot': ' -l h_vmem=6g ',
#        'cachetraining': ' -l h_vmem=6g ',
#        'hadd': ' -l h_vmem=6g ',
#        }
#
#
# ------------------------------------------------------------------------------

# Default values
submitScriptTemplate = 'qsub {options} -o {logfile} {runscript}'
submitScriptOptionsTemplate = '-V -cwd -q %(queue)s -N %(name)s -j y -pe smp %(nprocesses)s'
submitScriptSpecialOptions = {
<<<<<<< HEAD
        'mergesyscachingdcsplit': ' -l h_vmem=6g ',
        'sysnew': ' -l h_vmem=6g ',
        'singleeval': ' -l h_vmem=6g ',
        'runtraining': ' -l h_vmem=6g ',
        'eval': ' -l h_vmem=4g ',
        'cachedc': ' -l h_vmem=6g ',
        'runtraining': ' -l h_vmem=6g ',
        'cacheplot': ' -l h_vmem=6g ',
        'cachetraining': ' -l h_vmem=6g ',
        'hadd': ' -l h_vmem=6g ',
        }
condorBatchGroups = {}
=======
    'mergesyscachingdcsplit': ' -l h_vmem=6g ',
    'singleeval': ' -l h_vmem=6g ',
    'runtraining': ' -l h_vmem=6g ',
    'eval': ' -l h_vmem=4g ',
    'cachedc': ' -l h_vmem=6g ',
    'cacheplot': ' -l h_vmem=6g ',
    'cachetraining': ' -l h_vmem=6g ',
    'hadd': ' -l h_vmem=6g ',
}

# Overwrite by config
if pathconfig.has_section('SubmitOptions'):
    if pathconfig.has_option('SubmitOptions', 'submitScriptTemplate'):
        submitScriptTemplate = pathconfig.get('SubmitOptions', 'submitScriptTemplate')
>>>>>>> subm_conf

    if pathconfig.has_option('SubmitOptions', 'submitScriptOptionsTemplate'):
        submitScriptOptionsTemplate = pathconfig.get('SubmitOptions', 'submitScriptOptionsTemplate')

    if pathconfig.has_option('SubmitOptions', 'submitScriptSpecialOptions'):
        submitScriptSpecialOptions.update(eval(pathconfig.get('SubmitOptions', 'submitScriptSpecialOptions')))

condorBatchGroups = {}
# ------------------------------------------------------------------------------
# get job queue
# ------------------------------------------------------------------------------
def getJobQueue():
    if 'condor' in whereToLaunch:
        getJobQueueCommand = ["condor_q", "-nobatch"]
    else:
        getJobQueueCommand = ["qstat -xml | tr '\\n' ' ' | sed 's#<job_list[^>]*>#\\n#g' | sed 's#<[^>]*>##g' | grep ' ' | column -t"]
    p = subprocess.Popen(getJobQueueCommand, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    return out.split("\n")

# ------------------------------------------------------------------------------
# wait for other jobs to finish first
# ------------------------------------------------------------------------------
def waitFor(jobNameList):
    matches = -1
    while matches != 0:
        matches = 0
        jobNames = [x.strip() for x in jobNameList.split(',') if len(x.strip()) > 0]
        if len(jobNames) > 0:
            jobQueue = getJobQueue()
            for jobName in jobNames:
                matches += len([jobName for job in jobQueue if jobName in job])
                print "\x1b[44m\x1b[97mwaiting for \x1b[92m", matches, "\x1b[97m jobs of type \x1b[93m", jobName, "\x1b[97m to finish....\x1b[0m"
                if matches > 0:
                    break
        if matches > 0:
            time.sleep(30)

# ------------------------------------------------------------------------------
# filter sample list with simple wildcard (*) syntax, used for -S option 
# ------------------------------------------------------------------------------
def filterSampleList(sampleIdentifiers, samplesList):
    if samplesList and len([x for x in samplesList if x]) > 0:
        filteredList = []
        for expr in samplesList:
            if expr in sampleIdentifiers:
                filteredList.append(expr)
            elif '*' in expr:
                for sampleIdentifier in sampleIdentifiers:
                    if fnmatch.fnmatch(sampleIdentifier, expr):
                        filteredList.append(sampleIdentifier)
        filteredList = list(set(filteredList))
        return filteredList
    else:
        return sampleIdentifiers

# ------------------------------------------------------------------------------
# STANDARD WORKFLOW SUBMISSION FUNCTION
# TODO: separate classes for SGE and HTCondor
# ------------------------------------------------------------------------------
def submit(job, repDict):
    global counter
    global submitScriptSubmitAll
    global submitScriptRunAllLocally
    repDict['job'] = job
    counter += 1
    repDict['name'] = '%(job)s_%(en)s%(task)s' %repDict

    # -----------------------------------------------------------------------------
    # prepare RUN SCRIPT
    # (this is independent of batch system or local)
    # -----------------------------------------------------------------------------
    runScript = 'runAll.sh %(job)s %(en)s '%(repDict)
    runScript += opts.task + ' ' + repDict['nprocesses'] + ' ' + repDict['job_id'] + ' ' + repDict['additional']

    # add named arguments to run script
    if 'arguments' in repDict:
        for argument, value in repDict['arguments'].iteritems():
            runScript += (' --{argument}={value} '.format(argument=argument, value=value)) if len('{value}'.format(value=value)) > 0 else ' --{argument} '.format(argument=argument)

    # log=batch system log, out=stdout of process
    logOutputPath = '%(logpath)s/%(task)s_%(timestamp)s_%(job)s_%(en)s_%(additional)s.log' %(repDict)
    errorOutputPath = '%(logpath)s/%(task)s_%(timestamp)s_%(job)s_%(en)s_%(additional)s.err' %(repDict)
    outOutputPath = '%(logpath)s/%(task)s_%(timestamp)s_%(job)s_%(en)s_%(additional)s.out' %(repDict)

    # -----------------------------------------------------------------------------
    # CONDOR
    # -----------------------------------------------------------------------------
    if 'condor' in whereToLaunch:
        with open('batch/condor/template.sub', 'r') as template:
            lines = template.readlines()

        firstFileOfBatch = False
        isBatched = 'batch' in repDict and not opts.condorNobatch
        if isBatched:
            if repDict['batch'] not in condorBatchGroups:
                # first file of batch -> make new submit file
                firstFileOfBatch = True
                condorBatchGroups[repDict['batch']] = '%(task)s_%(timestamp)s_%(batch)s'%(repDict)
            # use existing submit file and append
            dictHash = condorBatchGroups[repDict['batch']]
        else:
            # create a new submit file
            dictHash = '%(task)s_%(timestamp)s'%(repDict) + '_%x'%hash('%r'%repDict)

        condorDict = {
            'runscript': runScript.split(' ')[0],
            'arguments': ' '.join(runScript.split(' ')[1:]),
            'output': outOutputPath,
            'log': logOutputPath,
            'error': errorOutputPath,
            'queue': 'workday',
        }
        submitFileName = 'condor_{hash}.sub'.format(hash=dictHash)

        # append to existing bath
        if isBatched:
            with open(submitFileName, 'a') as submitFile:
                submitFile.write("\n")
                for line in lines:
                    submitFile.write(line.format(**condorDict))
            command = None
        else:
            with open(submitFileName, 'w') as submitFile:
                for line in lines:
                    submitFile.write(line.format(**condorDict))
            command = 'condor_submit {submitFileName}'.format(submitFileName=submitFileName)
        print "COMMAND:\x1b[35m", runScript, "\x1b[0m"

    # -----------------------------------------------------------------------------
    # SGE
    # -----------------------------------------------------------------------------
    else:
        qsubOptions = submitScriptOptionsTemplate%(repDict)

        if opts.task in submitScriptSpecialOptions:
            qsubOptions += submitScriptSpecialOptions[opts.task]

        command = submitScriptTemplate.format(options=qsubOptions, logfile=outOutputPath, runscript=runScript)
        dumpConfigFileName = "%(logpath)s/%(timestamp)s_%(task)s.config" %(repDict)
        if not os.path.isfile(dumpConfigFileName):
            dump_config(configs, dumpConfigFileName)

    # -----------------------------------------------------------------------------
    # RUN command
    # -----------------------------------------------------------------------------
    if command:
        if opts.interactive and not submitScriptSubmitAll:
            print "SUBMIT:\x1b[34m", command, "\x1b[0m\n(press ENTER to run it and continue, \x1b[34ml\x1b[0m to run it locally, \x1b[34ma\x1b[0m to run all jobs locally and \x1b[34ms\x1b[0m to submit the remaining jobs)"
            if submitScriptRunAllLocally:
                answer = 'l'
            else:
                answer = raw_input().strip()
            if answer.lower() in ['no', 'n']:
                return
            elif answer.lower() == 's':
                submitScriptSubmitAll = True
            elif answer.lower() in ['l', 'local', 'a','d']:
                if answer.lower() == 'a':
                    submitScriptRunAllLocally = True
                print "run locally"
<<<<<<< HEAD
                command = 'sh {runscript} | tee {logfile}'.format(runscript=runScript,logfile=outOutputPath)
=======
                command = 'sh {runscript}'.format(runscript=runScript)
                if answer.lower() == 'd':
                    command = "XBBDEBUG=1 " + command
>>>>>>> subm_conf
        else:
            print "the command is ", command
        subprocess.call([command], shell=True)

def printSamplesStatus(samples, regions, status):
    print("regions:")
    for i,region in enumerate(regions):
        print ('  %02d: %s'%(i, region))

    print("-"*80)
    header = ' '*40 + ' '.join(['%02d'%x for x in range(len(regions))])
    print(header)
    nFound = 0
    nNotFound = 0
    sampleNames = sorted(list(set([sample.name for sample in samples])))
    for sampleName in sampleNames: 
        line = sampleName.rjust(39) + ' '
        for region in regions:
            if region in status and sampleName in status[region]:
                if status[region][sampleName]:
                    line += '\x1b[42m\x1b[97m[+]\x1b[0m'
                    nFound += 1
                else:
                    line += '\x1b[41m\x1b[97m[X]\x1b[0m'
                    nNotFound += 1
            else:
                line += '\x1b[45m[?]\x1b[0m'
        print line
    print('summary:\n----------\nfound: %d\nnot found:%d'%(nFound, nNotFound))
    return nFound, nNotFound

# -----------------------------------------------------------------------------
# check prerequisities
# -----------------------------------------------------------------------------
if opts.waitFor:
    waitFor(opts.waitFor)

# -----------------------------------------------------------------------------
# DATASETS: create list of .root files for each dataset
# -----------------------------------------------------------------------------
if opts.task == 'datasets':
    dasQuery = config.get("Configuration", "dasQuery")
    datasetsFileName = config.get("Configuration", "datasets")
    samplefiles = config.get('Directories', 'samplefiles')
    try:
        os.makedirs(samplefiles)
    except:
        pass
    with open(datasetsFileName, 'r') as datasetsFile:
        datasets = datasetsFile.readlines()
    for dataset in datasets:
        print "DATASET:", dataset
        getDatasetFilesComand = [dasQuery.format(dataset=dataset.strip())]
        p = subprocess.Popen(getDatasetFilesComand, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        print "command:", getDatasetFilesComand
        print "root files:\n", out
        datasetName = dataset.strip().strip('/').split('/')[0]
        with open(samplefiles + '/' + datasetName + '.txt', 'w') as datasetFile:
            datasetFile.write(out)

# -----------------------------------------------------------------------------
# summary: To check prep, sysnew and eval step
# At the end of the prep, sysnew and eval, a table witht the number of files missing BEFORE the submission
# -----------------------------------------------------------------------------

def PrintProcessedFiles(path, subfolder, sampleFileList, prepOUT = None):

    n_missing_files = 0
    n_total_files = len(sampleFileList)

    for sampleFile in  sampleFileList:
        #if step == sys:
        filename=fileLocator.getFilenameAfterPrep(sampleFile)
        FileExist = fileLocator.isValidRootFile("{path}/{subfolder}/{filename}".format(path=pathOUT, subfolder=subfolder, filename=filename))
        #print 'yeah'
        #FileExist = fileLocator.remoteFileExists("{path}/{subfolder}/{filename}".format(path=pathOUT, subfolder=subfolder, filename=filename))

        if not FileExist:
            #If prepOUT is not None: checking the sysnew step or the eval step. Need to check if the broken/missing file in the SYSout/MVAout was in the PREPout to start with. If not, this file will NOT be reported ass missing.
            if prepOUT:
                FileExist = fileLocator.remoteFileExists("{path}/{subfolder}/{filename}".format(path=prepOUT, subfolder=subfolder, filename=filename))
                if FileExist:
                    n_missing_files += 1
                else:
                    n_total_files -= 1
            else:
                n_missing_files += 1

    #if n_missing_files == 0:
    #    print "\x1b[32m All files for",  subfolder, "were already produced. Nothing to submit \x1b[0m.."
    #else:
    #    print "\x1b[36m", n_missing_files,"/", len(sampleFileList), "\x1b[0m missing files for sample \x1b[36m", subfolder, " \x1b[0m.."

    return [n_missing_files, n_total_files]

# -----------------------------------------------------------------------------
# PREP: copy skimmed ntuples to local storage
# -----------------------------------------------------------------------------
if opts.task == 'prep' or opts.task == 'checkprep':

    pathOUT = config.get("Directories", "PREPout")
    samplefiles = config.get('Directories', 'samplefiles')
    info = ParseInfo(samplesinfo, pathOUT)
    sampleIdentifiers = filterSampleList(info.getSampleIdentifiers(), samplesList)

    chunkSize = 10 if int(opts.nevents_split_nfiles_single) < 1 else int(opts.nevents_split_nfiles_single)

    # for checksysnew step: dic contains missing number of files for each sample
    missingFiles = {}

    # process all sample identifiers (correspond to folders with ROOT files)
    for sampleIdentifier in sampleIdentifiers:
        sampleFileList = filelist(samplefiles, sampleIdentifier)
        if opts.limit and len(sampleFileList) > int(opts.limit):
            sampleFileList = sampleFileList[0:int(opts.limit)]
        splitFilesChunks = [sampleFileList[i:i+chunkSize] for i in range(0, len(sampleFileList), chunkSize)]

        # for checksysnew step: only list of missing files are printed. No jobs are submited
        if opts.task == 'checkprep':
            print "going to check \x1b[36m", len(sampleFileList), "\x1b[0m files for sample \x1b[36m", sampleIdentifier, " \x1b[0m.."
            missingFiles[sampleIdentifier] =  PrintProcessedFiles(pathOUT, sampleIdentifier, sampleFileList)
            continue

        print "going to submit \x1b[36m", len(splitFilesChunks), "\x1b[0m jobs for sample \x1b[36m", sampleIdentifier, " \x1b[0m.."
        # submit a job for a chunk of N files
        for chunkNumber, splitFilesChunk in enumerate(splitFilesChunks):

            if opts.skipExisting:
                skipChunk = all([fileLocator.isValidRootFile("{path}/{subfolder}/{filename}".format(path=pathOUT, subfolder=sampleIdentifier, filename=fileLocator.getFilenameAfterPrep(fileName))) for fileName in splitFilesChunk])
            else:
                skipChunk = False

            if not skipChunk or opts.force:
                jobDict = repDict.copy()
                jobDict.update({
                    'arguments': {
                        'sampleIdentifier': sampleIdentifier,
                        'fileList': FileList.compress(splitFilesChunk),
                    },
                    'batch': 'prep_' + sampleIdentifier,
                    })
                if opts.force:
                    jobDict['arguments']['force'] = ''
                jobName = 'prep_{sample}_part{part}'.format(sample=sampleIdentifier, part=chunkNumber)
                submit(jobName, jobDict)
            else:
                print "SKIP: chunk #%d, all files exist and are valid root files!"%chunkNumber

    # printing the content of missingFiles
    if opts.task == 'checkprep':
        print '\n================'
        print 'SUMMARY: checkprep'
        print '==================\n'
        for sampleIdentifier in sampleIdentifiers:
            n_missing_files = missingFiles[sampleIdentifier][0]
            n_total_files = missingFiles[sampleIdentifier][1]
            if n_missing_files == 0:
                print "\x1b[32m All files for \x1b[36m", sampleIdentifier, "\x1b[32m were already produced. Nothing to submit \x1b[0m.."
            else:
                print "\x1b[31m WARNING:", n_missing_files,"/", n_total_files, "missing or broken root files for sample \x1b[36m", sampleIdentifier, " \x1b[0m.."

# -----------------------------------------------------------------------------
# HADD: this can merge files partially to avoid too many small trees
# -----------------------------------------------------------------------------
if opts.task == 'hadd':
    from hadd import PartialFileMerger

    inputPath = config.get("Directories", "HADDin")
    outputPath = config.get("Directories", "HADDout")

    samplefiles = config.get('Directories', 'samplefiles')
    info = ParseInfo(samplesinfo, inputPath)
    sampleIdentifiers = filterSampleList(info.getSampleIdentifiers(), samplesList)

    samplefilesMerged = samplefiles + '/merged/'
    fileLocator.makedirs(samplefilesMerged)

    # process all sample identifiers (correspond to folders with ROOT files)
    for sampleIdentifier in sampleIdentifiers:

        chunkSize = 10
        if config.has_section('Hadd') and config.has_option('Hadd', sampleIdentifier):
            chunkSize = int(config.get('Hadd', sampleIdentifier).strip())
            print "INFO: chunkSize read from config => ", chunkSize

        sampleFileList = filelist(samplefiles, sampleIdentifier)
        if opts.limit and len(sampleFileList) > int(opts.limit):
            sampleFileList = sampleFileList[0:int(opts.limit)]
        splitFilesChunks = [sampleFileList[i:i+chunkSize] for i in range(0, len(sampleFileList), chunkSize)]

        mergedFileNames = []
        for i, splitFilesChunk in enumerate(splitFilesChunks):

            # only give good files to hadd
            fileNames = []
            for fileName in splitFilesChunk:
                fileNameAfterPrep = "{path}/{sample}/{fileName}".format(path=config.get('Directories','HADDin'), sample=sampleIdentifier, fileName=fileLocator.getFilenameAfterPrep(fileName))
                #if fileLocator.isValidRootFile(fileNameAfterPrep):
                if fileLocator.exists(fileNameAfterPrep):
                    fileNames.append(fileName)
                    print ".",
                else:
                    print "x",
            print "INFO: #files=", len(splitFilesChunk), ", good=", len(fileNames)

            if len(fileNames) > 0:
                # 'fake' filenames to write into text file for merged files
                partialFileMerger = PartialFileMerger(fileNames, i, config=config, sampleIdentifier=sampleIdentifier)
                mergedFileName = partialFileMerger.getMergedFakeFileName()
                mergedFileNames.append(mergedFileName)

                outputFileName = partialFileMerger.getOutputFileName()

                if (opts.force or not fileLocator.isValidRootFile(outputFileName)):
                    jobDict = repDict.copy()
                    jobDict.update({
                        'queue': 'short.q',
                        'arguments':{
                            'sampleIdentifier': sampleIdentifier,
                            'fileList': FileList.compress(fileNames),
                            'chunkNumber': i,
                        },
                        'batch': opts.task + '_' + sampleIdentifier,
                        })
                    if opts.force:
                        jobDict['arguments']['force'] = ''
                    jobName = 'hadd_{sample}_part{part}'.format(sample=sampleIdentifier, part=i)
                    submit(jobName, jobDict)
            else:
                print "\x1b[31mERROR: no good files for this sample available:",sampleIdentifier,"!\x1b[0m"

        # write text file for merged files
        mergedFileListFileName = '{path}/{sample}.{ext}'.format(path=samplefilesMerged, sample=sampleIdentifier, ext='txt')
        with open(mergedFileListFileName, 'w') as mergedFileListFile:
            mergedFileListFile.write('\n'.join(mergedFileNames))

        print "INFO: hadd {sample}: {a} => {b}".format(sample=sampleIdentifier, a=len(sampleFileList), b=len(splitFilesChunks))
        print "INFO:  > {mergedFileListFileName}".format(mergedFileListFileName=mergedFileListFileName)


# -----------------------------------------------------------------------------
# SYSNEW: add additional branches and branches for sys variations
# -----------------------------------------------------------------------------
if opts.task == 'sysnew' or opts.task == 'checksysnew':

    # need prepout to get list of file processed during the prep. Files missing in both the prepout and the sysout will not be considered as missing during the sys step
    prepOUT = config.get("Directories", "PREPout")
    path = config.get("Directories", "SYSin")
    pathOUT = config.get("Directories", "SYSout")
    samplefiles = config.get('Directories','samplefiles')
    info = ParseInfo(samplesinfo, path)
    sampleIdentifiers = filterSampleList(info.getSampleIdentifiers(), samplesList)

    chunkSize = 10 if int(opts.nevents_split_nfiles_single) < 1 else int(opts.nevents_split_nfiles_single)

    # check for empty list of collections to add
    addCollections = opts.addCollections
    if not addCollections or len(addCollections.strip())<1:
        print "\x1b[31mWARNING: No collections specified, using the default \x1b[32m'Sys.all'\x1b[31m instead, to force adding nothing, use \x1b[32m--addCollections None\x1b[31m!\x1b[0m"
        addCollections = 'Sys.all'

    # for checksysnew step: dic contains missing number of files for each sample
    missingFiles = {}

    # process all sample identifiers (correspond to folders with ROOT files)
    for sampleIdentifier in sampleIdentifiers:
        sampleFileList = filelist(samplefiles, sampleIdentifier)
        if opts.limit and len(sampleFileList) > int(opts.limit):
            sampleFileList = sampleFileList[0:int(opts.limit)]
        splitFilesChunks = [sampleFileList[i:i+chunkSize] for i in range(0, len(sampleFileList), chunkSize)]

        # for checksysnew step: only list of missing files are printed. No jobs are submited
        if opts.task == 'checksysnew':
            print "going to check \x1b[36m", len(sampleFileList), "\x1b[0m files for sample \x1b[36m", sampleIdentifier, " \x1b[0m.."
            missingFiles[sampleIdentifier] =  PrintProcessedFiles(pathOUT, sampleIdentifier, sampleFileList, prepOUT)
            continue

        print "going to submit \x1b[36m", len(splitFilesChunks), "\x1b[0m jobs for sample \x1b[36m", sampleIdentifier, " \x1b[0m.."

        # for sysnew
        # submit a job for a chunk of N files
        for chunkNumber, splitFilesChunk in enumerate(splitFilesChunks):

            if opts.skipExisting:
                # skip, if all output files exist
                skipChunk = all([fileLocator.isValidRootFile("{path}/{subfolder}/{filename}".format(path=pathOUT, subfolder=sampleIdentifier, filename=fileLocator.getFilenameAfterPrep(fileName))) for fileName in splitFilesChunk])
                # skip, if all input files do not exist/are broken
                #allInputFilesMissing = False
                allInputFilesMissing = all([not fileLocator.isValidRootFile("{path}/{subfolder}/{filename}".format(path=path, subfolder=sampleIdentifier, filename=fileLocator.getFilenameAfterPrep(fileName))) for fileName in splitFilesChunk])
            else:
                skipChunk = False
                allInputFilesMissing = False

            if (not skipChunk and not allInputFilesMissing) or opts.force:
                jobDict = repDict.copy()
                jobDict.update({
                    'arguments':{
                        'sampleIdentifier': sampleIdentifier,
                        'fileList': FileList.compress(splitFilesChunk),
                        'addCollections': addCollections,
                    },
                    'batch': opts.task + '_' + sampleIdentifier,
                    })
                if opts.force:
                    jobDict['arguments']['force'] = ''
                jobName = 'sysnew_{sample}_part{part}'.format(sample=sampleIdentifier, part=chunkNumber)
                submit(jobName, jobDict)
            else:
                if allInputFilesMissing:
                    print "\x1b[31mSKIP: chunk %d, all input files of this chunk are missing!\x1b[0m"%chunkNumber
                else:
                    print "SKIP: chunk #%d, all files exist and are valid root files!"%chunkNumber

    if opts.task == 'checksysnew':
        # printing the content of missingFiles
        print '\n=================='
        print 'SUMMARY: checksysnew'
        print '====================\n'
        for sampleIdentifier in sampleIdentifiers:
            #print 'sampleIdentifier is', sampleIdentifier
            n_missing_files = missingFiles[sampleIdentifier][0]
            n_total_files = missingFiles[sampleIdentifier][1]
            if n_missing_files == 0:
                print "\x1b[32m All files for \x1b[36m", sampleIdentifier, "\x1b[32m were already produced wrt the current prepOUT path \x1b[0m.."
            else:
                print "\x1b[31m WARNING:", n_missing_files,"/", n_total_files, "missing or broken root files wrt the current prepOUT path for sample \x1b[36m", sampleIdentifier, " \x1b[0m.."

# -----------------------------------------------------------------------------
# CACHETRAINING: prepare skimmed trees including the training/eval cuts 
# -----------------------------------------------------------------------------
if opts.task.startswith('cachetraining'):
    trainingRegions = [x.strip() for x in (config.get('MVALists','List_for_submitscript')).split(',')]
    allBackgrounds = list(set(sum([eval(config.get(trainingRegion, 'backgrounds')) for trainingRegion in trainingRegions], [])))
    allSignals = list(set(sum([eval(config.get(trainingRegion, 'signals')) for trainingRegion in trainingRegions], [])))
    print "backgrounds:"
    for sampleName in sorted(allBackgrounds):
        print " >", sampleName
    print "signals:"
    for sampleName in sorted(allSignals):
        print " >", sampleName
    
    # get samples info
    info = ParseInfo(samplesinfo, config.get('Directories', 'MVAin'))
    samples = info.get_samples(allBackgrounds + allSignals)

    # find all sample identifiers that have to be cached, if given list is empty, run it on all
    sampleIdentifiers = filterSampleList(list(set([sample.identifier for sample in samples])), samplesList)
    print "sample identifiers: (", len(sampleIdentifiers), ")"
    for sampleIdentifier in sampleIdentifiers:
        print " >", sampleIdentifier
    
    # per job parallelization parameter can split regions into several job
    if opts.parallel:
        regionChunkSize = int(opts.parallel)
        regionChunks = [trainingRegions[i:i + regionChunkSize] for i in xrange(0, len(trainingRegions), regionChunkSize)]
    else:
        # default is all at once
        regionChunks = [trainingRegions]
    
    # submit separate jobs for all samples
    for sampleIdentifier in sampleIdentifiers:

        # number of files to process per job 
        splitFilesChunkSize = min([sample.mergeCachingSize for sample in samples if sample.identifier==sampleIdentifier])
        splitFilesChunks = SampleTree({'name': sampleIdentifier, 'folder': config.get('Directories', 'MVAin')}, countOnly=True, splitFilesChunkSize=splitFilesChunkSize, config=config).getSampleFileNameChunks()
        print "DEBUG: split after ", splitFilesChunkSize, " files => number of parts = ", len(splitFilesChunks)
        
        # submit all the single chunks for one sample
        for chunkNumber, splitFilesChunk in enumerate(splitFilesChunks, start=1):

            for regionChunkNumber, regionChunk in enumerate(regionChunks):
                jobDict = repDict.copy()
                jobDict.update({
                    'queue': 'short.q',
                    'arguments':
                        {
                            'trainingRegions': ','.join(regionChunk),
                            'sampleIdentifier': sampleIdentifier,
                            'chunkNumber': chunkNumber,
                            'splitFilesChunks': len(splitFilesChunks),
                            'splitFilesChunkSize': splitFilesChunkSize,
                        },
                    'batch': opts.task + '_' + sampleIdentifier,
                    })
                if opts.force:
                    jobDict['arguments']['force'] = ''
                # pass file list, if only a chunk of it is processed
                if len(splitFilesChunks) > 1:
                    compressedFileList = FileList.compress(splitFilesChunk)
                    jobDict['arguments']['fileList'] = compressedFileList
                jobName = 'training_cache_{sample}_part{part}_{regionChunkNumber}'.format(sample=sampleIdentifier, part=chunkNumber, regionChunkNumber=regionChunkNumber)
                submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# RUNTRAINING: train mva, outputs .xml file. Needs cachetraining before.
# -----------------------------------------------------------------------------
if opts.task.startswith('runtraining'):
    # training regions
    trainingRegions = [x.strip() for x in (config.get('MVALists', 'List_for_submitscript')).split(',')]

    # separate job for all training regions
    for trainingRegion in trainingRegions:
        jobDict = repDict.copy()
        jobDict.update({'arguments': {'trainingRegions': trainingRegion}, 'queue': 'short.q'})
        if opts.mvaScan is not None:
            jobDict['arguments']['scan'] = opts.mvaScan
        jobName = 'training_run_{trainingRegions}'.format(trainingRegions=trainingRegion)
        submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# CACHEPLOT: prepare skimmed trees with cuts for the CR/SR
# -----------------------------------------------------------------------------
if opts.task.startswith('cacheplot'):
    regions = [x.strip() for x in (config.get('Plot_general', 'List')).split(',')]
    sampleNames = eval(config.get('Plot_general', 'samples'))
    dataSampleNames = eval(config.get('Plot_general', 'Data'))

    # get samples info
    info = ParseInfo(samplesinfo, config.get('Directories', 'plottingSamples'))
    samples = info.get_samples(sampleNames + dataSampleNames)

    # find all sample identifiers that have to be cached, if given list is empty, run it on all
    sampleIdentifiers = filterSampleList(sorted(list(set([sample.identifier for sample in samples]))), samplesList)
    print "sample identifiers: (", len(sampleIdentifiers), ")"
    for sampleIdentifier in sampleIdentifiers:
        print " >", sampleIdentifier
    
    # submit jobs, 1 to n separate jobs per sample
    for sampleIdentifier in sampleIdentifiers:

        # per job parallelization parameter can split regions into several job
        if opts.parallel:
            regionChunkSize = int(opts.parallel)
            regionChunks = [regions[i:i + regionChunkSize] for i in xrange(0, len(regions), regionChunkSize)]
        else:
            # default is all at once
            regionChunks = [regions]

        # number of files to process per job 
        splitFilesChunkSize = min([sample.mergeCachingSize for sample in samples if sample.identifier == sampleIdentifier])
        splitFilesChunks = SampleTree({
                'name': sampleIdentifier, 
                'folder': config.get('Directories', 'plottingSamples')
            }, countOnly=True, splitFilesChunkSize=splitFilesChunkSize, config=config).getSampleFileNameChunks()
        print "DEBUG: split after ", splitFilesChunkSize, " files => number of parts = ", len(splitFilesChunks)
            
        # submit all the single parts
        for chunkNumber, splitFilesChunk in enumerate(splitFilesChunks, start=1):
            compressedFileList = FileList.compress(splitFilesChunk)

            # submit a separate job for all region chunks
            for regionChunkNumber, regionChunk in enumerate(regionChunks): 
                jobDict = repDict.copy()
                jobDict.update({
                        'queue': 'short.q',
                        'arguments':
                            {
                            'regions': ','.join(regionChunk),
                            'sampleIdentifier': sampleIdentifier,
                            'chunkNumber': chunkNumber,
                            'splitFilesChunks': len(splitFilesChunks),
                            'splitFilesChunkSize': splitFilesChunkSize,
                            },
                        'batch': opts.task + '_' + sampleIdentifier,
                        })
                if opts.force:
                    jobDict['arguments']['force'] = ''
                # pass file list, if only a chunk of it is processed
                if len(splitFilesChunks) > 1:
                    jobDict['arguments']['fileList'] = compressedFileList
                jobName = 'plot_cache_{sample}_part{chunk}_{regionChunkNumber}'.format(sample=sampleIdentifier, chunk=chunkNumber, regionChunkNumber=regionChunkNumber)
                submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# RUNPLOT: make CR/SR plots. Needs cacheplot before. 
# -----------------------------------------------------------------------------
if opts.task.startswith('runplot'):
    regions = [x.strip() for x in (config.get('Plot_general', 'List')).split(',')]
    plotVars = [x.strip() for x in (config.get('Plot_general', 'var')).split(',')]

    if opts.parallel:
        plotVarChunks = [plotVars[i:i + int(opts.parallel)] for i in xrange(0, len(plotVars), int(opts.parallel))]
    else:
        plotVarChunks = [plotVars]

    # submit all the plot regions as separate jobs
    for region in regions:

        # if --regions is given, only plot those regions 
        regionMatched = any([fnmatch.fnmatch(region, enabledRegion) for enabledRegion in opts.regions.split(',')]) if opts.regions else True
        if regionMatched:
            for j, plotVarList in enumerate(plotVarChunks):
                jobDict = repDict.copy()
                jobDict.update({
                    'arguments':
                        {
                            'regions': region,
                            'vars': ','.join(plotVarList),
                        }
                    })
                jobName = 'plot_run_{region}_{chunk}'.format(region=region, chunk=j)
                submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# CACHEDC: prepare skimmed trees for DC, which have looser cuts to include 
# variations of systematics. 
# -----------------------------------------------------------------------------
if opts.task.startswith('cachedc'):
    # get list of all sample names used in DC step
    sampleNames = []
    regions = [x.strip() for x in config.get('LimitGeneral', 'List').split(',') if len(x.strip()) > 0]
    if config.has_option('LimitGeneral', 'addSample_sys'):
        addSample_sys = eval(config.get('LimitGeneral', 'addSample_sys'))
        sampleNames += [addSample_sys[key] for key in addSample_sys]
    for region in regions:
        for sampleType in ['data', 'signal', 'background']:
            sampleNames += eval(config.get('dc:%s'%region, sampleType))

    # get samples info
    sampleFolder = config.get('Directories', 'dcSamples')
    info = ParseInfo(samplesinfo, sampleFolder)
    samples = info.get_samples(sampleNames)

    # find all sample identifiers that have to be cached, if given list is empty, run it on all
    sampleIdentifiers = filterSampleList(info.getSampleIdentifiers(), samplesList)
    print "sample identifiers: (", len(sampleIdentifiers), ")"
    for sampleIdentifier in sampleIdentifiers:
        print " >", sampleIdentifier
        
    # check existence of cached files before job submission, otherwise it will be checked at the beginning of the job
    if opts.skipExisting:
        status = {}
        for i, region in enumerate(regions):
            dcMaker = Datacard(config=config, region=region, verbose=False)
            status[region] = dcMaker.getCacheStatus(useSampleIdentifiers=sampleIdentifiers)
            print "INFO: done checking files for region\x1b[34m",region, "\x1b[0m(",i, "of", len(regions),")"
        printSamplesStatus(samples=samples, regions=regions, status=status)
    
    # per job parallelization parameter can split regions into several job
    if opts.parallel:
        regionChunkSize = int(opts.parallel)
        regionChunks = [regions[i:i + regionChunkSize] for i in xrange(0, len(regions), regionChunkSize)]
    else:
        # default is all at once
        regionChunks = [regions]

    # submit jobs, 1 to n separate jobs per sample
    for sampleIdentifier in sampleIdentifiers:

        # if file existence has already been checked, check if it exists for all the regions 
        if opts.skipExisting:
            sampleNames = sorted(list(set([sample.name for sample in samples if sample.identifier == sampleIdentifier])))
            filesMissing = False
            for sampleName in sampleNames:
                for region in regions:
                    if region in status and sampleName in status[region]:
                        if not status[region][sampleName]:
                            filesMissing = True
                            break
            if not filesMissing:
                print 'INFO: SKIP samples:', sampleNames,'files already exist!'
                continue
            else:
                print 'INFO: files do not exist yet!'

        # number of files to process per job 
        sampleSizesList = [sample.mergeCachingSize for sample in samples if sample.identifier == sampleIdentifier]
        splitFilesChunkSize = min(sampleSizesList) if len(sampleSizesList) > 0 else 3
        splitFilesChunks = SampleTree({
                'name': sampleIdentifier, 
                'folder': sampleFolder
            }, countOnly=True, splitFilesChunkSize=splitFilesChunkSize, config=config).getSampleFileNameChunks()
        print "DEBUG: split after ", splitFilesChunkSize, " files => number of parts = ", len(splitFilesChunks)

        # submit all the single parts
        for chunkNumber, splitFilesChunk in enumerate(splitFilesChunks, start=1):
            compressedFileList = FileList.compress(splitFilesChunk)

            # separate job for all region chunks (default: 1 chunk consisting of all regions)
            for regionChunkNumber, regionChunk in enumerate(regionChunks):
                jobDict = repDict.copy()
                jobDict.update({
                    'arguments':
                        {
                            'regions': ','.join(regionChunk),
                            'sampleIdentifier': sampleIdentifier,
                            'chunkNumber': chunkNumber,
                            'splitFilesChunks': len(splitFilesChunks),
                            'splitFilesChunkSize': splitFilesChunkSize,
                        },
                    'batch': opts.task + '_' + sampleIdentifier,
                    'queue': 'short.q',
                    })
                if opts.force:
                    jobDict['arguments']['force'] = ''
                # pass file list, if only a chunk of it is processed
                if len(splitFilesChunks) > 1:
                    jobDict['arguments']['fileList'] = compressedFileList
                jobName = 'dc_cache_{sample}_part{chunk}_{regionChunkNumber}'.format(sample=sampleIdentifier, chunk=chunkNumber, regionChunkNumber=regionChunkNumber)
                submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# RUNDC: produce DC .txt and .root files. Needs cachedc before.
# -----------------------------------------------------------------------------
if opts.task.startswith('rundc'):
    
    regions = Datacard.getRegions(config=config)
    samples = Datacard.getSamples(config=config, regions=regions)
    sampleIdentifiers = sorted(filterSampleList(list(set([sample.identifier for sample in samples])), samplesList))

    # check existence of cached files
    if opts.checkCached:
        status = {}
        for region in regions:
            dcMaker = Datacard(config=config, region=region, verbose=False)
            status[region] = dcMaker.getCacheStatus(useSampleIdentifiers=sampleIdentifiers)
        nFound, nNotFound = printSamplesStatus(samples=samples, regions=regions, status=status)
        if nNotFound>0:
            print('run cachedc again!')
            raise Exception("NotCached")

    # submit all the DC regions as separate jobs
    for region in regions:
        regionMatched = any([fnmatch.fnmatch(region, enabledRegion) for enabledRegion in opts.regions.split(',')]) if opts.regions else True
        if regionMatched:
            # submit separate jobs for either sampleIdentifiers
            for sampleIdentifier in sampleIdentifiers:
                jobDict = repDict.copy()
                jobDict.update({
                    'queue': 'short.q',
                    'arguments':
                        {
                            'regions': region,
                            'sampleIdentifier': sampleIdentifier,
                        },
                    'batch': opts.task + '_' + sampleIdentifier,
                    })
                if opts.force:
                    jobDict['arguments']['force'] = ''
                jobName = 'dc_run_' + '_'.join([v for k,v in jobDict['arguments'].iteritems()])
                submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# MERGEDC: merge DC .root files for all samples per region and produce combined
# .root and .txt files. Needs rundc before.
# -----------------------------------------------------------------------------
if opts.task.startswith('mergedc'):
    regions = Datacard.getRegions(config=config)

    # submit all the DC regions as separate jobs
    for region in regions:
        jobDict = repDict.copy()
        jobDict.update({
            'queue': 'short.q',
            'arguments':
                {
                    'regions': region,
                }
            })
        jobName = 'dc_merge_' + '_'.join([v for k,v in jobDict['arguments'].iteritems()])
        submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# regression training
# -----------------------------------------------------------------------------
if opts.task == 'trainReg':
    repDict['queue'] = 'all.q'
    submit('trainReg',repDict)

# -----------------------------------------------------------------------------
# OLD sys step (here to keep compatibility, use 'sysnew' if possible!!)
# -----------------------------------------------------------------------------
# ADD SYSTEMATIC UNCERTAINTIES AND ADDITIONAL HIGHER LEVEL VARIABLES TO THE TREES
if opts.task == 'sys' or opts.task == 'syseval':
    path = config.get("Directories", "SYSin")
    samplefiles = config.get('Directories','samplefiles')
    info = ParseInfo(samplesinfo, path)
    sampleIdentifiers = filterSampleList(info.getSampleIdentifiers(), samplesList)
    chunkSize = 10 if int(opts.nevents_split_nfiles_single) < 1 else int(opts.nevents_split_nfiles_single)

    # process all sample identifiers (correspond to folders with ROOT files)
    for sampleIdentifier in sampleIdentifiers:
        sampleFileList = filelist(samplefiles, sampleIdentifier)
        if opts.limit and len(sampleFileList) > int(opts.limit):
            sampleFileList = sampleFileList[0:int(opts.limit)]
        splitFilesChunks = [sampleFileList[i:i+chunkSize] for i in range(0, len(sampleFileList), chunkSize)]
        
        print "going to submit \x1b[36m",len(splitFilesChunks),"\x1b[0m jobs for sample \x1b[36m", sampleIdentifier, " \x1b[0m.." 
        # submit a job for a chunk of N files
        for chunkNumber, splitFilesChunk in enumerate(splitFilesChunks):
            jobDict = repDict.copy()
            jobDict.update({'arguments':{
                    'sampleIdentifier': sampleIdentifier,
                    'fileList': FileList.compress(splitFilesChunk),
                }})
            jobName = 'sys_{sample}_part{part}'.format(sample=sampleIdentifier, part=chunkNumber)
            submit(jobName, jobDict)

# -----------------------------------------------------------------------------
# EVALUATION OF EVENT BY EVENT BDT SCORE
# -----------------------------------------------------------------------------
if opts.task == 'eval' or opts.task.startswith('eval_'):
    #repDict['queue'] = 'long.q'
    path = config.get("Directories", "MVAin")
    pathOUT = config.get("Directories", "MVAout")
    info = ParseInfo(samplesinfo, path)
    samplefiles = config.get('Directories', 'samplefiles')
    sampleIdentifiers = filterSampleList(info.getSampleIdentifiers(), samplesList)
    chunkSize = 10 if int(opts.nevents_split_nfiles_single) < 1 else int(opts.nevents_split_nfiles_single)

    # process all sample identifiers (correspond to folders with ROOT files)
    for sampleIdentifier in sampleIdentifiers:
        splitFilesChunks = partitionFileList(filelist(samplefiles, sampleIdentifier), chunkSize=chunkSize) 

        # submit a job for each chunk of up to N files
        print "going to submit \x1b[36m",len(splitFilesChunks),"\x1b[0m jobs for sample \x1b[36m", sampleIdentifier, " \x1b[0m.."
        for chunkNumber, splitFilesChunk in enumerate(splitFilesChunks):
            # check existence of OUTPUT files
            if opts.skipExisting:
                skipChunk = all([fileLocator.isValidRootFile("{path}/{subfolder}/{filename}".format(path=pathOUT, subfolder=sampleIdentifier, filename=fileLocator.getFilenameAfterPrep(fileName))) for fileName in splitFilesChunk])
            else:
                skipChunk = False

            if not skipChunk or opts.force:
                jobDict = repDict.copy()
                jobDict.update({
                    'arguments':{
                        'sampleIdentifier': sampleIdentifier,
                        'fileList': FileList.compress(splitFilesChunk),
                    },
                    'batch': opts.task + '_' + sampleIdentifier,
                })
                jobName = 'eval_{sample}_part{part}'.format(sample=sampleIdentifier, part=chunkNumber)
                submit(jobName, jobDict)
            else:
                print "SKIP: chunk #%d, all files exist and are valid root files!"%chunkNumber

# -----------------------------------------------------------------------------
# summary: print list of cuts for CR+SR
# TODO: this should also run some basic checks on the configuration
# -----------------------------------------------------------------------------
if opts.task == 'summary':
    print "-"*80
    print " paths"
    print "-"*80
    for path in ['PREPout', 'SYSin', 'SYSout', 'MVAin', 'MVAout', 'tmpSamples', 'samplesinfo']:
        try:
            print "%s:"%path, "\x1b[34m", config.get('Directories', path) ,"\x1b[0m"
        except:
            print "\x1b[31mERROR: did not find path in config section [Directories]:",path,"\x1b[0m"

    print "-"*80
    print " pre-selection ('prep')"
    print "-"*80

    cutDict = {}
    samplesinfo=config.get('Directories','samplesinfo')
    info = ParseInfo(samplesinfo, '')
    # don't look at subsamples, because they have the same pre-selection cut
    samples = [x for x in info if not x.subsample]
    for sample in samples:
        addTreeCut = sample.addtreecut.replace(' ','')
        if addTreeCut not in cutDict:
            cutDict[addTreeCut] = []
        cutDict[addTreeCut].append(sample.identifier)
    for preselectionCut, listOfSamples in cutDict.iteritems():
        print "SAMPLES: \x1b[34m", ','.join(listOfSamples), "\x1b[0m"
        print "CUT: \x1b[32m", preselectionCut,"\x1b[0m"
        print "-"*40

    print "-"*80
    print " plot samples"
    print "-"*80
    plotSamples = eval(config.get('Plot_general', 'samples'))
    samplesUsed = [x for x in info if x.name in plotSamples]
    sampleIdentifiersUsed = sorted(list(set([x.identifier for x in samplesUsed])))
    for sampleIdentifier in sampleIdentifiersUsed:
        print sampleIdentifier
        for sample in samplesUsed:
            if sample.identifier == sampleIdentifier:
                print " >>> ", sample.name
    


# outputs a simple python code to read the whole sample as chain
if opts.task == 'sample':
    fileLocator = FileLocator(config=config)
    path = config.get("Directories", "PREPout")
    samplefiles = config.get('Directories','samplefiles')
    info = ParseInfo(samplesinfo, path)
    sampleIdentifiers = filterSampleList(info.getSampleIdentifiers(), samplesList)
    foldersToCheck = ["SYSout"] if len(opts.folders.strip()) < 1 else opts.folders.split(',')
    print "folders:", foldersToCheck
    basePaths = {x: config.get("Directories", x) for x in foldersToCheck}

    for folder in foldersToCheck:
        path = basePaths[folder]
        for sampleIdentifier in sampleIdentifiers:
            splitFilesChunks = SampleTree({'name': sampleIdentifier, 'folder': path}, countOnly=True, splitFilesChunkSize=-1, config=config).getSampleFileNameChunks()

            print "----",sampleIdentifier,"----"
            print "# BEGINNING OF PYTHON FILE"
            print "import ROOT"
            print "from myutils.sampleTree import SampleTree"
            print "sampleFiles = ["
            for sampleFileName in splitFilesChunks[0]:
                print "    '{fileName}',".format(fileName=sampleFileName)
            print "]"
            print "sampleTree = SampleTree(sampleFiles, treeName='Events')" 
            print "print 'number of events:', sampleTree.GetEntries()"
            print "# example how to loop over all events"
            print "# alternatively, the TChain object can be accessed as sampleTree.tree"
            print "#for event in sampleTree:"
            print "#if event.Vtype == 1:"
            print "#    print 'Vtype 1 event', event.event"
            print "# END OF PYTHON FILE"
            print "-"*20


# submit all jobs, which have been grouped in a batch
if 'condor' in whereToLaunch:
    for batchName, submitFileIdentifier in condorBatchGroups.iteritems():
        submitFileName = 'condor_{identifier}.sub'.format(identifier=submitFileIdentifier)
        command = 'condor_submit {submitFileName}  -batch-name {batchName}'.format(submitFileName=submitFileName, batchName=batchName)
        if opts.interactive:
            print "SUBMIT:\x1b[34m", command, "\x1b[0m\n(press ENTER to run it and continue)"
            answer = raw_input().strip()
            if answer.lower() in ['no', 'n', 'skip']:
                continue
        else:
            print "the command is ", command
        subprocess.call([command], shell=True)

