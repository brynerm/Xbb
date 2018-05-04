#!/usr/bin/env python
import json

#runLsList = 'existing_lumis_pp_DoubleMuon.txt'
#runLsList2 = 'existing_lumis_pp_DoubleEG.txt'
#jsonFile = '2017_json.txt'

runLsList = 'existing_lumis_nano_DoubleMuon_300122to300237.txt'
runLsList2 = 'existing_lumis_nano_DoubleEG_300122to300237.txt'
jsonFile = '2017_json_300122to300237.txt'

def runLsListToDict(existingData):
    existingDataDict = {}
    for i in existingData:
        if i[0] not in existingDataDict:
            existingDataDict[i[0]] = {}
        existingDataDict[i[0]][i[1]] = True
    return existingDataDict

with open(jsonFile,'r') as f:
    jsonData = json.load(f)
with open(runLsList,'r') as f:
    existingData = runLsListToDict(json.load(f))
with open(runLsList2,'r') as f:
    existingData2 = runLsListToDict(json.load(f))

runs = sorted(jsonData.keys())
counts = {
        'both': 0,
        'only1': 0,
        'only2': 0,
        'missing': 0,
        }
for run in runs:
    lsList = jsonData[run]
    for lsRange in lsList:
        printout = ("%d %r:"%(int(run), lsRange)).ljust(24)
        lsCount = 0
        for ls in range(lsRange[0],lsRange[1]+1):
            iRun = int(run)
            inData = iRun in existingData and ls in existingData[iRun]
            inData2 = iRun in existingData2 and ls in existingData2[iRun]
            if inData and inData2:
                printout += '\x1b[42m \x1b[0m'
                counts['both'] += 1
            elif inData: 
                printout += '\x1b[45m \x1b[0m'
                counts['only1'] += 1
            elif inData2:
                printout += '\x1b[44m \x1b[0m'
                counts['only2'] += 1
            else:
                printout += '\x1b[41m \x1b[0m'
                counts['missing'] += 1
            lsCount += 1
            if lsCount > 300:
                printout += "\n" + "".ljust(24)
                lsCount = 0
        print printout
print "%r"%counts



