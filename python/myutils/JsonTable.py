import json

class JsonTable(object):

    def __init__(self, jsonFiles):
        self.debug = False
        self.jsonFiles = jsonFiles
        self.data = {}
        # load raw json dictionary
        for jsonFileName in self.jsonFiles:
            with open(jsonFileName, 'r') as jsonFile:
                self.data.update(json.load(jsonFile))

    # convert dictionary to table for easy access
    def getEtaPtTable(self, tableName, subtableName='eta_pt_ratio' ):
        table = []
        if tableName in self.data:
            for etaKey in self.data[tableName][subtableName].keys():
                etaName = etaKey.split(':')[0]
                if etaName not in ["eta", "abseta"]:
                    raise Exception("JsonFileError")
                etaRange = eval(etaKey.split(':')[1])
                for ptKey in self.data[tableName][subtableName][etaKey].keys():
                    ptName = ptKey.split(':')[0]
                    if ptName != "pt":
                        raise Exception("JsonFileError")
                    ptRange = eval(ptKey.split(':')[1])
                    table.append({'eta_min': etaRange[0], 'eta_max': etaRange[1], 'pt_min': ptRange[0], 'pt_max': ptRange[1], 'eta_type': etaName, 'value': self.data[tableName][subtableName][etaKey][ptKey]['value']})
        return table

    def find(self, table, eta, pt, default=1.0):
        for sfBin in table:
            if ((sfBin['eta_type'] == 'eta' and sfBin['eta_min'] <= eta and eta < sfBin['eta_max']) or (sfBin['eta_type'] == 'abseta' and sfBin['eta_min'] <= abs(eta) and abs(eta) < sfBin['eta_max'])) and sfBin['pt_min'] <= pt and pt < sfBin['pt_max']:
                return sfBin['value']
        if self.debug:
            print "not found in table: eta:", eta, "pt:", pt, "->returning default =", default
        return default
