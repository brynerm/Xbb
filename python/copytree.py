import ROOT 
from ROOT import TFile
from printcolor import printc

        
def copytree(pathIN,pathOUT,prefix,newprefix,file,Aprefix,Acut):
    input = TFile.Open("%s/%s%s.root" %(pathIN,prefix,file),'read')
    output = TFile.Open("%s/%s%s%s.root" %(pathOUT,newprefix,Aprefix,file),'recreate')

    input.cd()
    obj = ROOT.TObject
    for key in ROOT.gDirectory.GetListOfKeys():
        input.cd()
        obj = key.ReadObj()
        #print obj.GetName()
        if obj.GetName() == 'tree':
            continue
        output.cd()
        #print key.GetName()
        obj.Write(key.GetName())

    inputTree = input.Get("tree")
    nEntries = inputTree.GetEntries()
    output.cd()
    print '\n\t copy file: %s with cut: %s' %(file,Acut)
    outputTree = inputTree.CopyTree(Acut)
    kEntries = outputTree.GetEntries()
    printc('blue','',"\t before cuts\t %s" %nEntries)
    printc('green','',"\t survived\t %s" %kEntries)
    #print "\t Factor for Scaling is %s" %factor
    outputTree.AutoSave()
    #Count.Scale(factor)
    #CountWithPU.Scale(factor)
    #CountWithPU2011B.Scale(factor)
    input.Close()
    output.Close()
