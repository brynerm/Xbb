import os
import sys
import shutil
import subprocess
import argparse

#Read input arguments



parser = argparse.ArgumentParser(description='Move plots to public webpage')
parser.add_argument('--input', dest='input', default = "./", metavar='I', help='input folder, default: current folder.')
#parser.add_argument('-r', dest='region', action='store_const',
#                   const=False, default=True,  help='use automatic region list insead of default')
parser.add_argument('--server', default = None, help='other server than lxplus')
parser.add_argument('--user', default = None, help='insert lxplus username, if lxplus user does not match tier3 username. Only is used when --server is not set.')
parser.add_argument('--eos', dest='webservice', action='store_const', default = False, const='eos', help='use default path with eos webserver')
parser.add_argument('--afs', dest='webservice', action='store_const', default = False, const='afs/cern.ch', help='use default path with afs webserver')
parser.add_argument('--name', default = None, help='output folder. default: log folder name')

parser.add_argument('--skipcomp', action='store_const', const=True, default=False,  help='skip comp_ plots')

parser.add_argument('--fext', default = 'png,pdf,root', help='only copy specified file extensions. Default: "png,pdf,root". Use "all" for no restriction')
parser.add_argument('--folders', default = "region", help='how to make subfolders: <none>, <region> (default), <variable>')
parser.add_argument('--tar', default = False, action='store_const', const=True,  help='use tar compression')
parser.add_argument('--no', dest='do_outp', action='store_const',
                   const=False, default=True,  help='no copying')
parser.add_argument('--ni', dest='do_inp', action='store_const',
                   const=False, default=True,  help='no preparation of folders for each Region')
parser.add_argument('--nh', dest='ht', action='store_const',
                   const=False, default=True,  help='no htaccess file is created')

#Move all plots in corresponding subfolders

def MakeSubFolders(plotfolder, current):

    #I am in Plots
    
    # name of new folder
    
    if not os.path.isdir(plotfolder):
        os.makedirs(plotfolder)

    #copy config
    print 'command is','cp -r ../config ' + plotfolder + '/'
    subprocess.call('cp -r ../config ' + plotfolder + '/', shell = True)
    
    #copy .htaccess
    if args.ht:
        if not os.path.isfile(current+'/.htaccess'):
            print current+'/.htaccess'
            raise Exception("HTaccessFileNotFound")
        print 'command is','cp -r '+current+'/.htaccess ' + plotfolder + '/'
        subprocess.call('cp -r '+current+'/.htaccess ' + plotfolder + '/', shell = True)
        subprocess.call('cp -r ' + current + '/.htaccess ' + plotfolder + '/config/', shell = True)

    #copy index.php
    if not os.path.isfile(current+'/index.php'):
        print current+'/index.php'
        raise Exception("IndexFileNotFound")
    subprocess.call('cp -r '+current+'/index.php ' + plotfolder + '/', shell = True)
    subprocess.call('cp -r ' + current + '/index.php ' + plotfolder + '/config/', shell = True)

    #Files, file extension
    FILE = os.listdir('.')
    fext = args.fext.split(',')

    print "File extensions to copy: " + ', '.join(fext)
    
    for file in FILE:
        #not existing files
        if not os.path.isfile(file):
            continue

        #only copy specified file extensions
        filename, ext = os.path.splitext(file)   
        if  not 'all' in fext and ext.split(".")[1] not in fext:
            continue
        
        #skip comp_* files
        if args.skipcomp and filename.startswith("comp_"):
            continue
            
        #prepare subfolder
        folder = None
        if args.folders == 'region':
            folder = filename.split("__")[0]
            folder = folder.replace("comp_","")
        elif args.folders == 'variable':
            folder = filename.split("__")[1]
   
        if folder is None:
            folder2 = plotfolder
        else:
            folder2 = os.path.join(plotfolder,folder)

        print file + ' --> ' + folder2
        if folder2:

            # make subfolder
            if not os.path.isdir(folder2):
                os.mkdir(folder2)
                #subprocess.call('cp -r ../config '+ folder2 + '/', shell = True)
                if args.ht:
                    if not os.path.isfile(current+'/.htaccess'):
                        raise Exception("HTaccessFileNotFound")
                    subprocess.call('cp -r '+current+'/.htaccess ' + folder2 + '/', shell = True)
                subprocess.call('cp -r '+current+'/index.php ' + folder2 + '/', shell = True)

            #copy file
            if os.path.isfile(file):
                shutil.copy(file,folder2)

            # ??? probably outdated
            if os.path.isfile('pdf/'+file.replace('png','pdf')):
                shutil.copy('pdf/'+file.replace('png','pdf'), folder2)
            else:
                pass
                #print 'pdf/'+file.replace('png','pdf'), 'doesn\'t exist'
            if os.path.isfile('root/'+file.replace('png','C')) :
                shutil.copy('root/'+file.replace('png','C'), folder2)
                #print 'root is here'
            else:
                pass
                #print 'root/'+file.replace('png','C'), 'doesn\'t exist'

def MoveSubFolders(plotfolder, output, server=None, user=None):
    if server is None:
        if user is None: 
            # try to use Tier3 username for LxPlus server
            process = os.popen('echo $USER')
            user = process.read().strip()
            process.close()
        server = user + '@lxplus.cern.ch'
    else:
        user = server.split('@')[0]
    
    
    if args.webservice:
        output = '/' + args.webservice+ '/user/' + user[0] + "/" + user + "/www/" + output
    
    if args.tar:
        tarfile = plotfolder + '.tar'
        zipcommand = 'tar -cf ' + tarfile + ' ' + plotfolder
        print zipcommand
        subprocess.call(zipcommand, shell = True)
        copyCommand = 'ssh '+ server +' "cd '+ output + ' && tar -xvv" < ' + tarfile
        print copyCommand
        subprocess.call(copyCommand, shell = True)
        remTarCmd = 'rm ' + tarfile
        print remTarCmd
        subprocess.call(remTarCmd, shell = True)
    
    else:
        print 'gonna lunch the command'
        copyCommand = 'scp -r ' + plotfolder + ' ' + server + ':' + output
        print copyCommand
        subprocess.call(copyCommand, shell = True)


args = parser.parse_args()

_input = os.path.abspath(args.input)

#prepare plotfolder name and output path
_plotfolder = ""

if args.name:
    _output = '/'.join(args.name.split('/')[0:-1])
    _plotfolder = args.name.split('/')[-1]

else:
    _output = "/"

if _plotfolder == "":
    _plotfolder = _input.split('/')[-2]


print 'Input folder is', _input
print 'Output folder is', _output
print 'Plotfolder name is',_plotfolder

#Makro location
_current = '/'.join(os.path.realpath(__file__).split('/')[:-1])

#change to plot location
os.chdir(_input)

if args.do_inp:
    #prepare input folder
    MakeSubFolders(_plotfolder, _current)

if args.do_outp:
    #copy to output path
    MoveSubFolders(_plotfolder, _output, server=args.server, user=args.user )

