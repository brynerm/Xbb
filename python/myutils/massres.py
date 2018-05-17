import ROOT
import numpy as np
#import scipy.special as sc
#import pandas as pd
#from scipy.stats import crystalball
from scipy.stats import norm as gauss
from scipy.optimize import curve_fit
from scipy.stats import chisquare
#from scipy.interpolate import UnivariateSpline as spline
from scipy.stats import skew
from matplotlib import pyplot as plt
import argparse
import os 

parser = argparse.ArgumentParser(description='read file')
parser.add_argument('filename', metavar='file1', type=str, nargs='+',
                           help='root file')
#parser.add_argument('--showBins',action='store_const', const=True, default=False, dest='showBins', help='show all bin entries')
parser.add_argument('--showLogs', action='store_const', const=True, default=False, dest='showLogs', help='show full Logs')
#parser.add_argument('--total', action='store_const', const=True, default=False, dest='total', help='print total number of entries in regions')
args=parser.parse_args()
n = 120
def binToArray(tree,name = "tree"):
    array = np.zeros(20)
    if args.showLogs:
        print "read "+name
    for b in range(1,21):
        array[b-1] = tree.GetBinContent(b)
    return array

def histToArray(tree,name = "tree",n=n):
    bins = np.zeros(n)
    entries = np.zeros(n)
    if args.showLogs:
        print "read "+name
    for b in range(1,n+1):
        bins[b-1] = tree.GetBinCenter(b)
        entries[b-1] = tree.GetBinContent(b)
    return (bins,entries)

tree = "ZHbb"
#dfs = {}
def cryst(x,beta,m,loc,scale,norm):
    _norm_pdf_C = np.sqrt(2*np.pi)

    N = norm / (m/beta / (m-1) * np.exp(-beta**2 / 2.0) + _norm_pdf_C * sc.ndtr(beta))
    rhs = lambda x, beta, m: np.exp(-x**2 / 2)
    lhs = lambda x, beta, m: (m/beta)**m * np.exp(-beta**2 / 2.0) * (m/beta - beta - x)**(-m)
    y = (x-loc)/scale
    return N * np.where((y > -beta), rhs(y,beta,m), lhs(y,beta,m))

def normal(x,loc,scale,norm):
    return norm *  gauss.pdf(x,loc,scale)

def biasednormal(x,loc,scale,norm,q,m):
    return normal(x,loc,scale,norm) +  m*(x-loc) + q


for filename in args.filename:

    f = os.getcwd() + "/" + filename
    print "file: " + filename
    #directory = filename[8:-5]+"/"
    directory = ""
    if args.showLogs:
        print "directory: " + directory
    dc = ROOT.TFile.Open(f, 'read')
#    df = pd.DataFrame()
    t = dc.Get(directory + tree)
    if t:
#        t.Print()
        #df["bincenter"], df["entry"] = histToArray(t,tree)
        #df["mass_w"] = df.bincenter*df.entry
        #sum_entries = df.entry.sum()
        #mean = df.mass_w.sum()/sum_entries
        #df["std_w"] = (df.bincenter-mean)**2 * df.entry
        #std = np.sqrt(df.std_w.sum()/sum_entries)
        #print mean
        #print std
        #i_max = 60
        fitfunc = biasednormal
        bins, arr = histToArray(t,tree)
        i_max=np.argmax(arr)
        bounds = np.array([(i_max-n/8),(i_max+n/8)])
        cutted = np.arange(*bounds)
        gauss_parms, gauss_cov = curve_fit(fitfunc,bins[cutted],arr[cutted],[115.,20.,20.,0,-0.02])
        #cryst_parms=[0.,1.]
        #for i in range(1):
        #    cryst_parms, cryst_cov = curve_fit(lambda x,beta,m: cryst(x,beta,m,*gauss_parms),bins,arr,[cryst_parms[0],cryst_parms[1]])
        #    gauss_parms, gauss_cov = curve_fit(lambda x,loc,scale,norm: cryst(x,cryst_parms[0],cryst_parms[1],loc,scale,norm),bins,arr,[gauss_parms[0],gauss_parms[1],gauss_parms[2]])
        #[1.,1.,gauss_parms[0],gauss_parms[1],gauss_parms[2]])
        #diff = spline(bins,arr,k=4).derivative()
        #x_max = diff.roots()
        #if len(x_max) != 3:
        #    print "len x_max ="+str(len(x_max))+"!"
        #x_max = x_max[1]
        #hm = np.max(arr)/2.0
        #interpol = spline(bins,arr-hm,k=3)
        #x_hm = interpol.roots()
        #print x_max
        #print x_hm[1]-x_hm[0]
        #print (x_hm[1]-x_hm[0])/2.355
        #print cryst_parms
        print gauss_parms

        #print np.sqrt(np.diag(cryst_cov))
        error =  np.sqrt(np.diag(gauss_cov))
        print error
        corr_coeff = gauss_cov / error[:,None] / error[None,:]
        #print corr_coeff
        peak = gauss_parms[0]+gauss_parms[4]*gauss_parms[1]**3/gauss_parms[2]*np.sqrt(2*np.pi)
        r = gauss_parms[1]/peak
        r_err = r*np.sqrt(error[1]**2/gauss_parms[1]**2 + error[0]**2/gauss_parms[0]**2)
        #error[1]**2/gauss_parms[1]**2 + error[0]**2/gauss_parms[0]**2 - gauss_cov[0,1]/(gauss_parms[0]*gauss_parms[1]))
        print (r,r_err)
        #plt.figure()
        #plt.plot(bins[cutted],arr[cutted],'x-',label="hist")
        #plt.plot(np.linspace(*bins[bounds]),fitfunc(np.linspace(*bins[bounds]),*gauss_parms),label="fit")
        #plt.plot(peak,fitfunc(peak,*gauss_parms),'ro',label="peak")
        #plt.title(filename)
        #plt.legend()
        #plt.xlabel("H mass")
        #plt.ylabel("expected")
        #plt.ylim([0,6])
        
        #gr_fit = ROOT.TGraph(len(cutted),bins[cutted],y_fitted )

        #gr_fit.Draw("AC*")
        #print chisquare(arr[cutted],fitfunc(bins[cutted],*gauss_parms),len(gauss_parms))
#        print cryst(bins,cryst_parms[0],cryst_parms[1],gauss_parms[0],gauss_parms[1],gauss_parms[2])
        #print arr-cryst(bins,cryst_parms[0],cryst_parms[1],gauss_parms[0],gauss_parms[1],gauss_parms[2])
        #hm = arr[i_max]/2.0
        #peak = bins[i_max]
        #cutted = np.arange(n)
        #print peak
        #print bins[bounds]
        #print arr[bounds]
        pdf = arr[cutted] / np.sum(arr[cutted])
        mean = np.sum(bins[cutted]*pdf)
        std = np.sqrt(np.sum((bins[cutted]-mean)**2 * pdf))
        print mean
        print std/peak
        print std

        #print t.GetSkewness()
        #arr1 = np.sort(arr[5:i_max-5])
        #arr2 = np.sort(arr[-5:i_max+5:-1])[::-1]
        #i_hm1 = np.digitize(hm,arr1)
        #i_hm2 = i_max + 5 + np.digitize(hm,arr2)
        #fwhm = bins[i_hm2]-bins[i_hm1]
        #print t.GetMean()
        #print t.GetStdDev()
        #print t.GetRMSError()
        #print peak
        #print arr
        #print hm
        #print fwhm
        #print fwhm/2.355
        print ""
    else:
        print "missing hist: "+tree
#    dfs[filename] = df
#panel = pd.Panel(dfs)
#print panel
#if args.showLogs:
plt.show()
