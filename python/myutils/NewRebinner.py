import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.misc import derivative
import os

#TODO verbosity, descriptions and comments, chi2 test, default logscale edges

class Rebinner:
    def __init__(self,mva_min,mva_max,mva_var=None,df_signal=None,df_background=None,make_plots=True,plot_dir="binner_plots"):

        self.mva_min = mva_min
        self.mva_max = mva_max
        self.mva = mva_var
        self.s_hist = None
        self.b_hist = None
        self.splines = []
        self.makePlots = make_plots
        self.plot_dir = plot_dir

        if df_signal is not None and df_background is not None:
            self.prepare(df_signal,df_background)
    
    def prepare(self,df_signal,df_background,fitRoc=True):

        print("INFO: branches in given dataframe: %s"%str(df_signal.columns.values))
        
        if self.mva is None:
            print("INFO: No mva variable set, try to read from dataframe")
            self.mva = df_signal.columns.values[0]
            if not "Nominal" in self.mva:
                raise "wrong mva selected"
            print("INFO: mva variable set to %s"%self.mva)

        print("INFO: preparing signal and background histograms")
        self.total = {}
        self.s_hist = self.sumWeights(df_signal,"signal")
        self.b_hist = self.sumWeights(df_background,"background")
   
        self.s_limit = self.getVal(self.getVal(1./self.total["background"],self.b_hist.eff,self.b_hist.x),self.s_hist.x,self.s_hist.eff)
        print ">= 1 bkg event per bin --> signal efficiency limit: " + str(self.s_limit)
        
        self.bs_ratio = self.total["background"]/self.total["signal"]

        if fitRoc and not self.fitRoc():
            raise "ROC fit failed"



    def sumWeights(self,df,datatype):

        df = df[df.weight > 0]
        df = df.groupby(self.mva).sum()
        df = df.sort_index().reset_index()
        df.weight = df.weight[::-1].cumsum()[::-1]
        self.total[datatype] = df.weight[0]
        print "total " + datatype + ": " + str(self.total[datatype])

        df.weight /= df.weight[0]
        df.append({self.mva: self.mva_max, "weight": (df.weight)[(df.weight).size-1]/2},ignore_index=True)
        return pd.DataFrame({"eff":df.weight, "x":df[self.mva]})

    def load_hdf(self,filename):
        df_signal = pd.read_hdf(filename,"SIG")
        df_background = pd.read_hdf(filename,"BKG")
        self.prepare(df_signal,df_background)
        return self

    def getVal(self, x, inp_col, outp_col):
        idx = np.digitize(x,inp_col,right=False)
        low = 0 if np.all(outp_col >= 0) else self.mva_min
        high = self.mva_max if np.all(outp_col <= self.mva_max) else 1
        if outp_col[0] < outp_col[1]:
            outp_col2 = np.concatenate((np.array([low]),outp_col,np.array([high])))
        else:
            outp_col2 = np.concatenate((np.array([high]),outp_col,np.array([low])))
        return outp_col2[idx]

    def e_b_fit(self,x,a, b, c,d,e,f):
        a_b = lambda x,y,z: y*(1-2*z)**2/(1+2*y)+x
        c_b = lambda x,y: 1+y*x
        return  (np.power((1-x),d)*f*x**e)+np.exp((1-x)*a)*(((x-c_b(b,c))*(a_b(0,b,c_b(b,c))+b)+np.sqrt(2*a_b(0,b,c_b(b,c))*b*(2*(x-c_b(b,c))**2+(b-a_b(0,b,c_b(b,c))))))/(b-a_b(0,b,c_b(b,c)))+1-c_b(b,c))

    def e_b(self,x):
        return np.clip(self.e_b_fit(x,*self.roc_parms),0,1)

    def fitRoc(self):
        if self.makePlots:
            try:
                plt.figure(0)
                plt.clf()
                plt.title("Signal efficiency")
                plt.plot(self.s_hist.eff,self.s_hist.x)
                plt.xlabel("signal efficiency")
                plt.ylabel(self.mva)
                plt.legend()
                if not os.path.exists(self.plot_dir):
                    os.makedirs(self.plot_dir)
                plt.savefig("%s/%s"%(self.plot_dir,"seff_vs_mva.pdf"))

            except:
                print("WARNING: couldn't connect to display")
                self.makePlots = False

        print("Control values: should be mva_min, mva_max, 1, 0")
        print self.getVal(1,self.s_hist.eff,self.s_hist.x)
        print self.getVal(0,self.s_hist.eff,self.s_hist.x)
        print self.getVal(self.mva_min,self.s_hist.x,self.s_hist.eff)
        print self.getVal(self.mva_max,self.s_hist.x,self.s_hist.eff)
        e_b_raw = self.getVal(self.s_hist.x,self.b_hist.x,self.b_hist.eff)
        
        fitlim =  max(0.01,self.s_limit/2)
        fitrange = self.s_hist.eff > fitlim
        #guess paramters
        q=[0.73264402,0.19657415,0.95333252,0.25953929,3.08998159,0.09257951]
        
        converged = False
        print("INFO: Fitting ROC curve")
        try:
            e_b_p, cov = optimize.curve_fit(self.e_b_fit,self.s_hist.eff[fitrange],e_b_raw[fitrange],q, sigma = (self.s_hist.eff)[fitrange])
            converged = True
            print("Fit parameters: %s"%str(e_b_p))
            self.roc_parms = e_b_p
            err = np.sqrt(np.diag(cov))
            print("Standard errors: %s"%str(err))
            print("Correlation Matrix")
            print cov/err[:,np.newaxis]/err[np.newaxis,:]
            if self.makePlots:
                plt.figure(1)
                plt.clf()
                plt.loglog(self.s_hist.eff[fitrange],abs(e_b_raw[fitrange]-self.e_b(self.s_hist.eff[fitrange])),label='fit')
                plt.figure(2)
                plt.clf()
                plt.loglog(self.s_hist.eff[fitrange],abs(e_b_raw[fitrange]-self.e_b(self.s_hist.eff[fitrange]))/e_b_raw[fitrange],label='fit')
                plt.figure(3)
                plt.clf()
                x = np.logspace(np.log(fitlim),0)
                plt.plot(x,self.e_b(x),label='fit')
                plt.figure(4)
                plt.clf()
                x = np.linspace(0,1)
                plt.plot(x,self.e_b(x),label='fit')

        except:
            print("ERROR: ROC curve fit did not converge. Try to set other starting values")
            pass
        
        if self.makePlots:
            plt.figure(1)
            plt.title("absolute difference")
            plt.loglog(self.s_hist.eff[fitrange],abs(e_b_raw[fitrange]-self.e_b_fit(self.s_hist.eff[fitrange],*q)),':',label='guess parameters')
            plt.ylim([1e-6,0.1])
            plt.xlabel("signal efficiency")
            plt.ylabel("delta background efficiency")
            plt.legend()
            plt.savefig("%s/%s"%(self.plot_dir,"roc_fit_diff.pdf"))
            
            plt.figure(2)
            plt.title("relative difference")
            plt.loglog(self.s_hist.eff[fitrange],abs(e_b_raw[fitrange]-self.e_b_fit(self.s_hist.eff[fitrange],*q))/e_b_raw[fitrange],':',label='guess parameters')
            plt.ylim([1e-4,10])
            plt.xlabel("signal efficiency")
            plt.ylabel("delta background efficiency")
            plt.legend()
            plt.savefig("%s/%s"%(self.plot_dir,"roc_fit_diffrel.pdf"))

            plt.figure(3)
            plt.title("ROC")
            plt.loglog(self.s_hist.eff[fitrange],e_b_raw[fitrange],label='samples')
            plt.loglog(self.s_hist.eff[fitrange],self.e_b_fit(self.s_hist.eff[fitrange],*q),':',label='guess parameters')
            plt.ylim([1e-5,1])
            plt.xlabel("signal efficiency")
            plt.ylabel("background efficiency")
            plt.legend()
            plt.savefig("%s/%s"%(self.plot_dir,"roc_fit_log.pdf"))


            plt.figure(4)
            plt.title("ROC")
            plt.plot(self.s_hist.eff[fitrange],e_b_raw[fitrange],label='samples')
            plt.plot(self.s_hist.eff[fitrange],self.e_b_fit(self.s_hist.eff[fitrange],*q),':',label='guess parameters')
            plt.xlabel("signal efficiency")
            plt.ylabel("background efficiency")
            plt.legend()
            plt.savefig("%s/%s"%(self.plot_dir,"roc_fit.pdf"))

            #plt.show()
        return converged

    def getBinCut(self, x, spline):
        return np.where(np.power(10,spline(np.log10(x))) > self.s_limit,np.power(10,spline(np.log10(x))),self.s_limit*np.ones_like(x))

    def sign1(self,s,b):
        return 2*((s+b)*np.log(1+s/b)-s)
        #return s**2/(s+b)

    def signN(self, x, c, n , verbose = False, signal = 1, getCuts = False):
        p = self.splines
        b = self.bs_ratio
        if n == 1:
            s = self.sign1(c,b*self.e_b(c))
            if verbose:
                #signal, background, significance**2
                print np.array([c,b*self.e_b(c),s])*signal
            if getCuts:
                return (s*signal,[self.getVal(c,self.s_hist.eff,self.s_hist.x)])
            return s*signal
        else:
            x2 = self.getBinCut(x,p[n-1])
            s = self.sign1(c-x,b*(self.e_b(c)-self.e_b(x)))
            if verbose:
                #signal, background, significance**2
                print np.array([b*(self.e_b(c)-self.e_b(x)),c-x,s])*signal

            if getCuts:
                sig, cuts = self.signN(x2,x,n-1,verbose=verbose,signal=signal,getCuts=True)
                return (sig + signal*s,[self.getVal(c,self.s_hist.eff,self.s_hist.x)] + cuts )

            return  self.signN(x2,x,n-1,verbose=verbose,signal=signal) + signal*s


    def fitBinCuts(self):
        print("INFO: Fit bin edges iteratively")
        self.splines =[lambda x: 0.5]*21
        fitboundaries = lambda n: [-0.001,-1.5+np.sqrt(n)*0.1]
        converged = True
        for n in range(2,16):
            scan_c = np.logspace(*fitboundaries(n))
            roots = []
            guess = min(0.095 * n,0.95)
            scan_copy = []
            for c in scan_c:

                x_0_p = guess * c

                d_sign_dx = lambda x: derivative(lambda y: self.signN(y,c,n),x,dx=0.0001*c)
                opt = optimize.root(d_sign_dx,x_0_p)
                x_0 = opt.x[0]
                if opt.success and 0 < x_0 < c:
                    guess = x_0/c
                    roots.append(x_0)
                    scan_copy.append(c)
                else:
                    print "root not found" 
                    
            try:
                q = UnivariateSpline(np.log10(scan_copy[::-1]),np.log10(np.array(roots)[::-1]),s=0.0001)
                self.splines[n] = q
                print "parameters for n="+str(n)+" found"
            except:
                print "ERROR: fit failed"
                converged = False
                break

        return converged

           
    def getBinEdges(self,n):
        if len(self.splines) <= n:
            if not self.fitBinCuts():
                return None
        print "Estimated events in each bin:"
        print "SIGNAL | BACKGROUND | SIGNIF.**2"
        sig2, cutlist = self.signN(self.getBinCut(1,self.splines[n]),1,n,verbose=True,getCuts=True,signal=self.total["signal"])
        cutlist.append(1.0)
        print "\nbin edges: " + str(cutlist)
        print "significance estimate: " + str(np.sqrt(sig2))
        print "significance: " + str(self.EvalSign(cutlist,plot=True))
        return cutlist

    
    def EvalSign(self,cutList,plot=False):
        r = 0
        r_i = []
        sig = self.total["signal"]*self.getVal(np.array(cutList),self.s_hist.x,self.s_hist.eff)
        bkg = self.total["background"]*self.getVal(np.array(cutList),self.b_hist.x,self.b_hist.eff)
        for i in range(1,len(cutList)):
            s = sig[i-1]-sig[i]
            b = bkg[i-1]-bkg[i]
            l = self.sign1(s,b)
            r_i.append(l)
            r += l
            #print [s,b,s**2/(s+b)]
        print "background in last bin: "+str(b)
        if self.makePlots and plot:
            plt.figure()
            plt.step(cutList[1:],-np.diff(sig)/self.total["signal"],label='sig',where='pre')
            plt.step(cutList[1:],-np.diff(bkg)/self.total["background"],label='bkg',where='pre')
            plt.step(cutList[1:],np.array(r_i)/r,where='pre',label='significance contribution')
            plt.legend()
            plt.savefig("%s/%s"%(self.plot_dir,"rebin_shape.pdf"))
        return np.sqrt(r)
    
    def showPlots(self):
        plt.show()
