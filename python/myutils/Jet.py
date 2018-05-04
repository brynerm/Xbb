import ROOT
class Jet(object):
    def __init__(self, pt, eta, phi, mass, btag=0.0, index=0):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass
        self.btag = btag
        self.index = index

    @staticmethod
    def deltaR(jet1, jet2):
        deltaPhi = ROOT.TVector2.Phi_mpi_pi(jet1.phi-jet2.phi)
        deltaEta = jet1.eta-jet2.eta
        return ROOT.TMath.Sqrt(deltaEta*deltaEta + deltaPhi*deltaPhi)
