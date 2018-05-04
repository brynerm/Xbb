class Jet(object):
    def __init__(self, pt, eta, phi, mass, btag=0.0, index=0):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass
        self.btag = btag
        self.index = index
