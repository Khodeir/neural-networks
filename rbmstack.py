from scipy.io import savemat, loadmat
from rbm import RBM
from layer import sample_binary_stochastic
class RBMStack(object):
    def __init__(self, data, rbms):
        '''Assumes for all rbm[i] in rbms, rbm[i].numhid == rbm[i+1].numvis'''
        self.rbms = rbms
        self.data = data

    def data_for(self, mac_i):
        probs = data = self.data
        for mac in self.rbms[0:mac_i]:
            data = mac.sample_hid(data)
            probs = mac.get_hidlayer().probs
        return data, probs

    def train(self, macindex, K=1, epochs=100, learning_rate=0.1, weightcost=0.1, dropoutrate=0, data=None):
        data = data if data is not None else self.data_for(macindex)[1]
        for epoch in range(epochs):
            recons = self.rbms[macindex].train(data, K, learning_rate, weightcost, dropoutrate)
        return recons

    def top_down(self, case):
        for mac in self.rbms[::-1]:
            case = mac.sample_vis(case)
        return case

    def bottom_up(self, case):
        for mac in self.rbms:
            case = mac.sample_hid(case)
        return case

    def up_and_down(self, case, K):
        for k in range(K):
            case = self.bottom_up(case)
            case = self.top_down(case)
        return case

    def save_to_matfile(self, matfilename):
        data = {}
        data['numrbms'] = len(self.rbms)
        data['stack_data'] = self.data
        for mac_i in range(len(self.rbms)):
            data[str(mac_i)+"_visbias"] = self.rbms[mac_i].get_vislayer().bias
            data[str(mac_i)+"_hidbias"] = self.rbms[mac_i].get_hidlayer().bias
            data[str(mac_i)+"_vishid"] = self.rbms[mac_i].get_vishid()
        savemat(matfilename, data)

    @classmethod
    def load_from_matfile(cls, matfilename):
        data = loadmat(matfilename)
        stack_data = data.get('stack_data')
        numrbms = data.get('numrbms')
        rbms = []
        for mac_i in range(numrbms):
            vbias = data.get(str(mac_i)+"_visbias")
            hbias = data.get(str(mac_i)+"_hidbias")
            vishid = data.get(str(mac_i)+"_vishid")
            rbm = RBM(vbias.size, hbias.size)
            rbm.get_vislayer().bias = vbias
            rbm.get_hidlayer().bias = hbias
            rbm.weights[0] = vishid
            rbms.append(rbm)
        return cls(stack_data, rbms)
