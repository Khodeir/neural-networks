from scipy.io import savemat
class RBMStack(object):
    def __init__(self, data, rbms):
        self.rbms = rbms
        self.data = data

    def data_for(self, mac_i):
        data = self.data
        for mac in self.rbms[0:mac_i]:
            data = mac.sample_hid(data)
        return data

    def train(self, macindex, K=1, epochs=100, learning_rate=0.1, weightcost=0.1, dropoutrate=0):
        data = self.data_for(macindex)
        self.rbms[macindex].train(data, K, epochs, learning_rate, weightcost, dropoutrate)

    def top_down(self, case):
        for mac in self.rbms[1:][::-1]:
            case = mac.sample_vis(case)
        return self.rbms[0].sample_vis(case)

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
        for mac_i in range(len(self.rbms)):
            data[str(mac_i)+"_visbias"] = self.rbms[mac_i].get_vislayer().bias
            data[str(mac_i)+"_hidbias"] = self.rbms[mac_i].get_hidlayer().bias
            data[str(mac_i)+"_vishid"] = self.rbms[mac_i].get_vishid()
        savemat(matfilename, data)
