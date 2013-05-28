class RBMMachine(object):
    def __init__(self, data, layers):
        self.layers = layers
        self.data = data

    def train(self, data, epochs=100, K=1):
        for mac in self.layers:
            mac.train(data, K, epochs)
            data = mac.sample_hid(data)

    def top_down(self, case, binary_stochastic=False):
        for mac in self.layers[1:][::-1]:
            case = mac.sample_vis(case, binary_stochastic)
        return self.layers[0].sample_vis(case)

    def bottom_up(self, case, binary_stochastic=True):
        for mac in self.layers:
            case = mac.sample_hid(case, binary_stochastic)
        return case

    def up_and_down(self, case, K):
        for k in range(K):
            case = self.top_down(case)
            case = self.bottom_up(case)
        return case
