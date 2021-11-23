import numpy as np


class Network:
    def __init__(self, size):
        
        self.size = size
        self.num_layers = len(size)
        
        self.weights = None
        self.biases = None
        #pass

    def train():
        pass
    
    def net_parameters(self):
        biases = 0
        weights = 0
        for i in range(1, self.num_layers):
            biases += self.size[i]
            if i > 0 and i < self.num_layers:
                weights += self.size[i-1]*self.size[i]
            
        print("Biases : ", biases)
        print("Weights : ", weights)
    
    


net = Network([784, 10, 10])
net.net_parameters()