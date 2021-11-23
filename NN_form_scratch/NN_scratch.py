import numpy as np
import random

class Network:
    def __init__(self, size):
        
        self.size = size
        self.num_layers = len(size)
        
        self.weights = []
        self.biases = []
        #pass

    def train(self):
        pass
    
    
    def parameters_initializer(self):
        for i in range(self.num_layers):
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(self.size[:-1], self.size[1:])]
            #self.weights = [np.random.rand((self.size[i], self.size[i+1])) for i in range(self.num_layers)]
            
            
            
            
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
net.parameters_initializer()
