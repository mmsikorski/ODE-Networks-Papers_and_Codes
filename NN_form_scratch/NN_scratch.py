# %%
import numpy as np
import random
import mnist_loader
import sklearn.metrics as sk

training_data_ZIP, validation_data_ZIP, test_data_ZIP = mnist_loader.load_data_wrapper()
training_data = list(training_data_ZIP)
validation_data = list(validation_data_ZIP)
test_data = list(test_data_ZIP)

# %%
class Activations():
    
    sigmoid = "sigmoid"
    tanh = "tanh"
    
    
    def sigmoid(self, z):
        return np.exp(z)/(1+np.exp(z))
    
    def diff_sigmoid(self, z):
        
        return self.sigmoid(z) * (1- self.sigmoid(z))
    
    def tanh(self, z):
        return np.tanh(z)
        
    def diff_tanh(self, z):
        return ( 1 - (self.tanh(z))**2 )


    
    


class Network:
    def __init__(self, size):
        
        self.size = size
        self.num_layers = len(size)
        self.activation = Activations()
        
        self.weights = []
        self.biases = []
        self.monitor_training_data = True
        #pass
    
    def cost(self, a, y):
        return sk.mean_squared_error(y, a)
    
    def diff_cost(self, a, y, z):
        return (a-y)*self.activation.diff_sigmoid(z)
    
    """def feedforward(self, z):
        pass"""
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation.sigmoid(np.dot(w, a)+b)
        return a
        
    def train(self, train, test, epochs, batch_size, eta):
        
        
        n = len(train)
        for j in range(epochs):
            batches = []
            random.shuffle(training_data)
            for i in range(0, n, batch_size):
                batches.append(train[i:i+batch_size])
            
            for batch in batches:
                nabla_b, nabla_w = self.SGD(batch)
                
                self.weights = [w-(eta/len(batch))*nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases, nabla_b)]
            print(j)
            if self.monitor_training_data:
                print("Epoch {0}: {1} / {2} = {3}".format(j, self.evaluate(train, self.monitor_training_data), len(training_data), self.evaluate(train, self.monitor_training_data)/len(train)))
    
        """batches = [
            train[i:i+batch] for i in range(0, n, batch)
            ]"""
        
        return batches
        #print(batches)
    
    def evaluate(self, data, convert = False):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        #np.argmax(x[1])
        if convert == True: #Training_data
            results = [(np.argmax(self.feedforward(x[0])), np.argmax(x[1]))
                        for x in data]
            #print(results)
        else: #Test_data
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
            #print(results)
        return sum(int(x == y) for (x, y) in results)
    
    def SGD(self, batch):
        for i in batch:
            
            x = i[0]
            y = i[1]
            """Return a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``."""
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x #    sport! - 
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = self.activation.sigmoid(z)
                activations.append(activation)
            # backward pass
            delta = self.diff_cost(activations[-1], y, zs[-1])
           # delta = (self.cost).delta(zs[-1], activations[-1], y)
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = self.activation.diff_sigmoid(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (nabla_b, nabla_w)
        
        
        
        
        
        pass
    
    def parameters_initializer(self):
        for i in range(self.num_layers):
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(self.size[:-1], self.size[1:])]
            self.biases = [np.random.randn(x,1) for x in self.size[1:]]
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
        print("Parameters : ", biases+weights)
    
    


net = Network([784, 30, 10])
net.net_parameters()
net.parameters_initializer()
A = net.train(training_data, test_data, 20, 20, 5)
#print(A)


















