# %%
import numpy as np
import random
import mnist_loader
import sklearn.metrics as sk
import time
import matplotlib.pyplot as plt
import data_sets_functions as dsf

# %%
class Activations():
    
    sigmoid = "sigmoid"
    tanh = "tanh"
    
    
    def sigmoid(self, z):
        return np.exp(z)/(1+np.exp(z))
    
    def diff_sigmoid(self, z):
        
        return self.sigmoid(z) * (1- self.sigmoid(z))
    
    def tanh(self, z):
        return (np.tanh(z)+1)/2
        
    def diff_tanh(self, z):
        return ( 1 - (self.tanh(z))**2 )/2


    
    


class Network:
    def __init__(self, size, shuf):
        
        self.size = size
        self.num_layers = len(size)
        self.activation = Activations()
        self.shuf = shuf
        self.weights = []
        self.biases = []
        self.monitor_training_data = True
        self.monitor_test_data = True
        self.monitor_trianing_cost = True
        self.monitor_test_cost = True
        
        self.training_cost = []
        self.accuracy_test = []
        self.accuracy_training = []
        self.test_cost = []
        #pass
    
    def cost(self, a, y):
        
        return 0.5*np.linalg.norm(a-y)**2
        #return sk.mean_squared_error(a, y)
    
    def diff_cost(self, a, y, z):
        return (a-y) #*self.activation.diff_sigmoid(z)
    
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation.sigmoid(np.dot(w, a)+b)
        return a
        
    def train(self, train, test, epochs, batch_size, eta):
        
        
        n = len(train)
        for j in range(epochs):
            batches = []
            if self.shuf == True: random.shuffle(train)
            for i in range(0, n, batch_size):
                batches.append(train[i:i+batch_size])
            
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            
            for batch in batches:
                nabla_b, nabla_w = self.SGD(batch)
                
                self.weights = [w-(eta/len(batch))*nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases, nabla_b)]
            print("{0} / {1}".format(j, epochs))
            
            if self.monitor_test_data:
                A = np.round(self.evaluate_test(test)/len(test),4)
                print("(Test) Epoch {0}: {1} / {2} = {3}".format(j, self.evaluate_test(test), len(test), A))
                self.accuracy_test.append(A)
                #print("Epoch {0}: {1} / {2} = {3}".format(j, self.evaluate(test, False), len(test), self.evaluate(test, False)/len(test)))
            if self.monitor_training_data:
                A = np.round(self.evaluate_train(train)/len(train),4)
                print("(Train) Epoch {0}: {1} / {2} = {3}".format(j, self.evaluate_train(train), len(train), A))
                self.accuracy_training.append(A)
            if self.monitor_trianing_cost: 
                cost_train = self.total_cost(train, True)
                self.training_cost.append(cost_train/len(train))
            
            if self.monitor_test_cost:
                cost_test = self.total_cost(test, False)
                self.test_cost.append(cost_test/len(test))
                
            if j == epochs-1: #evaluate in last epoch
                print("Epoch {0}: {1} / {2} = {3}".format(j, self.evaluate_test(test), len(test), self.evaluate_test(test)/len(test)))
                print("Epoch {0}: {1} / {2} = {3}".format(j, self.evaluate_train(train), len(train), self.evaluate_train(train)/len(train)))

    def total_cost(self, data, convert = True):
        #if data are a traning data then convert must be True
        cost = 0
        if convert == True:
            for x, y in data:
                a = self.feedforward(x)
                cost += self.cost(a, y)
        else:
            for x,y in data:
                a = self.feedforward(x)
                y = self.vectorized_result(y)
                cost += self.cost(a, y)
        return cost
        """batches = [
            train[i:i+batch] for i in range(0, n, batch)
            ]"""
        
        #print(batches)
    def vectorized_result(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the j'th position
        and zeroes elsewhere.  This is used to convert a digit (0...9)
        into a corresponding desired output from the neural network.

        """
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
    
    def eval_fashion(self, data, is_fashion):
        
        if is_fashion == True: #Training_data
            results = [(np.argmax(self.feedforward(x[0])), x[1])
                        for x in data]
            #print(results)
        else: #Test_data
            results = [(np.argmax(self.feedforward(x)), x[1])
                        for (x, y) in data]
            #print(results)
        return sum(int(x == y) for (x, y) in results)
        
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
    
    def evaluate_train(self, data, convert = True):
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
    
    def evaluate_test(self, data, convert = False):
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
    
    def detect_misses(self, data, convert = False):
        misses = []
        if convert == True: #Training_data
            results = [(x[0], x[1], np.argmax(self.feedforward(x[0])), np.argmax(x[1]))
                        for x in data]
            #print(results)
        else: #Test_data
            results = [(x, np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
            #print(results)
        if convert == True:
            for (a,b,c,d) in results:
                if c != d:
                    misses.append( (a,b) )
        else:
            for (a,b,c) in results:
                if b != c:
                    misses.append( (a, c)  )
           
        return misses
        #return sum(int(x == y) for (z,x, y) in results)
        
    def SGD(self, batch):
        for i in batch:
            
            x = i[0]
            y = i[1]
            #print(x, y)
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
            delta = self.diff_cost(activations[-1], y, zs[-1])*self.activation.diff_sigmoid(zs[-1])
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
    
    
        
        
    
    
    def parameters_initializer(self, better_init = False):
        for i in range(self.num_layers):
            if better_init == True:
                self.weights = [np.random.randn(y, x)/np.sqrt(x)
                            for x, y in zip(self.size[:-1], self.size[1:])]
            else:
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
        print("Layers : ", self.size)
        print("Biases : ", biases)
        print("Weights : ", weights)
        print("Parameters : ", biases+weights)
    
    


# %%

#net = Network([784, 80, 30, 10])

#(training_data, test_data) = data_sets_functions.mnist_fashion()


"""
in dsf.below()
def cifar_dataset():
def mnist(): 
def mnist_fashion_dataset():
    
"""
#(train, test) = dsf.mnist()
#(train, test) = dsf.mnist_fashion_dataset()
#(train, test) = dsf.cifar_dataset_flatten()

"""(train, test) = dsf.mnist()
input_size = train[0][0].shape[0]
net = Network([input_size, 30, 10])
net.net_parameters()
net.parameters_initializer()

net.train(train, test, 30, 10, 3)"""

# %%
"""(train, test) = dsf.mnist_fashion_dataset()
input_size = train[0][0].shape[0]
net = Network([input_size, 30, 10])
net.net_parameters()
net.parameters_initializer()

net.train(train, test, 30, 10, 3)"""

# %%
#(train, test) = dsf.mnist_dataset()
(train, test) = dsf.cifar_dataset_flatten()
train = train[:10000]
#(train, test) = dsf.mnist_fashion_dataset()
input_size = train[0][0].shape[0]

"""net1 = Network([input_size, 30, 10], False)
net1.net_parameters()
net1.parameters_initializer()
net1.train(train, test, 50, 10, 3)"""

#train = train[:5000]
epoch = 150
batch = 10
eta = 3

# %%
#True = shufle
net1 = Network([input_size,30, 10], False)
net1.net_parameters()
net1.parameters_initializer(True)

net1.train(train, test, epoch, batch, eta)


# %%

net2 = Network([input_size, 30, 10], True)
net2.net_parameters()
net2.parameters_initializer(True)

net2.train(train, test, epoch, batch, eta)
# %%

plt.plot(net1.accuracy_test, label = "shuf = False")
plt.plot(net2.accuracy_test, label = "shuf = True, init = true")
plt.title("test_accuracy")
plt.legend()
plt.show()


plt.plot(net1.accuracy_training, label = "shuf = False")
plt.plot(net2.accuracy_training, label = "shuf = True, init = true")
plt.title("train_accuracy")
plt.legend()
plt.show() 

plt.plot(net1.training_cost, label = "shuf = false")
plt.plot(net2.training_cost, label = "shuf = true, init = true")
plt.title("train_cost")
plt.legend()
plt.show() 

plt.plot(net1.test_cost, label = "shuf = false")
plt.plot(net2.test_cost, label = "shuf = true, init = true")
plt.title("test_cost")
plt.legend()
plt.show() 

"""plt.plot(net2.test_cost, label = "test_cost")
plt.plot(net2.training_cost, label = "train_cost")
plt.plot()
plt.show()"""
# %%
#A = train[0]
"""for i in range(10,20):
    img = train[i][0].reshape(32,32,3)
    
    plt.imshow(img)
    plt.show()"""
    
    






















