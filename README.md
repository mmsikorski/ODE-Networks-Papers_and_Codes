# ODE Neural Networks

Below I gather interesting and valuable papers about Physics-Informed Neural Networks, so-called ODE-NN (Ordinary differential equation - Neural Network)

ODE-NNs are rather new field in neural networks that attract people due to their advantage over standard neural networks that need a lot of memory what is crucial in big networks which sizes are huge. In ODE-NNs we don't have to focus on a weights in layers, we have one layer which is parametrized by a set of boundary conditions took into account in cost function.

- [ODE Neural Networks](#sections)
	- [The most important papers](#The-most-important-papers)

## The most important papers 

1. **Neural differential ordinary equations**, *Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud*, NeurIPS 2018, 2021. [[paper](https://arxiv.org/pdf/1806.07366.pdf)]

It's high likely that paper above is a milestone in an area of neural networks that involves physics. In this paper, main ideas are in using theory od dynamical systems to construct architecture of neural network and in observation that a index used to numerate hidden layers in neural network can be viewed, in terms of limitation when index undergoes to infinity, as a time step that indicate when an information flow through a network.