# ODE Neural Networks

Below I gather interesting and valuable papers about Physics-Informed Neural Networks, so-called ODE-NN (Ordinary differential equation - Neural Network)

ODE-NNs are rather new field in neural networks that attract people due to their advantage over standard neural networks that need a lot of memory what is crucial in big networks which sizes are huge. In ODE-NNs we don't have to focus on a weights in layers, we have one layer which is parametrized by a set of boundary conditions took into account in cost function.

- [ODE Neural Networks](#sections)
	- [The most important papers](#The-most-important-papers)
	- [Interesting papers that touch applications](#interesting-papers)

## The most important papers 

1. **Neural differential ordinary equations**, *Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud*, NeurIPS 2018. [[paper](https://arxiv.org/pdf/1806.07366.pdf)]

It's high likely that paper above is a milestone in an area of neural networks that involves physics. In this paper, main ideas are in using theory of dynamical systems to construct architecture of neural network and in observation that an index used to numerate hidden layers in neural network can be viewed, in terms of limit when index undergoes to infinity, as a time step that indicate when an information flow through a network.

The fundamental question is, like in each neural network, how to train neural network? In ODE-NN an answer is more complicated than in standard neural networks. Of course, an inspiration to learn neural network we take from standard neural networks, that means we must use a backpropagation algorithm, but it's not clear how to use that.

## Interesting papers that touch applications

1. **Latent ODEs for Irregularly-Sampled Time Series**, *Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud*, NeurIPS 2019. [[paper](https://papers.nips.cc/paper/2018/file69386f6bb1dfed68692a24c8686939b9-Paper.pdf)]