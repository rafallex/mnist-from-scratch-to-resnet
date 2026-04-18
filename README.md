# Deep Learning (1RT720)

Coursework for Deep Learning (1RT720) in the M.Sc. in Image Analysis and Machine Learning at Uppsala University. Two hand-in assignments building from a NumPy-only implementation of a fully-connected neural network to a residual CNN trained with Adam, Batch Normalization, and Dropout on MNIST.

## Contents

- **ha1-mnist-numpy/** -- Fully-connected neural network implemented in NumPy only, supporting arbitrary depth (e.g. [784, 128, 64, 10]). Includes He initialization, numerically-stable softmax + cross-entropy, a fully vectorised forward / backward pass, and mini-batch SGD. Compares ReLU and sigmoid activations; reaches **98.04% test accuracy** on MNIST with the two-hidden-layer architecture.
- **ha2-cnn-pytorch/** -- Systematic progression through modern CNN components in PyTorch, with learning curves, parameter counts, and analysis of the generalisation gap at each step:
  - PyTorch re-implementation of the HA1 fully-connected baseline -- 98.20% test
  - Plain convolutional network with SGD -- 98.20% test, 5.1x fewer parameters than the FC baseline
  - Same CNN with Adam -- 98.99% test, faster convergence
  - Residual CNN with Adam -- 99.15% test
  - Residual CNN + Batch Normalization -- 99.12% test
  - Residual CNN + Dropout -- **99.29% test**

## Notes

The goal of the coursework was to understand each building block (initialization, activation derivatives, backpropagation, optimiser math, skip connections, regularisation) rather than to reach state-of-the-art accuracy on MNIST. Reports are included alongside the code. Generative AI was used as a coding assistant for boilerplate and report drafting; architectural, optimisation, and analysis decisions are my own, as disclosed in each report.
