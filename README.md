# b-cnns: Bayesian Convolutional Neural Networks
Bayesian Convolutional Neural Networks

This repository contains codes to run Bayesian Convolutional Neural Networks on CIFAR-10. It is shown (in `run_cnn.py`) that small convolutional net (with 26,000 parameters), is able to achieve 71% test set accuracy on CIFAR-10. A small dense network is able to achieve 55% accuracy. All computation was done on a local RTX 2080, with 8GB GPU RAM.

The codebase relies on `JAX`, `NumPyro` and some utilites of `tfdatasets`.
