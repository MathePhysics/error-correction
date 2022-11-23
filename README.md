# error-correction
Error correction of neural network solutions to Poisson’s equation.

This codebase implements a modified deep Galerkin method to numerically solve differential equations with neural networks. The method is modified through the construction of error correction networks that can appreciably improve the accuracy of solution.

Alongside error correction, sinusoidal representation networks (SIRENs) and random Fourier features are implemented to the same end. The validity of all such strategies is discussed in the accompanying paper ‘Enhancing Neural Network Differential Equation Solvers.’ 

# Poisson's equation
This codebase solves a particular class of Poisson’s equations: 

$$ \nabla^2(\phi) = -\omega^2 d\prod_{i=1}^d \sin(\omega x_i) $$

in $\Omega=\[-\pi,\pi\]^d$ and $\phi=0$ on $\partial\Omega$. This class is chosen because it admits a nice, unique solution $\phi=\prod_\limits{i=1}^d\sin(\omega x_i)$. The user may find it easier to parameterise the Poisson’s equation through its solution but, crucially, the neural network does not and is only given the equation itself.

# Getting started

The codebase relies on three scripts:

1. _data.py_ samples the points upon which the neural network is trained and defines the equations which must be solved.
2. _network.py_ initialises the SIREN.
3. _train.py_ trains the SIREN, and defines the solution (for plotting purposes) and plotting functions.

In the experiments folder, these are imported into executable scripts which serve as templates for the user. Their names follow the convention _{d}D{omega}-{batch size}x{epochs}-EC{order}_ and, once run, plots of the solutions and training metrics are saved under similar names too.

For an explanation of error correction, please refer to the accompanying paper. The codebase is built to perform an arbitrary number of error corrections, but the plots will suffer if this number is too large. Practically, also, only a couple of error corrections are recommended.
