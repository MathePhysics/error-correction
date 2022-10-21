# error-correction
Error correction of neural network solutions to Poisson’s equation.

This codebase implements a modified deep Galerkin method to numerically solve differential equations with neural networks. The method is modified through the construction of error correction networks that can appreciably improve the accuracy of solution.

Alongside error correction, sinusoidal representation networks (SIRENs) and random Fourier features are implemented to the same end. The validity of all such strategies is discussed in the accompanying paper ‘Enhancing Neural Network Differential Equation Solvers.’

# Poisson's equation
This codebase solves a particular class of Poisson’s equations: 

$$ \nabla^2(\phi) = -\omega^2 d\prod_{i=1}^d \sin(\omega x_i) $$

in $\Omega=\[-\pi,\pi\]^d$ and $\phi=0$ on $\partial\Omega$. This class is chosen because it admits a nice, unique solution $\phi=\prod_\limits{i=1}^d\sin(\omega x_i)$. The user may find it easier to specify this Poisson’s equation through its solution but, crucially, the neural network does not and is only given the equation itself.

# Getting started

The codebase relies on three scripts:

1. _data.py_ samples the points upon which the neural network is trained and defines the equations which must be solved.
2. _network.py_ initialises the SIREN.
3. _train.py_ trains the SIREN, and defines the solution (for plotting purposes) and plotting functions.

In the experiments folder, these are imported into executable scripts which serve as templates for the user. Their names follow the convention _{d}D{omega}-{batch size}x{epochs}-EC{order}_ and, once run, plots of the solutions and training metrics are saved under similar names too.

For an explanation of error correction, please refer to the accompanying paper. The codebase is built to perform an arbitrary number of error corrections, but the plots will suffer if this number is too large. Practically, also, only a couple of error corrections is recommended. The user may see the merit in this by comparing results with epochs (iterations) equal to $\[N\], \[N/2, N/2\]$ and $\[N/3, N/3, N/3\]$, which ensures the same number of epochs across order zero, one and two error corrections respectively, and all other hyperparameters fixed.

# Going further

To solve a different Poisson’s equation, the user must make a few changes:

* In _data.py_, change _f_ to reflect the new RHS of the equation. Change the corners of the hypercube domain in the _init_ method.
* In _train.py_, change _phi_ to reflect the corresponding solution. If the new equation is non-zero (but constant) on the boundary, subtract this constant off the second part of the loss function.

If the equation is non-constant on the boundary, or you wish to solve over a new domain, significant modifications are required.

To solve a different differential equation:

* In _data.py_, duplicate the _Poisson_ class and change the _diffeq_ method. Instructions on how to formulate the recursive error correction equations can be found in the accompanying paper.

Good luck!
