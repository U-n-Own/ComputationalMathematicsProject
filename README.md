
# Computational Mathematics Project 27

### Project Members:
1. MAT: 616233, Lorenzo Pace
2. MAT: 667591, Vincenzo Gargano

### Project Track:

**(P)** is a sparse linear system of the form:

\[
\begin{bmatrix}
D & E^T\\
E & 0
\end{bmatrix}
\begin{bmatrix}
x\\y
\end{bmatrix}
=
\begin{bmatrix}
b\\c
\end{bmatrix}
\]

where \(D \in \mathbb{R}^{m \times m}\) is a diagonal positive definite matrix (i.e., \(D = \operatorname{diag}(D) > 0\)) and \(E \in \mathbb{R}^{(n-1) \times m}\) is obtained by removing the last row from the node-arc incidence matrix of a given connected directed graph. These problems arise as the KKT system of the convex quadratic separable Min-Cost Flow Problem; hence, you can look, e.g., [here](https://commalab.di.unipi.it/datasets/mcf) for ways to generate meaningful instances of the problem.

**(A1)** is GMRES, and you must solve the internal problems

\[
\min \bigl\| H_n y - \|b\| e_1 \bigr\|
\]

by updating the QR factorization of \(H_n\) at each step: given the QR factorization of \(H_{n-1}\) computed at the previous step, apply one more orthogonal transformation to compute that of \(H_n\).

**(A2)** is the same GMRES, but using the following preconditioner:

\[P = \begin{bmatrix}
     D^{-\frac12} & 0\\
     0 & R^{-T}
    \end{bmatrix}\]
where \(S\) is either \(S=ED^{-1}E^T\)
 or a sparse approximation of it (to obtain it, for instance, replace the smallest off-diagonal entries of \(S\)
 with zeros), and \(S = R^TR\) is its Cholesky factorization.


### Command Line Tool

#### Dependencies

The source code is written in Julia. The dependencies can be installed by running:

```bash
julia requirements.jl
```
which will install the packages used in our project.

#### Usage

By running `julia main.jl --help`, you will obtain the following:

```
usage: main.jl [--iterations ITERATIONS] [--restart RESTART]
               [--diagonal DIAGONAL] [--tol TOL] [--step STEP]
               [--precond] [--restarted] [-h] mode data

positional arguments:
  mode                  Program mode. Can be "compute" or "bench".
  data                  Path to the problem data.

optional arguments:
  --iterations ITERATIONS
                        Number of iterations. If 0, uses size of
                        matrix. (type: Int64, default: 0)
  --restart RESTART     Restart parameter of restarted algorithm.
                        (type: Int64, default: 300)
  --diagonal DIAGONAL   Can be "ones", "uniform" or "quadratic"
                        (default: "quadratic")
  --tol TOL             Tolerance parameter of restarted algorithm.
                        (type: Float64, default: 1.0e-9)
  --step STEP           Step between iter values in bench mode. If 0,
                        uses iterations/4 (type: Int64, default: 0)
  --precond             Preconditioned? [used in compute mode].
  --restarted           Restarted? [used in compute mode].
  -h, --help            show this help message and exit
```

#### Examples

The following will compute the residual obtained by running the non-preconditioned, non-restarted algorithm with ones on the diagonal and a limit of 200 iterations.
```
$ julia main.jl compute ./dataset/net10_8_1.dmx_out.npz --diagonal ones  --iterations 200
Compute mode
============
Performing 200 iterations:
residual: 1.8868471430781007e-10 computed in 0.108904859s
```

The following will show a scatter plot comparing the performance of the different algorithms. Since the step is not defined as a command-line option, it is set by default to a quarter of the number of iterations.
```
$ julia main.jl bench ./dataset/net10_8_1.dmx_out.npz --diagonal ones  --iterations 200
Performing 200 iterations with step = 50
 * Breakdown
 * [...]
 * Did not converge within 151 iterations. Final residual norm: 5.1003964751279926e-9
press enter to terminate program.
```
