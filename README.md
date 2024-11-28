
# Computational Mathematics Project 27

## Project Track: Solving Minimum Cost Flow Problem

### Project Members:
1. MAT: 616233, Lorenzo Pace
2. MAT: 667591, Vincenzo Gargano

### Project Description:

The Minimum Cost Flow Problem (MCFP) is a well-known optimization problem that can be solved by using different algorithms. In this project, we will implement the solution in two fashion using an iterative algorithm called GMRES (Generalized Minimal Residual Method) 

## Roadmap

   1. Generate some problem instances in the follwoing form:
   $$\begin{bmatrix}
    Q & E^T \\
    E & 0
   \end{bmatrix} \begin{bmatrix}
    x \\
    y
   \end{bmatrix} = \begin{bmatrix}
    b \\
    c
    \end{bmatrix}$$
   where $Q$ is a diagonal matrix and $E$ is the adjacency matrix of the graph.

   With:

$$
    \textbf{min}\{cx: Ex=b, 0 \leq x\leq u \}
$$

   1. Implement the GMRES algorithm to solve the MCFP problem.
   
   2. Use a Shur complement to solve the MCFP problem.

---

