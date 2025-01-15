# Complexity

## Non precond

The time complexity of Lanczos+QR is actually:
$$O(nm^2 + n^2)$$ 
and with sparsity:
$$O(n \cdot NNZ(A) + n^2$$

## Precond
A cubic factor is added due to the Cholesky computation.
$$O(n^3 + nm^2 + n^2)$$ 