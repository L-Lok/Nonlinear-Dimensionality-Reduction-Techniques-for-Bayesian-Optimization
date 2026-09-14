# Benchmark library

This package contains only the five full-rank objectives and the
curved-preimage objectives used by the manuscript studies: Ackley, Rosenbrock,
Rastrigin, canonical Levy, canonical Styblinski–Tang, and the curved-preimage
family with its internal Branin, Ackley, Rosenbrock, and Rastrigin bases.

The PyTorch objective classes follow the BO convention: `func(x)` returns the
internal maximization target, equal to the negative of the original
minimization objective. NumPy problem classes expose the original objective
for external optimizers.

The curved-preimage family supplies the benchmark geometry for the BO-VAE
versus EGORSE study.
