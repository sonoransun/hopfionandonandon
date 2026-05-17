"""Phase-B physics extensions.

* ``integrators`` — Heun (the default), RK4, and adaptive variants.
* ``two_temp`` — two-temperature ODE + Heun-Stratonovich stochastic LLG.
* ``dipolar`` — long-range magnetostatic field via FFT convolution.
* ``hessian`` — eigenmode analysis around relaxed configurations.
* ``bilayer`` — twisted-bilayer micromagnetics with interlayer coupling.

The Phase-A modules (``hopfion.energy``, ``hopfion.llg``, etc.) keep their
existing signatures; this sub-package layers on top of them.
"""
