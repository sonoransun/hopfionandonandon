# `hopfion.physics.dipolar`

Long-range magnetostatic (dipolar / demag) field via FFT convolution. Periodic BC only; k=0 mode is zeroed (uniform-mean convention).

## Math

$$\hat H_{\rm dip,i}(\mathbf{k}) = -\frac{k_i k_j}{|\mathbf{k}|^2}\,\hat m_j(\mathbf{k}), \qquad k \ne 0; \quad \hat H_{\rm dip}(0) = 0.$$

Energy: $E_{\rm dip} = -\tfrac{1}{2}\int \mathbf{m}\cdot\mathbf{H}_{\rm dip}\,d^3r$. Always non-negative.

## Public API

| Function | Returns |
|---|---|
| `dipolar_field(m, grid)` | $\mathbf{H}_{\rm dip}$ of shape $(3, n_x, n_y, n_z)$ |
| `dipolar_energy(m, grid)` | scalar (≥ 0) |

## Enabling via `EnergyParams`

```python
from hopfion.energy import EnergyParams
ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, dipolar=True)  # default False
```

When `dipolar=True`, `total_energy` and `effective_field` route through the dipolar term.

## Caveats

- Periodic-BC convention: a uniformly magnetized periodic box has *zero* dipolar field (the k=0 mode is zeroed). This is the right choice when the user means "long-wavelength dipolar coupling within the unit cell" rather than "shape-dependent demag of a finite sample."
- Numeric cost: dominated by one FFT of `m` per call. Cost grows as $N \log N$.

## See also

- [PHYSICS.md §3](../PHYSICS.md#3-micromagnetic-energy) — energy terms.
- [energy.md](energy.md) — `EnergyParams` routing.
