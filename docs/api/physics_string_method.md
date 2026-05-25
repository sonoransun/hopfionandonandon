# `hopfion.physics.string_method`

Minimum-energy path (MEP) and activation barrier between two spin textures, via
the simplified string method, plus an Arrhenius lifetime helper. The barrier
feeds SOP-003 (lifetime estimation) together with the Hessian pre-factor from
[`physics/hessian.py`](physics_hessian.md).

## Public API

| Symbol | Purpose |
|---|---|
| `string_method(m_start, m_end, grid, ep, n_images=7, n_iter=200, dt=0.01, climbing=False, reparam_every=1)` | relax a chain of images onto the MEP; returns a `StringMethodResult` |
| `StringMethodResult` | dataclass: `barrier`, `saddle_index`, `energies` (per image), `images` |
| `arrhenius_lifetime(barrier, kT, prefactor=1.0, tau0=1.0)` | $\tau = (\tau_0/\text{prefactor})\,e^{\,\text{barrier}/kT}$ (overflow-guarded) |

## Method

The path is discretized into `n_images` configurations (endpoints fixed). Each
iteration: (1) one damped-LLG step per interior image (`llg.damped_step`) pulls it
downhill; (2) **reparameterization** redistributes images to equal arc length along
the string — geodesically (SLERP on $S^2$), which replaces NEB spring forces (no
spring constant to tune) and avoids the spurious exchange energy a linear blend
would inject. `barrier = max(E_image) - E_start`; `climbing=True` freezes the peak
image so it isn't pulled off the ridge.

## Caveats

- **Resolution-dependent.** A topological barrier (e.g. hopfion → uniform) can read
  as *downhill* on a coarse grid because the lattice unwinds the topology cheaply.
  Validated against the analytic anisotropy double-well (`tests/test_string_method.py`),
  where the barrier is $\approx K_u V$ with the in-plane saddle at the path midpoint.
- **NumPy backend only** (an offline analysis tool, like `hessian.py`).
- `dt` is bounded by the same exchange-stability limit as relaxation, `dt < dx²/(4 A_ex)`.

## Usage

```python
from hopfion.field import uniform
from hopfion.physics.string_method import string_method, arrhenius_lifetime
r = string_method(m_hopfion, uniform(grid, (0, 0, 1)), grid, ep, n_images=11, n_iter=400)
tau = arrhenius_lifetime(r.barrier, kT=0.5, prefactor=2.0)
```

## See also

- [docs/sop/ESTIMATE_LIFETIME.md](../sop/ESTIMATE_LIFETIME.md) — Method B workflow.
- [physics_hessian.md](physics_hessian.md) — the Arrhenius pre-factor.
