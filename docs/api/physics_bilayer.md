# `hopfion.physics.bilayer`

Twisted-bilayer micromagnetics. Two spin fields $\mathbf{m}^{(1)}, \mathbf{m}^{(2)}$ co-evolve under their own energy plus an interlayer-exchange coupling modulated by the moiré registry function $\chi(\mathbf{r}; \theta)$.

This is the physically motivated successor to `moire.MoirePotential` — instead of a scalar anisotropy modulation acting on a single field, the moiré pattern arises naturally from the geometry of two stacked + rotated layers.

## Math

Interlayer-exchange energy:

$$E_{\rm inter} = -J_0 \int \chi(\mathbf{r};\theta)\,\mathbf{m}^{(1)}\cdot\mathbf{m}^{(2)}\,d^3r,$$

with

$$\chi(\mathbf{r};\theta) = \tfrac{1}{3}\sum_{i=1,2,3} \cos(\mathbf{q}_i\cdot\mathbf{r}), \qquad |\mathbf{q}_i| = \tfrac{4\pi}{a\sqrt 3}\cdot 2\sin(\theta/2).$$

Moiré period: $a_{\rm moire} = a / (2\sin(\theta/2))$.

## Public API

| Symbol | Purpose |
|---|---|
| `BilayerConfig(theta, a, J0)` | Geometric + coupling parameters |
| `.moire_period` | $a / (2\sin(\theta/2))$ |
| `registry_field(grid, cfg)` | $\chi(\mathbf{r}; \theta)$ as a 3D scalar field |
| `interlayer_field(m_other, chi, J0)` | $H^{(1)}_{\rm inter} = J_0\,\chi\,\mathbf{m}^{(2)}$ |
| `BilayerLLG(ep1, ep2, cfg, gamma, alpha, dt)` | Two-layer Heun integrator |
| `.step(m1, m2, grid, chi)` | one full-LLG step |
| `.relax_step(m1, m2, grid, chi)` | one damped step |
| `.relax(m1, m2, grid, n_steps)` | run damped to convergence |
| `.integrate(m1, m2, grid, n_steps, chi=None, step_callback=None)` | run full LLG |

## Usage

```python
from hopfion.physics.bilayer import BilayerConfig, BilayerLLG
cfg = BilayerConfig(theta=0.1, a=4.0, J0=0.3)
ep1 = ep2 = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7)
engine = BilayerLLG(ep1=ep1, ep2=ep2, cfg=cfg, dt=0.002)
m1, m2 = engine.relax(m1_init, m2_init, grid, n_steps=200)
```

## Caveats

- The moiré pattern is encoded in $\chi$, which is z-invariant (the moiré is a 2D phenomenon). Both layers share the same 3D grid for code simplicity.
- Setting `J0 = 0` exactly decouples the layers — they evolve independently. This is the smoke-test pathway.
- Phase B's bilayer doesn't yet have a recipe `kind: bilayer` (it's a stand-alone module). Recipe support is on the Phase C wish-list.

## See also

- [PHYSICS.md §7](../PHYSICS.md#7-moiré-modulation) — moiré math.
- [moire.md](moire.md) — Phase-A single-layer toy potential.
