# Neural networks for turbulent channel flow

This project uses channel-flow DNS data at $Re = 5200$ to train two types of networks:

- A physics-informed neural network (PINN) for turbulent viscosity and diffusion.
- A neural network (NN) for the pressure-strain term.

A separate model is trained for each Reynolds stress: $\overline{u'^2}$ (`uu`), $\overline{v'^2}$ (`vv`), $\overline{w'^2}$ (`ww`) and $\overline{u'v'}$ (`uv`).

This README covers the root-level scripts:

| Script | Current behavior |
| --- | --- |
| `compute-pr-strain-load.py` | Trains pressure-strain NNs, or loads saved models with `--plot-only`. |
| `compute-vist-using-PINN-load.py` | Trains viscosity/diffusion PINNs from new weights, or loads saved checkpoints with `--plot-only`. |
| `compute-vist-using-PINN-save.py` | Intended to initialize checkpoints; currently has runtime issues listed below. |
| `RANS0_uu.py` | Verifies the solved `uu` equation using saved PINN diffusion and NN or standard pressure-strain. |

## Setup

Run commands from the project folder, where the Python scripts and `dns/` folder are located.

Install the Python packages:

```bash
python -m pip install numpy torch matplotlib scipy scikit-learn joblib
```

Keep these input files in `dns/`:

- `LM_Channel_5200_mean_prof.dat`
- `LM_Channel_5200_vel_fluc_prof.dat`
- `LM_Channel_5200_RSTE_k_prof.dat`
- `LM_Channel_5200_RSTE_uu_prof.dat`
- `LM_Channel_5200_RSTE_vv_prof.dat`
- `LM_Channel_5200_RSTE_ww_prof.dat`
- `LM_Channel_5200_RSTE_uv_prof.dat`
- `y_u_k_om_uv_5200-RANS-half-channel.txt` (the computational grid used by the PINN)

## Nomenclature

The flow is between two parallel walls. The mean flow is fully developed, so the profiles used here depend on the distance from the wall, $y$.

| Symbol / term | Meaning |
| --- | --- |
| $u'$, $v'$, $w'$ | Velocity fluctuations in the streamwise, wall-normal and spanwise directions; an overbar denotes an average. |
| $\overline{u'^2}$, $\overline{v'^2}$, $\overline{w'^2}$ (`uu`, `vv`, `ww`) | Mean squared velocity fluctuations in the three directions. |
| $\overline{u'v'}$ (`uv`) | Reynolds shear stress: the mean product of streamwise and wall-normal velocity fluctuations. |
| $\phi$ | The selected Reynolds stress component. |
| $k$ | Turbulent kinetic energy: $k = (\overline{u'^2} + \overline{v'^2} + \overline{w'^2})/2$. |
| $P_\phi$, $P_k$ | Production: transfer of energy from the mean flow into the selected Reynolds stress or turbulent kinetic energy. $P_k$ is `Pk` in the code. |
| $\Phi_\phi$ | Pressure-strain: redistribution of turbulent energy between velocity components for the selected stress. |
| $T_\phi$ | Turbulent transport of the selected Reynolds stress by velocity fluctuations. |
| $\Pi_\phi$ | Pressure transport associated with pressure fluctuations. |
| $D_{\nu,\phi}$ | Viscous diffusion caused by molecular viscosity: $D_{\nu,\phi} = \nu\,d^2\phi/dy^2$. |
| $\varepsilon_\phi$ | Viscous dissipation for the selected stress-budget component (`epsilon_phi`). |
| $\nu$ | Molecular kinematic viscosity; the scripts use the nondimensional value $\nu = 1/Re$. |
| $\nu_t$ | Turbulent viscosity used to model diffusion. It is learned separately for each stress here. |
| $u_\tau$ | Friction velocity used to scale velocity and wall distance. |
| $\delta$ | Channel half-height. |
| $U^+$ | Mean streamwise velocity in wall units: $U^+ = U/u_\tau$. |
| $y^+$ | Wall distance in viscous units: $y^+ = yu_\tau/\nu$. |
| $Re$ | Reynolds number used in the scripts: $Re = 5200$, with nondimensional viscosity $\nu = 1/Re = 1/5200$. |
| $D_{\mathrm{tot},\phi}$, $\mathcal{R}_\phi$ | Modeled total diffusion and the Reynolds stress equation residual. |
| RANS | Reynolds-averaged Navier–Stokes; the verification script solves the `uu` transport equation. |
| $U$ | Mean streamwise velocity, prescribed from DNS in the RANS verification. |
| $P_{11}$ | Production of $\overline{u'^2}$: $P_{11} = -2\overline{u'v'}\,dU/dy = 2P_k$. |
| $\Phi_{11}$ | Total pressure-strain for $\overline{u'^2}$, supplied by the trained NN or the standard AE.8 closure. |
| $\varepsilon$ | Scalar dissipation from the DNS $k$ budget; the RANS verification uses the isotropic component approximation $2\varepsilon/3$. |

### Equations and explanation

The steady Reynolds stress balance is written as:

$$
P_\phi + T_\phi + D_{\nu,\phi} + \Phi_\phi + \Pi_\phi - \varepsilon_\phi = 0.
$$

The PINN models the combined turbulent and pressure transport using a diffusion term:

$$
D_{\mathrm{tot},\phi}
= \frac{d}{dy}\left[(\nu + \nu_t)\frac{d\phi}{dy}\right]
= (\nu + \nu_t)\frac{d^2\phi}{dy^2}
+ \frac{d\nu_t}{dy}\frac{d\phi}{dy}.
$$

$$
\mathcal{R}_\phi = D_{\mathrm{tot},\phi} + P_\phi + \Phi_\phi - \varepsilon_\phi.
$$

Here $\phi$ is the selected Reynolds stress, while $\Phi_\phi$ is its pressure-strain term.
The PINN takes $y$ as input and predicts $\nu_t$. Stress profiles and source terms come from DNS. Training reduces the equation residual $\mathcal{R}_\phi$ and the boundary-condition error: $\nu_t = 0$ at the wall and a DNS-based value at the outer grid point.

The pressure-strain NN uses supervised learning. It takes four DNS inputs ($\phi$, $k$, $P_k$ and $P_\phi$) and predicts the total $\Phi_\phi$ for that component.

The scripts use $Re = 5200$ and $\nu = 1/Re = 1/5200$ in their nondimensional equations. DNS budget terms are divided by this viscosity to match the coordinate $y/\delta$.

For the RANS verification, `RANS0_uu.py` solves the steady channel-flow equation:

$$
0 = \frac{d}{dy}\left[(\nu + \nu_t)\frac{d\overline{u'^2}}{dy}\right]
+ P_{11} + \Phi_{11} - \frac{2}{3}\varepsilon,
\qquad
P_k = -\overline{u'v'}\frac{dU}{dy},\quad P_{11} = 2P_k.
$$

The saved `uu` PINN supplies fixed $\nu_t$. The NN predicts the complete $\Phi_{11}$ from the current solved stress, $k$, $P_k$ and $P_{11}$; `--no-nn` uses the standard AE.8 pressure-strain closure instead. DNS supplies $U$, $\overline{u'v'}$, $\overline{v'^2}$, $\overline{w'^2}$ and $\varepsilon$, while $k$ updates with the solved stress. The wall condition is $\overline{u'^2} = 0$, with zero diffusive flux at the channel centre. The approximation $2\varepsilon/3$ differs from the component dissipation used to train the PINN. Further derivation is in [RANS0_uu_notes.md](RANS0_uu_notes.md).

## Run Instructions

Run these commands from the project folder after completing the setup above.

### Initialization

`compute-vist-using-PINN-save.py` is intended to perform a short initialization run and save `nn/checkpoint_pinn_<stress>.ct`, including model/optimizer states and loss. Its intended command for all stress components is:

```bash
python compute-vist-using-PINN-save.py --stress all
```

Use `--stress uv` for one component or `--stress uu uv` for selected components. The default stress is `uv`.

**This command currently fails in the root-level script.** It calls `parse_stress_args()` without the required `args.stress`, defines `viscous` but later uses `viscos`, and uses the tuple returned by `deriv()` directly in tensor arithmetic. These issues must be corrected before using this initialization workflow.

The load script already passes the stress arguments, uses `viscos` consistently, and unpacks the derivative. Its direct training workflow below does not depend on repairing the save script.

### PINN training

The root-level load script can train directly; it does not require an initialization checkpoint:

```bash
python compute-vist-using-PINN-load.py --stress all --epochs 10000 --lr 0.01
```

Training options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--stress` | `uv` | One or more components: `uu`, `vv`, `ww`, `uv`, or `all`. |
| `--epochs` | `10000` | Number of training epochs. |
| `--lr` | `0.01` | Initial learning rate. |
| `--bc-weight` | `1000` | Weight of the boundary-condition loss. |

Aliases: `--stresses` and `--stress-choice` for `--stress`; `--max-epochs`, `--max_epchs` and `--max_epoch` for `--epochs`.

Use `--stress uv` for one component or `--stress uu uv` for selected components. Training uses Adam, halves the learning rate at 30%, 60%, and 85% of the requested epochs, and restores the weights associated with the lowest recorded total training loss.

Despite its name, the load script starts a new network in training mode. It does not resume a saved checkpoint or restore an optimizer state. Re-running training overwrites the selected components' trained artifacts.

For each stress, training saves:

- `nn/checkpoint_pinn_<stress>_trained.ct`: model weights, epoch count, stress name, best loss, and loss history.
- `nn/predicted_turb_diffusion_<stress>.txt`: columns `y_kom`, predicted turbulent diffusion, total diffusion, molecular viscous diffusion, and predicted $\nu_t$. The last column is in outer units, before division by `viscos`.
- Plots in `figs/`: viscosity, turbulent and total diffusion, diffusion components, molecular diffusion, inferred pure turbulent transport, PDE residual, and training loss.

The script prints diffusion RMSE/relative errors, PDE residual RMS/maximum, and predicted versus target boundary values. A completed training run alone does not establish a converged or physically accurate fit.

### NN training

Train all four components:

```bash
python compute-pr-strain-load.py --stress all --epochs 2000 --lr 0.005
```

Train only one component:

```bash
python compute-pr-strain-load.py --stress uu --epochs 2000
```

Train selected components:

```bash
python compute-pr-strain-load.py --stress uu uv --epochs 5000 --lr 0.001
```

Despite its filename, this script initializes and trains a new network. It scales the inputs and target, uses an 80/20 random training/test split, and restores the weights with the lowest test loss before saving. It prints errors and saves comparison plots. This split checks performance on points from the same DNS profile; it does not test a different Reynolds number. The test split also selects the best weights, and scalers are fitted on the full profile before splitting, so the reported test error is not an independent final generalization estimate.

Training options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--stress` | All four components | One or more components: `uu`, `vv`, `ww`, `uv`, or `all`. |
| `--epochs` | `10000` | Number of training epochs. |
| `--lr` | `0.005` | Learning rate. |
| `--optimizer` | `adam` | Optimizer: `adam` or `sgd`. |
| `--seed` | `45` | Random seed for the training/test split. |

Aliases: `--stresses` and `--stress-choice` for `--stress`; `--max-epochs`, `--max_epchs` and `--max_epoch` for `--epochs`; `--learning-rate` for `--lr`.

An epoch is one training pass; the learning rate controls the size of the weight updates.

For each component, the script saves these files in `nn/` (shown for `uu`):

- `model-pressure-strain-uu.pth`: trained model.
- `scalers-X-pressure-strain-uu.bin`: input scalers.
- `scaler-Phi-pressure-strain-uu.bin`: output scaler.
- `features-pressure-strain-uu.txt`: input feature names.

Plots go into `figs/`, including full-profile, near-wall and loss plots. Re-running a component overwrites its saved files.

### RANS verification

After training the `uu` PINN and pressure-strain NN above, run both pressure-strain modes:

```bash
python RANS0_uu.py
python RANS0_uu.py --no-nn
```

The default uses NN pressure-strain (`--nn`); both modes use `nn/checkpoint_pinn_uu_trained.ct`. NN mode also requires the saved `uu` pressure-strain model and both scaler files. This script solves only `uu` and does not train either network.

Optional solver controls are `--cells` (default `70`), `--max-iterations` or `--maxit` (`40000`), `--relaxation` or `--urf` (`0.5`), and `--tolerance` (`1e-6`). Results are saved separately in `nn/rans0_uu_nn.npz` and `nn/rans0_uu_standard.npz`; the script reports equation convergence and the relative error against DNS.

Comparison plots require both result files. On the first run, NN mode saves its result before plotting reports the missing standard result; run the second command to create it, then regenerate NN plots using plot-only mode below.

### Plot only

Use `--plot-only` to regenerate plots without training. The PINN and NN scripts load saved models and accept `--stress` to select one, several, or all components. The RANS script reads saved solver results. Plots are saved in `figs/`.

#### PINN plots

```bash
python compute-vist-using-PINN-load.py --stress all --plot-only
```

For each component, this loads `nn/checkpoint_pinn_<stress>_trained.ct` when available, otherwise `nn/checkpoint_pinn_<stress>.ct`. A missing checkpoint causes loading to fail. Plot-only mode regenerates diagnostics and figures without saving new checkpoints or profile text files. The loss plot is generated only when the checkpoint contains a nonempty loss history.

#### NN plots

Load saved models and scalers, print validation metrics, and regenerate comparison plots without training:

```bash
python compute-pr-strain-load.py --stress all --plot-only
```

#### RANS verification plots

After running both verification modes:

```bash
python RANS0_uu.py --plot-only
python RANS0_uu.py --no-nn --plot-only
```

These commands read the saved `.npz` results without loading models or rerunning the solver, and regenerate the full-profile, near-wall and convergence plots in `figs/`. Both result files must exist; solver controls only apply to a new solve.

## Credits and sources

The base code is credited to **Lars Davidson, Chalmers University of Technology**. This project adapts it for the Reynolds stress diffusion and pressure-strain tasks. His [IMS135 course website](https://www.cfd-sweden.se/lada/ML-IMS135/index.html) provides the original PINN scripts, related Python examples and assignment material.

Physics and assignment reference: Lars Davidson, *Fluid mechanics, turbulent flow and turbulence modeling*, Appendix AE.2 and AE.2.1, in the supplied course PDF. The course website links to the eBook.

The DNS data are by **Myoungkyu Lee and Robert D. Moser, University of Texas at Austin**. The [channel-flow DNS dataset](https://turbulence.oden.utexas.edu/channel2015/content/Data_2015_5200.html) provides the profiles used here at $Re = 5200$. The local headers list the older `turbulence.ices.utexas.edu` address; the linked site uses `turbulence.oden.utexas.edu`.

As requested in the DNS headers, cite the data paper when publishing results:

Myoungkyu Lee and Robert D. Moser (2015), Journal of Fluid Mechanics, **774**, 395-415. [Paper](https://doi.org/10.1017/jfm.2015.268).
# pinn-optimisation-viscosity
