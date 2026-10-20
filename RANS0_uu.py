import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse import linalg
from joblib import load
import matplotlib.pyplot as plt


#same pressure-strain network as task 2
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll1 = nn.Linear(4, 10)
        self.tanh = nn.Tanh()
        self.ll2 = nn.Linear(10, 10)
        self.ll3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        out = self.tanh(self.ll1(x))
        out = self.tanh(self.ll2(out))
        out = self.tanh(self.ll3(out))
        return self.output(out)


#same viscosity network as task 1
class PINNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(1, 30)
        self.layer_2 = nn.Linear(30, 30)
        self.layer_3 = nn.Linear(30, 30)
        self.layer_4 = nn.Linear(30, 30)
        self.layer_5 = nn.Linear(30, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer_1(x))
        x = torch.sigmoid(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))
        x = torch.sigmoid(self.layer_4(x))
        return torch.sigmoid(self.layer_5(x))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Solve the uu equation using DNS and saved models.")
    parser.add_argument('--nn', action=argparse.BooleanOptionalAction, default=True,
                        help="Use NN pressure-strain; --no-nn uses the AE.8 model")
    parser.add_argument('--max-iterations', '--maxit', type=int, default=40000)
    parser.add_argument('--relaxation', '--urf', type=float, default=0.5)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--cells', type=int, default=70)
    parser.add_argument('--plot-only', dest='plotOnly', action='store_true',
                        help="Plot saved results without solving or loading models")
    args = parser.parse_args()
    return args


args = parse_arguments()
NN_bool = args.nn
plotOnly = args.plotOnly
mode = 'nn' if NN_bool else 'standard'
result_file = f'nn/rans0_uu_{mode}.npz'
plt.rcParams.update({'font.size': 14})
plt.interactive(False)
os.makedirs('figs', exist_ok=True)
os.makedirs('nn', exist_ok=True)

if plotOnly:
    with np.load(result_file) as saved:
        yp, yplus, uu = saved['yp'], saved['yplus'], saved['uu']
        yplus_DNS, u2_DNS = saved['yplus_DNS'], saved['u2_DNS']
        Phi, Phi_DNS = saved['Phi'], saved['Phi_DNS']
        vist, diffusion, diffusion_DNS = saved['vist'], saved['diffusion'], saved['diffusion_DNS']
        history, viscos = saved['history'], float(saved['viscos'])
        residual = float(saved['residual'])
        converged = bool(saved['converged'])
else:
    Re = 5200
    c1, c2, c1w, c2w = 1.8, 0.52, 0.5, 0.3

    viscos = 1 / Re
    nj = args.cells
    maxit = args.max_iterations
    urf = args.relaxation

    yfac = 1.1 #stretching
    yc = np.r_[0.0, np.cumsum(yfac ** np.arange(nj))]
    yc = yc / yc[-1]
    yp = 0.5 * (yc[:-1] + yc[1:])
    delta_y = np.diff(yc)
    yplus = yp / viscos
    
    DNS_mean = np.genfromtxt('dns/LM_Channel_5200_mean_prof.dat', comments='%')
    DNS_stress = np.genfromtxt('dns/LM_Channel_5200_vel_fluc_prof.dat', comments='%')
    DNS_k = np.genfromtxt('dns/LM_Channel_5200_RSTE_k_prof.dat', comments='%')
    DNS_uu = np.genfromtxt('dns/LM_Channel_5200_RSTE_uu_prof.dat', comments='%')
    yplus_DNS = DNS_stress[:, 0] / viscos
    u2_DNS = DNS_stress[:, 2]
    uv = np.interp(yp, DNS_stress[:, 0], DNS_stress[:, 5])
    vv = np.interp(yp, DNS_stress[:, 0], DNS_stress[:, 3])
    ww = np.interp(yp, DNS_stress[:, 0], DNS_stress[:, 4])
    dudy_DNS = np.gradient(DNS_mean[:, 2], DNS_mean[:, 0], edge_order=2)
    dudy = np.interp(yp, DNS_mean[:, 0], dudy_DNS)

    #wall-unit budgets to outer units used in the equation
    eps = np.interp(yp, DNS_k[:, 0], DNS_k[:, 7]) / viscos
    Phi_DNS = np.interp(yp, DNS_uu[:, 0], DNS_uu[:, 5]) / viscos
    diffusion_DNS = np.interp(yp, DNS_uu[:, 0], DNS_uu[:, 3] + DNS_uu[:, 4]
                              + DNS_uu[:, 6]) / viscos

    Pk = -uv * dudy
    P11 = 2 * Pk

    #load uu diffusion PINN in both pressure-strain modes
    checkpoint = torch.load('nn/checkpoint_pinn_uu_trained.ct', map_location='cpu', weights_only=False)
    model = PINNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        vist = model(torch.tensor(yp[:, None], dtype=torch.float32)).numpy().flatten()
    if not np.isfinite(vist).all() or np.any(vist < 0):
        raise ValueError('The uu PINN returned invalid viscosity')
    print(f'Pressure-strain: {mode}; diffusion: uu PINN; DNS U, uv and epsilon fixed')

    if NN_bool:
        NN = torch.load('nn/model-pressure-strain-uu.pth', map_location='cpu', weights_only=False)
        NN.eval()
        scalers = load('nn/scalers-X-pressure-strain-uu.bin')
        scaler_Y = load('nn/scaler-Phi-pressure-strain-uu.bin')

    #diffusion coefficients as in RANS0
    fy = (yc[1:-1] - yp[:-1]) / np.diff(yp)
    vist_face = fy * vist[1:] + (1 - fy) * vist[:-1]
    an = np.r_[(viscos + vist_face) / np.diff(yp), 0.0]
    as1 = np.r_[0.0, an[:-1]]
    ap_diff = an + as1
    ap_diff[0] += viscos / yp[0]
    #uu=0 at the wall; zero flux at the centre
    A_diff = sparse.diags([ap_diff, -an[:-1], -as1[1:]], [0, 1, -1], format='csr')

    uu = 2 * (1 - np.exp(-yplus / 30))
    history = []
    converged = False

    for n in range(maxit + 1):
        #uu is unknown; vv and ww close k using DNS
        k = np.maximum(0.5 * (uu + vv + ww), 1e-10)
        if NN_bool:
            inputs = [uu, k, Pk, P11]
            X = np.column_stack([scalers[i].transform(inputs[i][:, None])[:, 0]
                                 for i in range(4)])
            X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            preds = NN(X_t)
            Phi = scaler_Y.inverse_transform(preds.detach().numpy()).flatten()
            #implicit NN slope includes k changing with uu
            slopes = torch.autograd.grad(preds.sum(), X_t)[0].detach().numpy()
            dPhi = (slopes[:, 0] * scalers[0].scale_[0]
                    + 0.5 * slopes[:, 1] * scalers[1].scale_[0]) / scaler_Y.scale_[0]
        else:
            #four pressure-strain terms in AE.8
            f = k ** 1.5 / (2.55 * yp * eps)
            Phi22_2 = c2 * 2 / 3 * Pk
            Phi = (-c1 * eps / k * (uu - 2 / 3 * k)
                   - (c2 + 8) / 11 * (P11 - 2 / 3 * Pk)
                   + c1w * eps / k * vv * f + c2w * Phi22_2 * f)
            dPhi = (-c1 * eps * (1 / k - 0.5 * uu / k ** 2)
                    + c1w * vv * np.sqrt(k) / (2.55 * yp) / (4 * k)
                    + c2w * Phi22_2 * f * 0.75 / k)

        #scalar epsilon gives the AE.8 sink, not DNS epsilon_11
        source = P11 + Phi - 2 / 3 * eps
        residual = np.linalg.norm(A_diff @ uu - source * delta_y)
        residual /= max(np.linalg.norm(P11 * delta_y), 1e-12)
        history.append(residual)
        #stopping criteria
        if n % 1000 == 0 or residual < args.tolerance:
            print(f'iteration {n:5d}: relative equation residual = {residual:.3e}')
        if residual < args.tolerance:
            converged = True
            break
        if n == maxit:
            break

        #negative source slope goes into the diagonal
        sp = np.minimum(dPhi, 0.0) * delta_y
        ap = (ap_diff - sp) / urf
        su = source * delta_y - sp * uu + (1 - urf) * ap * uu
        A = sparse.diags([ap, -an[:-1], -as1[1:]], [0, 1, -1], format='csr')
        uu = linalg.spsolve(A, su)
        

    history = np.asarray(history)
    diffusion = -(A_diff @ uu) / delta_y
    if NN_bool:
        counts = np.count_nonzero((X < -1e-6) | (X > 1 + 1e-6), axis=0)
        print(f'NN inputs outside training range [uu, k, Pk, P11]: {counts}')
    if not converged:
        print('Maximum iterations reached; saved profiles are unconverged')
    if np.any(uu < 0):
        print(f'Negative uu at {np.count_nonzero(uu < 0)} cells; minimum = {uu.min():.3e}')
    np.savez(result_file, yp=yp, yplus=yplus, uu=uu, yplus_DNS=yplus_DNS,
             u2_DNS=u2_DNS, Phi=Phi, Phi_DNS=Phi_DNS, vist=vist,
             diffusion=diffusion, diffusion_DNS=diffusion_DNS, history=history,
             viscos=viscos, residual=residual, converged=converged)
    print(f'Saved {result_file}')

uu_DNS = np.interp(yplus, yplus_DNS, u2_DNS)
relative_error = 100 * np.linalg.norm(uu - uu_DNS) / np.linalg.norm(uu_DNS)
print(f'Converged: {converged}; residual: {residual:.3e}; uu relative L2 error: {relative_error:.2f}%')
label = 'SM + NN' if NN_bool else 'SM'
if not converged:
    label += ' (unconverged)'

#load both profiles for the uu comparison
with np.load('nn/rans0_uu_nn.npz') as saved:
    yplus_pinn, uu_pinn = saved['yplus'], saved['uu']
with np.load('nn/rans0_uu_standard.npz') as saved:
    yplus_sm, uu_sm = saved['yplus'], saved['uu']

#full profile
plt.figure(figsize=(8, 6))
plt.plot(yplus_DNS, u2_DNS, 'k-', linewidth=2, label='DNS')
plt.plot(yplus_sm, uu_sm, 'r:', linewidth=2, label='SM')
plt.plot(yplus_pinn, uu_pinn, 'b-', linewidth=2, label='PINN model')
plt.xlabel(r'$y^+$')
plt.ylabel(r'$\overline{u\prime u\prime}^{+}$')
plt.legend()
plt.grid(True)
plt.savefig(f'figs/fig12_uu-vs-yplus-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()

#near wall profile
plt.figure(figsize=(8, 6))
plt.plot(yplus_DNS, u2_DNS, 'k-', linewidth=2, label='DNS')
plt.plot(yplus_sm, uu_sm, 'r:', linewidth=2, label='SM')
plt.plot(yplus_pinn, uu_pinn, 'b-', linewidth=2, label='PINN model')
plt.xlabel(r'$y^+$')
plt.ylabel(r'$\overline{u\prime u\prime}^{+}$')
plt.xlim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f'figs/fig13_uu-vs-yplus-zoom-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()

#solver convergence
plt.figure(figsize=(8, 6))
plt.semilogy(np.arange(len(history)), np.maximum(history, 1e-16), 'b-')
plt.xlabel('Iteration')
plt.ylabel('Relative equation residual')
plt.grid(True)
plt.savefig(f'figs/fig14_uu-convergence-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()
'''
#pressure-strain in wall units
plt.figure(figsize=(8, 6))
plt.plot(yplus, Phi_DNS * viscos, 'b-', linewidth=2, label='DNS')
plt.plot(yplus, Phi * viscos, 'r--', linewidth=2, label=label)
plt.xlabel(r'$y^+$')
plt.ylabel(r'$\Phi_{11}^{+}$')
plt.xlim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f'figs/fig15_uu-pressure-strain-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()

#pinn viscosity in wall units
plt.figure(figsize=(8, 6))
plt.plot(yplus, vist / viscos, 'b-', linewidth=2, label=r'uu PINN $\nu_t/\nu$')
plt.xlabel(r'$y^+$')
plt.ylabel(r'$\nu_t^{+}$')
plt.legend()
plt.grid(True)
plt.savefig(f'figs/fig16_uu-viscosity-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()

#total diffusion in wall units
plt.figure(figsize=(8, 6))
plt.plot(yplus, diffusion_DNS * viscos, 'b-', linewidth=2, label='DNS')
plt.plot(yplus, diffusion * viscos, 'r--', linewidth=2, label=label)
plt.xlabel(r'$y^+$')
plt.ylabel(r'$D_{11}^{+}$')
plt.xlim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f'figs/fig17_uu-total-diffusion-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()
'''
#errors against DNS on each model grid
error_pinn = uu_pinn - np.interp(yplus_pinn, yplus_DNS, u2_DNS)
error_sm = uu_sm - np.interp(yplus_sm, yplus_DNS, u2_DNS)
rmse_pinn = np.sqrt(np.mean(error_pinn ** 2))
rmse_sm = np.sqrt(np.mean(error_sm ** 2))

#full error profile with overall rmse
plt.figure(figsize=(8, 6))
plt.plot(yplus_sm, np.abs(error_sm), 'r:', linewidth=2,
         label=f'SM')
plt.plot(yplus_pinn, np.abs(error_pinn), 'b--', linewidth=2,
         label=f'PINN')
plt.xlabel(r'$y^+$')
plt.ylabel(r'Error $\overline{u\prime u\prime}^{+}$')
plt.title("Absolute Error")
plt.legend()
plt.grid(True)
plt.savefig(f'figs/fig18_uu-error-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()

#near wall error and rmse use only yplus below 100
near_pinn = yplus_pinn < 100
near_sm = yplus_sm < 100
rmse_pinn_zoom = np.sqrt(np.mean(error_pinn[near_pinn] ** 2))
rmse_sm_zoom = np.sqrt(np.mean(error_sm[near_sm] ** 2))
plt.figure(figsize=(8, 6))
plt.plot(yplus_sm[near_sm], np.abs(error_sm[near_sm]), 'r:', linewidth=2,
         label=f'SM')
plt.plot(yplus_pinn[near_pinn], np.abs(error_pinn[near_pinn]), 'b-', linewidth=2,
         label=f'PINN')
plt.xlabel(r'$y^+$')
plt.ylabel(r'Error $\overline{u\prime u\prime}^{+}$')
plt.title("Absolute Error Near Wall")
plt.xlim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f'figs/fig19_uu-error-zoom-{mode}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'RMSE: PINN = {rmse_pinn:.4f}, SM = {rmse_sm:.4f}')
print(f'Near-wall RMSE (y+ < 100): PINN = {rmse_pinn_zoom:.4f}, SM = {rmse_sm_zoom:.4f}')

