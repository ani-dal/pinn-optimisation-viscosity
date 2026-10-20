import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import os
import sys
import argparse
import copy

#friction viscosity 
#Re = 5185.897 #Re given in tbe DNS files
Re = 5200
viscos = 1/Re

# Plot saved models without training.
plotOnly = False

os.makedirs('figs', exist_ok=True)
os.makedirs('nn', exist_ok=True)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
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
        x = torch.sigmoid(self.layer_5(x))
        return x

#for derivatives
def deriv(y,x):
    return grad(y, x, torch.ones(x.size()[0], 1), create_graph=True)

def parse_stress_args(stress_args):
    """
    Parse stress arguments flexibly, handling inputs like:
    --stress 'uv'
    --stress choice 'uv'
    --stress uu vv ww uv
    --stress all
    """
    valid_stresses = ['uu', 'vv', 'ww', 'uv']
    cleaned = []
    for s in stress_args:
        # Strip extraneous words, quotes or punctuation
        s_clean = s.strip().lower().replace("'", "").replace('"', "").replace(',', '')
        if s_clean in ['choice', 'stresses', 'stress', '']:
            continue
        if s_clean == 'all':
            return valid_stresses
        if s_clean in valid_stresses:
            if s_clean not in cleaned:
                cleaned.append(s_clean)
        else:
            print(f"Warning: unrecognized stress '{s}', skipping.")
    return cleaned if cleaned else ['uv']

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Load checkpoint and train PINN to predict turbulent diffusion and turbulent viscosity for Reynolds stress transport."
    )
    parser.add_argument(
        '--stress', '--stresses', '--stress-choice',
        dest='stress',
        nargs='+',
        default=['uv'],
        help="Stress choice(s): 'uu', 'vv', 'ww', 'uv', or 'all' (e.g. --stress choice 'uv' or --stress all)"
    )
    parser.add_argument(
        '--max-epochs', '--epochs', '--max_epchs', '--max_epoch',
        dest='max_epochs',
        type=int,
        default=10000,
        help="Maximum number of training epochs (default: 10000)"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        '--bc-weight',
        type=float,
        default=1000.0,
        help="Weight for boundary condition loss (default: 1000.0)"
    )
    parser.add_argument("--plot-only", dest="plotOnly", action="store_true", help="Plot saved models without training")
    return parser.parse_args()

args = parse_arguments()
plotOnly = args.plotOnly
stresses = parse_stress_args(args.stress)
max_epochs = args.max_epochs

print("PINN Turbulent Diffusion & Viscosity Prediction")
print(f" Stress          : {stresses}")
print(f" Max epochs      : {max_epochs}")
print(f" Learning rate   : {args.lr}")

#load DNS data
DNS_mean = np.genfromtxt("dns/LM_Channel_5200_mean_prof.dat", comments="%")
y_DNS = DNS_mean[:, 0]
yplus_DNS = DNS_mean[:, 1]
u_DNS = DNS_mean[:, 2]
dudy_DNS = np.gradient(u_DNS, y_DNS)

DNS_stress = np.genfromtxt("dns/LM_Channel_5200_vel_fluc_prof.dat", comments="%")
u2_DNS = DNS_stress[:, 2]
v2_DNS = DNS_stress[:, 3]
w2_DNS = DNS_stress[:, 4]
uv_DNS = DNS_stress[:, 5]

#load computational grid
kom_data = np.loadtxt('dns/y_u_k_om_uv_5200-RANS-half-channel.txt')
y_kom = kom_data[:, 0]

file_map = {
    'uu': 'dns/LM_Channel_5200_RSTE_uu_prof.dat',
    'vv': 'dns/LM_Channel_5200_RSTE_vv_prof.dat',
    'ww': 'dns/LM_Channel_5200_RSTE_ww_prof.dat',
    'uv': 'dns/LM_Channel_5200_RSTE_uv_prof.dat'
}

stress_dict = {
    'uu': u2_DNS,
    'vv': v2_DNS,
    'ww': w2_DNS,
    'uv': uv_DNS
}

nu_t_DNS = np.abs(uv_DNS / np.maximum(np.abs(dudy_DNS), 1e-7)) #avoid null division

for stress_choice in stresses:
    print(f"\nLoading and Training PINN for {stress_choice}")

    DNS_phi_terms = np.genfromtxt(file_map[stress_choice], comments="%")

    diff_turb_DNS = DNS_phi_terms[:, 3] / viscos
    diff_visc_DNS = DNS_phi_terms[:, 4] / viscos
    diff_p_DNS = DNS_phi_terms[:, 6] / viscos

    #turbulent diffusion target = velocity transport + pressure transport
    diff_turb_DNS = diff_turb_DNS + diff_p_DNS
    #full total DNS diffusion
    diff_total_DNS = diff_turb_DNS + diff_visc_DNS

    phi_DNS = stress_dict[stress_choice]

    P_phi = DNS_phi_terms[:, 2] / viscos
    Phi_phi = DNS_phi_terms[:, 5] / viscos
    eps_phi = DNS_phi_terms[:, 7] / viscos

    dphidy_DNS = np.gradient(phi_DNS, y_DNS, edge_order=2)
    d2phidy2_DNS = np.gradient(dphidy_DNS, y_DNS, edge_order=2)

    #Re-sig derivaives
    dphidy_DNS = np.gradient(phi_DNS, y_DNS, edge_order=2)
    d2phidy2_DNS = np.gradient(dphidy_DNS, y_DNS, edge_order=2)

    #interpolate to grid
    dphidy_t = torch.tensor(np.interp(y_kom, y_DNS, dphidy_DNS), dtype=torch.float32).view(-1,1)
    d2phidy2_t = torch.tensor(np.interp(y_kom, y_DNS, d2phidy2_DNS), dtype=torch.float32).view(-1,1)
    P_phi_t = torch.tensor(np.interp(y_kom, y_DNS, P_phi), dtype=torch.float32).view(-1,1)
    eps_phi_t = torch.tensor(np.interp(y_kom, y_DNS, eps_phi), dtype=torch.float32).view(-1,1)
    Phi_phi_t = torch.tensor(np.interp(y_kom, y_DNS, Phi_phi), dtype=torch.float32).view(-1,1)
    y_t  = torch.tensor(y_kom, requires_grad=True, dtype=torch.float32).view(-1,1)

    diff_turb_DNS_t = torch.tensor(np.interp(y_kom, y_DNS, diff_turb_DNS), dtype=torch.float32).view(-1, 1)
    diff_visc_DNS_t = torch.tensor(np.interp(y_kom, y_DNS, diff_visc_DNS), dtype=torch.float32).view(-1, 1)
    diff_total_DNS_t = torch.tensor(np.interp(y_kom, y_DNS, diff_total_DNS), dtype=torch.float32).view(-1, 1)

    viscos_lam = torch.ones_like(y_t)*viscos

    #boundary conditions nut = 0 @ y=0 wall
    vist_0 = 0
    vist_1 = float(np.interp(y_kom[-1], y_DNS[:-1], nu_t_DNS[:-1]))

    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #dynamic learning rate decay milestones based on max_epochs
    m1 = int(0.30 * max_epochs)
    m2 = int(0.60 * max_epochs)
    m3 = int(0.85 * max_epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2, m3], gamma=0.5)

    checkpoint_file = f'nn/checkpoint_pinn_{stress_choice}.ct'

    def PDE(y, vist_pred):
        dvist_dy = deriv(vist_pred, y)[0]

        total_diffusion = (vist_pred + viscos_lam) * d2phidy2_t + dphidy_t * dvist_dy
        
        imbalance = total_diffusion + (P_phi_t + Phi_phi_t - eps_phi_t)
        loss_de = torch.sum(imbalance ** 2)
        loss_bc = (vist_pred[0] - vist_0) ** 2 + (vist_pred[-1] - vist_1) ** 2
        return loss_de, loss_bc, imbalance

    loss_history = []
    best_loss = 1e30
    best_model_state = copy.deepcopy(model.state_dict())

    # Load saved weights for plotting, otherwise train.
    if plotOnly:
        trained_file = f'nn/checkpoint_pinn_{stress_choice}_trained.ct'
        checkpoint_file = trained_file if os.path.exists(trained_file) else checkpoint_file
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('nan')))
        loss_history = checkpoint.get('loss_history', [])
        print(f"Loaded checkpoint '{checkpoint_file}'")
    else:
        print_step = max(1, max_epochs // 10)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            vist_pred = model(y_t)
            loss_de, loss_bc, imbalance = PDE(y_t, vist_pred)
            loss = loss_de + args.bc_weight * loss_bc
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = copy.deepcopy(model.state_dict())

        
            if epoch % print_step == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch:6d}/{max_epochs} | Loss: {current_loss:.2e} (DE: {loss_de.item():.2e}, BC: {loss_bc.item():.2e}) | Best: {best_loss:.2e}")

        model.load_state_dict(best_model_state)
        print(f"Restored best PINN model weights with loss: {best_loss:.2e}")

    #evaluate final physics predictions
    vist_final = model(y_t)
    dvist_dy_final = deriv(vist_final, y_t)[0]

    # predicted turbulent diffusion
    diff_turb_pred = vist_final * d2phidy2_t + dphidy_t * dvist_dy_final

    #predicted molecular viscous diffusion
    diff_visc_pred = viscos_lam * d2phidy2_t

    #predicted total diffusion
    diff_total_pred = (vist_final + viscos_lam) * d2phidy2_t + dphidy_t * dvist_dy_final

    #final PDE imbalance
    _, _, final_imbalance = PDE(y_t, vist_final)
    residual_plot = final_imbalance.detach().numpy().flatten()

    #save predicted turbulent diffusion and components
    if not plotOnly:
        output_txt = f'nn/predicted_turb_diffusion_{stress_choice}.txt'
        save_arr = np.c_[
            y_kom,
            diff_turb_pred.detach().numpy().flatten(),
            diff_total_pred.detach().numpy().flatten(),
            diff_visc_pred.detach().numpy().flatten(),
            vist_final.detach().numpy().flatten()
        ]

        header_str = "1:y_kom  2:predicted_turb_diffusion  3:predicted_total_diffusion  4:predicted_visc_diffusion  5:predicted_nu_t"
        np.savetxt(output_txt, save_arr, header=header_str, fmt="%.8e")
        print(f"Saved predicted diffusion and viscosity profiles to '{output_txt}'")

        #save updated checkpoint
        trained_checkpoint = {
            'epoch': max_epochs,
            'stress': stress_choice,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'loss_history': loss_history,
        }
        torch.save(trained_checkpoint, f'nn/checkpoint_pinn_{stress_choice}_trained.ct')

    #validation error: L^2
    turb_diff_rmse = np.sqrt(np.mean((diff_turb_pred.detach().numpy() - diff_turb_DNS_t.detach().numpy()) ** 2))
    turb_diff_dns_rms = np.sqrt(np.mean(diff_turb_DNS_t.detach().numpy() ** 2))
    turb_diff_rel_err = (turb_diff_rmse / np.maximum(turb_diff_dns_rms, 1e-6)) * 100

    total_diff_rmse = np.sqrt(np.mean((diff_total_pred.detach().numpy() - diff_total_DNS_t.detach().numpy()) ** 2))
    total_diff_dns_rms = np.sqrt(np.mean(diff_total_DNS_t.detach().numpy()**2))
    total_diff_rel_err = (total_diff_rmse / np.maximum(total_diff_dns_rms, 1e-6)) * 100

    print(f"\nValidation Summary for {stress_choice}")
    print(f"Turbulent Diffusion RMSE    : {turb_diff_rmse:.4e} (Rel Err: {turb_diff_rel_err:.2f}%)")
    print(f"Total Diffusion RMSE        : {total_diff_rmse:.4e} (Rel Err: {total_diff_rel_err:.2f}%)")
    print(f"PDE Residual RMS            : {np.sqrt(np.mean(residual_plot**2)):.4e}")
    print(f"PDE Residual Max            : {np.max(np.abs(residual_plot)):.4e}")
    print(f"Wall BC nu_t(0)             : {vist_final[0].item():.4e} (Target: 0.0)")
    print(f"Outer BC nu_t(ymax)         : {vist_final[-1].item():.4e} (Target: {vist_1:.4e})")

    #plots
    yplus_plot = np.interp(y_kom, y_DNS, yplus_DNS)
    vist_DNS_plot = np.interp(y_kom, y_DNS, nu_t_DNS) / viscos

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(y_kom / viscos, diff_turb_DNS_t.detach().numpy(), 'r-', linewidth=2, label=r"DNS")
    plt.plot(y_kom / viscos, diff_turb_pred.detach().numpy(), 'b--', linewidth=2, label=r"PINN")
    ax.legend(loc='best')
    plt.xlabel('$y^+$')
    plt.ylabel(rf'$D_{{\mathrm{{{stress_choice}}},t}}$', fontsize=14)
    plt.title(f'Turbulent Diffusion Comparision ({stress_choice})', fontsize=14)
    plt.xlim(0, 100)
    plt.grid(True)
    plt.savefig(f'figs/diffusion-PINN-5200-predicted-{stress_choice}.png', bbox_inches='tight')
    plt.close()

    #Fig 1: PINN turbulent viscosity vs DNS nut/nu
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_plot, vist_final.detach().numpy() / viscos, 'b-', linewidth=2, label=r'PINN $\nu_{t,\mathrm{PINN}}/\nu$')
    plt.plot(yplus_plot, vist_DNS_plot, 'k--', linewidth=2, label=r'DNS $\nu_t/\nu$')
    plt.xlabel(r'$y^+$', fontsize=14)
    plt.ylabel(r'$\nu_t/\nu$', fontsize=14)
    plt.title(f'PINN Turbulent Viscosity ({stress_choice})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig(f'figs/fig1_viscosity_{stress_choice}.png', bbox_inches='tight')
    plt.close()

    #Fig 2: Turbulent Diffusion Comparison (PINN vs DNS)
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_plot, diff_turb_DNS_t.detach().numpy(), 'k-', linewidth=2, label=r'$D_{t,DNS}$')
    plt.plot(yplus_plot, diff_turb_pred.detach().numpy(), 'r--', linewidth=2, label=r'$D_{t,PINN}$')
    plt.xlabel(r'$y^+$', fontsize=14)
    plt.ylabel(rf'$D_{{\mathrm{{{stress_choice}}},t}}$',fontsize=14)
    plt.title(f'Turbulent Diffusion Comparison ({stress_choice})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig(f'figs/fig2_turb_diffusion_{stress_choice}.png', bbox_inches='tight')
    plt.close()

    #Fig 3: Consistent Total Diffusion Comparison (PINN vs DNS)
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_plot, diff_total_DNS_t.detach().numpy(), 'k-', linewidth=2, label=r'DNS ($D_{\nu} + D_t$)')
    plt.plot(yplus_plot, diff_total_pred.detach().numpy(), 'm--', linewidth=2, label=r'PINN ($D_{\nu} + D_t$)')
    plt.xlabel(r'$y^+$', fontsize=14)
    plt.ylabel(fr'$D_{{\mathrm{{total}},{stress_choice}}}$', fontsize=14)
    plt.title(f'Total Diffusion Comparison ({stress_choice})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig(f'figs/fig3_total_diffusion_{stress_choice}.png', bbox_inches='tight')
    plt.close()

    # Fig 4: Diffusion Components Breakdown
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_plot, diff_turb_DNS_t.detach().numpy(), 'k-', linewidth=2, label=r'$D_{t,\mathrm{DNS}}$')
    plt.plot(yplus_plot, diff_visc_DNS_t.detach().numpy(), 'g-', linewidth=1.5, label=r'$D_{\nu,\mathrm{DNS}}$')
    plt.plot(yplus_plot, diff_turb_pred.detach().numpy(), 'r--', linewidth=2, label=r'$D_{t,\mathrm{PINN}}$')
    plt.plot(yplus_plot, diff_visc_pred.detach().numpy(), 'm--', linewidth=1.5, label=r'$D_{\nu,\mathrm{PINN}}$')
    plt.xlabel(r'$y^+$', fontsize=14)
    plt.ylabel('Diffusion terms', fontsize=14)
    plt.title(f'Diffusion Components Breakdown ({stress_choice})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig(f'figs/fig4_diffusion_breakdown_{stress_choice}.png', bbox_inches='tight')
    plt.close()

    # Separate molecular diffusion comparison.
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_plot, diff_visc_DNS_t.detach().numpy(), 'k-', linewidth=2,
             label=r'DNS viscous diffusion ($D_\nu$)')
    plt.plot(yplus_plot, diff_visc_pred.detach().numpy(), 'm--', linewidth=2,
             label=r'PINN viscous diffusion ($D_\nu$)')
    plt.axhline(0.0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel(r'$y^+$', fontsize=14)
    plt.ylabel('Viscous diffusion', fontsize=14)
    plt.title(f'Viscous Diffusion Comparison ({stress_choice})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig(f'figs/fig5_viscous_diffusion_{stress_choice}.png', bbox_inches='tight')
    plt.close()

   
    pressure_transport_plot = np.interp(y_kom, y_DNS, diff_p_DNS)
    turbulent_transport_plot = np.interp(y_kom, y_DNS, DNS_phi_terms[:, 3] / viscos)
    turbulent_transport_inferred = diff_turb_pred.detach().numpy().flatten() - pressure_transport_plot

    plt.figure(figsize=(8, 6))
    plt.plot(yplus_plot, turbulent_transport_plot, 'k-', linewidth=2,
             label=r'DNS $D_t$')
    plt.plot(yplus_plot, turbulent_transport_inferred, 'r--', linewidth=2,
             label=r'PINN $D_t$')
    plt.axhline(0.0, color='gray', linestyle=':', linewidth=1)
    plt.xlabel(r'$y^+$', fontsize=14)
    plt.ylabel(rf'$D_{{\mathrm{{{stress_choice}}},t}}$',fontsize=14)
    plt.title(f'Turbulent Diffusion Comparison ({stress_choice})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig(f'figs/fig6_pure_turbulent_transport_{stress_choice}.png', bbox_inches='tight')
    plt.close()

    # Fig 5: PDE Residual
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_plot, residual_plot, 'g-', linewidth=2, label='PDE Residual Q')
    plt.axhline(0.0, color='k', linestyle='--', linewidth=1.5)
    plt.xlabel(r'$y^+$', fontsize=14)
    plt.ylabel('PDE residual', fontsize=14)
    plt.title(f'PDE Residual of Stress Transport ({stress_choice})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.savefig(f'figs/fig7_residual_{stress_choice}.png', bbox_inches='tight')
    plt.close()

    # Fig 6: Training Loss Convergence
    if loss_history:
        plt.figure(figsize=(8, 6))
        plt.semilogy(range(len(loss_history)), loss_history, 'b-', linewidth=2, label='Total Loss')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f'Training Loss Convergence ({stress_choice})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.savefig(f'figs/fig8_loss_{stress_choice}.png', bbox_inches='tight')
        plt.close()

print("\n Plotting Completed" if plotOnly else "\n Training Completed")

    
