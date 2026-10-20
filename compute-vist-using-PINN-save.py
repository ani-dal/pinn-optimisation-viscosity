import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import os
import sys
import argparse

#friction viscosity 
#Re = 5185.897 #Re given in tbe DNS files
Re = 5200
viscous = 1/Re

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
        description="Initialize and save PINN checkpoint weights for Reynolds stress turbulent viscosity."
    )
    parser.add_argument(
        '--stress', '--stresses', '--stress-choice',
        dest='stress',
        nargs='+',
        default=['uv'],
        help="Stress choice(s): 'uu', 'vv', 'ww', 'uv', or 'all' (e.g. --stress choice 'uv' or --stress all)"
    )
    parser.add_argument(
        '--max-epochs', '--epochs', '--max_epchs', '--init-epochs',
        dest='max_epochs',
        type=int,
        default=1,
        help="Number of initialization epochs before saving checkpoint (default: 1)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=5,
        help="Random seed for weight initialization (default: 5)"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help="Learning rate for initialization step (default: 0.01)"
    )
    return parser.parse_args()

args = parse_arguments()
stresses = parse_stress_args()
init_epochs = args.max_epochs

print(f"Initialising and saving check point")
print(f"stress to initialize: {stresses}")

#load DNS data
DNS_mean = np.genfromtxt("dns/LM_Channel_5200_mean_prof.dat", comments="%")
y_DNS = DNS_mean[:, 0]
u_DNS = DNS_mean[:, 2]
dudy_DNS = np.gradient(u_DNS, y_DNS)

DNS_stress = np.genfromtxt("dns/LM_Channel_5200_vel_fluc_prof.dat", comments="%")
u2_DNS = DNS_stress[:, 2]
v2_DNS = DNS_stress[:, 3]
w2_DNS = DNS_stress[:, 4]
uv_DNS = DNS_stress[:, 5]

# Load computational grid
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
    print(f"\nInitializing and saving PINN for {stress_choice}")
    DNS_phi_terms = np.genfromtxt(file_map[stress_choice], comments="%")

    phi_DNS = stress_dict[stress_choice]

    #DNS sources need to divided by nut for conversion from length scale to wall scale
    P_phi = DNS_phi_terms[:, 2] / viscos
    Phi_phi = DNS_phi_terms[:, 5] / viscos
    eps_phi = DNS_phi_terms[:, 7] / viscos

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

    viscos_lam = torch.ones_like(y_t)*viscos

    #boundary conditions nut = 0 @ y=0 wall
    vist_0 = 0
    vist_1 = float(np.interp(y_kom[-1], y_DNS[:-1], nu_t_DNS[:-1]))

    #initialize network
    torch.manual_seed(args.seed)
    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def PDE(y, vist_pred):
        dvist_dy = deriv(vist_pred, y)
        total_diffusion = (vist_pred + viscos_lam) * d2phidy2_t + dphidy_t * dvist_dy
        imbalance = total_diffusion + (P_phi_t + Phi_phi_t - eps_phi_t)
        loss_de = torch.sum(imbalance ** 2)
        loss_bc = (vist_pred[0] - vist_0) ** 2 + (vist_pred[-1] - vist_1) ** 2
        return loss_de, loss_bc

    for ep in range(init_epochs):
        optimizer.zero_grad()
        outputs = model(y_t)
        loss_de, loss_bc = PDE(y_t, outputs)
        loss = loss_de + 1000.0 * loss_bc
        loss.backward()
        optimizer.step()

    checkpoint = {
        'epoch': init_epochs,
        'stress': stress_choice,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }

    ckpt_filename = f'nn/checkpoint_pinn_{stress_choice}.ct'
    torch.save(checkpoint, ckpt_filename)
    print(f"Saved initialization checkpoint to '{ckpt_filename}' (Loss: {loss.item():.4e})")