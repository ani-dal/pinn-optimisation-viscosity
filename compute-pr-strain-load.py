import argparse
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

#plot settings
plt.rcParams.update({'font.size': 14})
plt.interactive(False)
os.makedirs('figs', exist_ok=True)
os.makedirs('nn', exist_ok=True)

#friction viscosity
viscos = 1 / 5200

#plot saved models without training.
plotOnly = False

#pressure strain network
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        #4 inputs
        self.ll1 = nn.Linear(4, 10)
        self.tanh = nn.Tanh()
        self.ll2 = nn.Linear(10, 10)
        self.ll3 = nn.Linear(10, 10)
        #1 output
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        out = self.tanh(self.ll1(x))
        out = self.tanh(self.ll2(out))
        out = self.tanh(self.ll3(out))
        out = self.output(out)
        return out

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
    return cleaned if cleaned else valid_stresses

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Neural Network models for Reynolds stress pressure-strain terms (Phi_ij)."
    )
    parser.add_argument(
        '--stress', '--stresses', '--stress-choice',
        dest='stress',
        nargs='+',
        default=['uu', 'vv', 'ww', 'uv'],
        help="Stress choice(s): 'uu', 'vv', 'ww', 'uv', or 'all' (e.g. --stress choice 'uv' or --stress all)"
    )
    parser.add_argument(
        '--max-epochs', '--epochs', '--max_epchs', '--max_epoch',
        dest='max_epochs',
        type=int,
        default=2000,
        help="Number of training epochs (default: 2000)"
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        dest='lr',
        type=float,
        default=0.005,
        help="Learning rate (default: 0.005)"
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'sgd'],
        default='adam',
        help="Optimizer choice: 'adam' (default) or 'sgd'"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=45,
        help="Random seed for train/test split (default: 45)"
    )
    parser.add_argument("--plot-only", dest="plotOnly", action="store_true", help="Plot saved models without training")
    return parser.parse_args()

args = parse_arguments()
plotOnly = args.plotOnly
stresses = parse_stress_args(args.stress)
N_epochs = max(1, args.max_epochs)
learning_rate = args.lr
rand_state = args.seed

print("Pressure Strain NN Training")
print(f" Stresses to train : {stresses}")
print(f" Epochs            : {N_epochs}")
print(f" Learning rate     : {learning_rate}")
print(f" Optimizer         : {args.optimizer}")

#load DNS data
DNS_mean = np.genfromtxt("dns/LM_Channel_5200_mean_prof.dat", comments="%")
y_DNS = DNS_mean[:, 0]
yplus_DNS = DNS_mean[:, 1]
u_DNS = DNS_mean[:, 2]

#Reynolds stresses
DNS_stress = np.genfromtxt("dns/LM_Channel_5200_vel_fluc_prof.dat", comments="%")
u2_DNS = DNS_stress[:, 2]
v2_DNS = DNS_stress[:, 3]
w2_DNS = DNS_stress[:, 4]
uv_DNS = DNS_stress[:, 5]
k_DNS = 0.5 * (u2_DNS + v2_DNS + w2_DNS)

#TKE terms
DNS_k_terms = np.genfromtxt("dns/LM_Channel_5200_RSTE_k_prof.dat", comments="%")
Pk_DNS = DNS_k_terms[:, 2] / viscos

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

for stress_choice in stresses:
    print(f"\nTraining pressure strain NN for {stress_choice}")
    DNS_RSTE = np.genfromtxt(file_map[stress_choice], comments="%")

    stress_DNS = stress_dict[stress_choice]
    P_stress_DNS = DNS_RSTE[:, 2] / viscos
    Phi_DNS = DNS_RSTE[:, 5] / viscos  #pressure strain in outer units

    print(f"DNS Points: {len(stress_DNS)} | Target: Phi_{stress_choice} range [{np.min(Phi_DNS):.2e}, {np.max(Phi_DNS):.2e}]")

    #inputs for each stress
    inputs = [stress_DNS, k_DNS, Pk_DNS, P_stress_DNS]
    feature_names = {
        'uu': ['u2', 'k', 'Pk', 'P11'],
        'vv': ['v2', 'k', 'Pk', 'P22'],
        'ww': ['w2', 'k', 'Pk', 'P33'],
        'uv': ['uv', 'k', 'Pk', 'P12']
    }
    input_names = feature_names[stress_choice]

    #scale pressure strain
    scaler_Y = MinMaxScaler()
    Y = scaler_Y.fit_transform(Phi_DNS.reshape(-1, 1))

    #scale inputs
    X = np.zeros((len(y_DNS), len(inputs)))
    scalers = []
    for i, inp in enumerate(inputs):
        inp_2d = inp.reshape(-1, 1)
        scaler = MinMaxScaler()
        X[:, i] = scaler.fit_transform(inp_2d)[:, 0]
        scalers.append(scaler)

    #split training and test data
    indices = np.arange(len(X))
    X_train, X_test, Y_train, Y_test, index_train, index_test = train_test_split(
        X, Y, indices, test_size=0.2, shuffle=True, random_state=rand_state
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    #initialize model
    torch.manual_seed(rand_state)
    NN = MyNet()
    loss_fn = nn.MSELoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(NN.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(NN.parameters(), lr=learning_rate)

    loss_v = np.zeros(N_epochs)
    best_loss = 1e30
    best_model_weights = copy.deepcopy(NN.state_dict())

    #training
    print_step = max(1, N_epochs // 10)
    t_start = time.time()

    # Load saved weights for plotting, otherwise train.
    if plotOnly:
        NN = torch.load(f'nn/model-pressure-strain-{stress_choice}.pth', map_location='cpu', weights_only=False)
        NN.eval()
        scalers = load(f'nn/scalers-X-pressure-strain-{stress_choice}.bin')
        scaler_Y = load(f'nn/scaler-Phi-pressure-strain-{stress_choice}.bin')
        for i, scaler in enumerate(scalers):
            X[:, i] = scaler.transform(inputs[i].reshape(-1, 1))[:, 0]
        X_test_tensor = torch.tensor(X[index_test], dtype=torch.float32)
        loss_v = []
    else:
        for epoch in range(N_epochs):
            NN.train()
            optimizer.zero_grad()
            train_preds = NN(X_train_tensor)
            loss = loss_fn(train_preds, Y_train_tensor)
            loss.backward()
            optimizer.step()

            NN.eval()
            with torch.no_grad():
                test_preds = NN(X_test_tensor)
                test_loss = loss_fn(test_preds, Y_test_tensor).item()

            loss_v[epoch] = test_loss

            if test_loss < best_loss:
                best_loss = test_loss
                best_model_weights = copy.deepcopy(NN.state_dict())

            if epoch % print_step == 0 or epoch == N_epochs - 1:
                print(f"Epoch {epoch+1:5d}/{N_epochs} | Train Loss: {loss.item():.2e} | Test Loss: {test_loss:.2e} | Best Test: {best_loss:.2e}")

        train_time = time.time() - t_start
        print(f"Training completed in {train_time:.2f}s")

        #load best model weights
        NN.load_state_dict(best_model_weights)
        NN.eval()
        print(f"Restored best model weights with test loss: {best_loss:.2e}")

    #test predictions in physical units
    with torch.no_grad():
        preds_test_scaled = NN(X_test_tensor).numpy()
    Phi_NN_test = scaler_Y.inverse_transform(preds_test_scaled)
    Phi_test_target = Phi_DNS[index_test].reshape(-1, 1)

    test_rmse = np.sqrt(np.mean((Phi_NN_test - Phi_test_target) ** 2))
    test_rms_ref = np.sqrt(np.mean(Phi_test_target ** 2))
    test_rel_err = (test_rmse / np.maximum(test_rms_ref, 1e-6)) * 100
    test_max_err = np.max(np.abs(Phi_NN_test - Phi_test_target))

    #predict full profile
    X_full_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds_full_scaled = NN(X_full_tensor).numpy()
    Phi_NN_full = scaler_Y.inverse_transform(preds_full_scaled).flatten()

    full_rmse = np.sqrt(np.mean((Phi_NN_full - Phi_DNS) ** 2))
    full_rms_ref = np.sqrt(np.mean(Phi_DNS ** 2))
    full_rel_err = (full_rmse / np.maximum(full_rms_ref, 1e-6)) * 100
    full_max_err = np.max(np.abs(Phi_NN_full - Phi_DNS))

    #errors in each region
    sublayer_mask = (yplus_DNS <= 5)
    buffer_mask = (yplus_DNS > 5) & (yplus_DNS <= 30)
    log_mask = (yplus_DNS > 30) & (yplus_DNS <= 300)
    outer_mask = (yplus_DNS > 300)

    err_sub = np.sqrt(np.mean((Phi_NN_full[sublayer_mask] - Phi_DNS[sublayer_mask])**2)) if np.any(sublayer_mask) else 0.0
    err_buf = np.sqrt(np.mean((Phi_NN_full[buffer_mask] - Phi_DNS[buffer_mask])**2)) if np.any(buffer_mask) else 0.0
    err_log = np.sqrt(np.mean((Phi_NN_full[log_mask] - Phi_DNS[log_mask])**2)) if np.any(log_mask) else 0.0
    err_out = np.sqrt(np.mean((Phi_NN_full[outer_mask] - Phi_DNS[outer_mask])**2)) if np.any(outer_mask) else 0.0

    print(f"\nValidation for Phi_{stress_choice}")
    print(f"Test Set RMSE           : {test_rmse:.4e} (Rel Error: {test_rel_err:.2f}%)")
    print(f"Test Set Max Error      : {test_max_err:.4e}")
    print(f"Full Profile RMSE       : {full_rmse:.4e} (Rel Error: {full_rel_err:.2f}%)")
    print(f"Full Profile Max Error  : {full_max_err:.4e}")
    print(f"Regional RMSEs -> Sublayer (y+ <= 5): {err_sub:.4e} | Buffer (5-30): {err_buf:.4e} | Log (30-300): {err_log:.4e} | Outer (>300): {err_out:.4e}")

    #save model and scalers
    model_filename = f'nn/model-pressure-strain-{stress_choice}.pth'
    scalers_filename = f'nn/scalers-X-pressure-strain-{stress_choice}.bin'
    scaler_y_filename = f'nn/scaler-Phi-pressure-strain-{stress_choice}.bin'
    features_filename = f'nn/features-pressure-strain-{stress_choice}.txt'

    if not plotOnly:
        torch.save(NN, model_filename)
        dump(scalers, scalers_filename)
        dump(scaler_Y, scaler_y_filename)
        np.savetxt(features_filename, input_names, fmt='%s')
        print(f"Saved model to '{model_filename}' and scalers to '{scalers_filename}', '{scaler_y_filename}'")

    #check saved model predictions
    reloaded_model = torch.load(model_filename, weights_only=False)
    reloaded_model.eval()
    reloaded_scalers = load(scalers_filename)
    reloaded_scaler_Y = load(scaler_y_filename)

    with torch.no_grad():
        reloaded_preds_scaled = reloaded_model(X_test_tensor).numpy()
        reloaded_preds = reloaded_scaler_Y.inverse_transform(reloaded_preds_scaled)

    inference_diff = np.max(np.abs(reloaded_preds - Phi_NN_test))
    print(f"Inference reload verification diff: {inference_diff:.2e} (Passed: {inference_diff < 1e-5})")

    #plots
    yplus_test = yplus_DNS[index_test]
    sort_idx = np.argsort(yplus_test)
    yplus_test_plot = yplus_test[sort_idx]
    Phi_NN_test_plot = Phi_NN_test.flatten()[sort_idx]

    #full profile
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_DNS, Phi_DNS, 'b-', linewidth=2, label='DNS Target')
    #plt.plot(yplus_DNS, Phi_NN_full, 'r--', linewidth=2, label='NN Full Profile')
    plt.plot(yplus_test_plot, Phi_NN_test_plot, 'go', markersize=4, label='NN Test Points')
    plt.xlabel(r"$y^+$", fontsize=14)
    plt.ylabel(fr"$\Phi_{{{stress_choice}}}$", fontsize=14)
    plt.title(f"Pressure-Strain Redistribution $\\Phi_{{{stress_choice}}}$", fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.savefig(f'figs/fig9_Phi_{stress_choice}-vs-yplus.png', dpi=300, bbox_inches='tight')
    plt.close()

    #near wall profile
    plt.figure(figsize=(8, 6))
    plt.plot(yplus_DNS, Phi_DNS, 'b-', linewidth=2, label='DNS Target')
    #plt.plot(yplus_DNS, Phi_NN_full, 'r--', linewidth=2, label='NN Full Profile')
    plt.plot(yplus_test_plot, Phi_NN_test_plot, 'go', markersize=5, label='NN Test Points')
    plt.xlabel(r"$y^+$", fontsize=14)
    plt.ylabel(fr"$\Phi_{{{stress_choice}}}$", fontsize=14)
    plt.title(fr"Near-Wall $\Phi_{{{stress_choice}}}$ ($y^+ \leq 100$)", fontsize=14)
    plt.xlim(0, 100)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.savefig(f'figs/fig10_Phi_{stress_choice}-vs-yplus-zoom.png', dpi=300, bbox_inches='tight')
    plt.close()

    #training loss
    if not plotOnly:
        plt.figure(figsize=(8, 6))
        plt.semilogy(range(len(loss_v)), loss_v, 'b-', linewidth=2, label='Test MSE Loss')
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("MSE Loss", fontsize=14)
        plt.title(fr"Training Convergence ($\Phi_{{{stress_choice}}}$)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.savefig(f'figs/fig11_loss-Phi-{stress_choice}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("\n Plotting Completed" if plotOnly else "\n Training Completed")

