import numpy as np
import torch 
import sys 
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
#from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from joblib import dump, load

plt.rcParams.update({'font.size': 22})
plt.interactive(True)
plt.close('all')

viscos= 1/5200

# load DNS data
# Re = 5200
#  y/delta  y^+  U  dU/dy  W   P        
DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];
dudy_DNS=np.gradient(u_DNS,y_DNS)

# % y/delta   y^+    u'u'    v'v'  w'w'  u'v'  u'w'  v'w'  k

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
y_DNS=DNS_stress[:,0];
u2_DNS=DNS_stress[:,2];
v2_DNS=DNS_stress[:,3];
w2_DNS=DNS_stress[:,4];
uv_DNS=DNS_stress[:,5];
k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

vist_DNS = np.abs(uv_DNS/dudy_DNS)

#y/delta y^+ Production Turbulent_Transport Viscous_Transport Pressure_Strain Pressure_Transport Viscous_Dissipation Balance
DNS_k_terms=np.genfromtxt("LM_Channel_5200_RSTE_k_prof.dat",comments="%")

diss_DNS=DNS_k_terms[:,7]
Pk_DNS=DNS_k_terms[:,2]
diff_DNS=DNS_k_terms[:,3]
diff_DNS_visc =   DNS_k_terms[:,4]

diss_iso_DNS = np.maximum(diss_DNS-diff_DNS_visc,1e-10)
diss_iso_DNS = diss_iso_DNS/viscos

# all velocioties are scaled with ustar
ustar_DNS = 1

re_t = k_DNS**2/diss_iso_DNS/viscos
u_diss = (diss_iso_DNS*viscos)**0.25
y_star_DNS = u_diss*y_DNS/viscos
y_plus_DNS = ustar_DNS*y_DNS/viscos

# current model of f_2
f_2=((1.-np.exp(-y_star_DNS/3.1))**2)*(1.-0.3*np.exp(-(re_t/6.5)**2))

# output is f_2_NN (see below) predicted by the NN. Our target is f_2
# transpose the target vector to make it a column vector  
Y = f_2.transpose()
Y= Y.reshape(-1,1)  # makes an array of size [len(Y), 1]


# we choose two inputs: yplus_DNS and ystar_DNS
# re-shape
yplus= y_plus_DNS.reshape(-1,1)
ystar= y_star_DNS.reshape(-1,1)
# use scaling, One for each input
scaler_yplus = MinMaxScaler()
scaler_ystar = MinMaxScaler()
# X = input matrix
X=np.zeros((len(yplus_DNS),2))
X[:,0] = scaler_yplus.fit_transform(yplus)[:,0]
X[:,1] = scaler_ystar.fit_transform(ystar)[:,0]

# split the feature matrix and target vector into training and validation sets
# test_size=0.2 means we reserve 20% of the data for validation
# random_state=42 is a fixed seed for the random number generator, ensuring reproducibility

# if you want to split the data differently every time you run the code, use randrange() as below
rand_state = randrange(100)

# use the same split every time
rand_state = 45

print('rand_state',rand_state)

indices = np.arange(len(X))
X_train, X_test, Y_train, Y_test, index_train, index_test = \
train_test_split(X, Y, indices,test_size=0.2,shuffle=True,random_state=rand_state)

f_2_train = f_2[index_train]
yplus_train = yplus[index_train]
ystar_train = ystar[index_train]

f_2_test = f_2[index_test]
yplus_test = yplus[index_test]
ystar_test = ystar[index_test]

# Set learning_rate (under-relaxation)
learning_rate = 0.05 
learning_rate = 0.1  #Epoch 200, Learning Rate: 1.00e-01, Loss: 2.85e-03, Loss_min: 2.85e-03
#learning_rate = 0.2  #Epoch 200, Learning Rate: 2.00e-01, Loss: 3.05e-03, Loss_min: 3.05e-03

# apply the optimizer element-wise in train_loop
N_epochs = 1000  # number of iterations
#N_epochs = 3

# convert the numpy arrays to PyTorch tensors with float32 data type
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.ll1 = nn.Linear(in_features=2,out_features=10) #axis 0: number of inputs
        self.tanh = nn.Tanh()
        self.ll2 = nn.Linear(in_features=10,out_features=10)
        self.ll3 = nn.Linear(in_features=10,out_features=10)
        self.output = nn.Linear(in_features=10,out_features=1) #axis 1: number of outputs
#
#  the NN below is much worse
#
#       self.input   = nn.Linear(2, 10)  #axis 0: number of inputs
#       self.hidden1 = nn.Linear(10, 10)
#       self.hidden2 = nn.Linear(10, 1)  #axis 1: number of outputs


    def forward(self, x):
        out = self.ll1(x)
        out = self.tanh(out)
        out = self.ll2(out)
        out = self.tanh(out)
        out = self.ll3(out)
        out = self.output(out)
#
#  the NN below is much worse
#
#       x = nn.functional.relu(self.input(x))
#       x = nn.functional.relu(self.hidden1(x))
#       out = self.hidden2(x)

        return out

def train_loop(model, loss_fn, optimizer):
        # Compute prediction and loss
    for n in range(0,len(X_train_tensor[:,1])):

      pred = model(X_train_tensor[n,:])
#     print('pred.shape',pred.shape)
#     print('Y_train_tensor[n,:].shape',Y_train_tensor[n,:].shape)
      loss = loss_fn(pred, Y_train_tensor[n,:])

        # Backpropagation
      optimizer.zero_grad()  # set all gradients from previous epoch to zero 
      loss.backward()
      optimizer.step()

def test_loop(model, loss_fn, loss_min):
    test_loss = 0


    pred = model(X_test_tensor)
# item(): Returns the value of this tensor as a standard Python number. This only works for tensors with one element
    test_loss = loss_fn(pred, Y_test_tensor)
    test_loss = test_loss.detach().numpy()


# Print the loss every epoch
    loss_min = np.minimum(test_loss,loss_min)
    torch.set_printoptions(precision=4)
#   lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, Learning Rate: {learning_rate:.2e}, Loss: {test_loss:.2e}, Loss_min: {loss_min:.2e}")


    return test_loss, loss_min
# Initiate the neural network
NN = MyNet()  

# Initiate the loss function
loss_fn = nn.MSELoss()

# Choose loss function, check out https://pytorch.org/docs/stable/optim.html for more info
# In this case we choose Stocastic Gradient Descent
optimizer = torch.optim.SGD(NN.parameters(), lr=learning_rate)
loss_v = np.zeros(N_epochs)

loss_min = 1e30
for epoch in range(N_epochs):
    train_loop(NN, loss_fn, optimizer)
    test_loss, loss_min = test_loop(NN, loss_fn, loss_min)
    loss_v[epoch] = test_loss

print("Done!")

preds = NN(X_test_tensor)

#transform from tensor to numpy
f_2_NN = preds.detach().numpy()
 
f_2_NN_old = f_2_NN

f_2_NN=f_2_NN[:,0]

f_2_std=np.std(f_2_NN-f_2_test)/(np.mean(f_2_test.flatten()**2))**0.5

print(f"STD error of f_2: {f_2_std:.2e}")

error_all=abs(f_2_NN-f_2_test)
error_index= error_all.argsort()

error_sorted = error_all[error_index]
# largest error:
largest_error = error_all[error_index[-1]]
print(f"Largest_error in f_2: {largest_error:.2e}")

# save NN model to disk
filename = 'model-neural-k-omega-f_2.pth'
torch.save(NN, filename)
dump(scaler_yplus,'scaler-yplus-k-omega-f_2.bin')
dump(scaler_ystar,'scaler-ystar-k-omega-f_2.bin')

yplus_min = np.min(yplus_train)
ystar_min = np.min(ystar_train)
f_2_min = np.min(f_2_train)

yplus_max = np.max(yplus_train)
ystar_max = np.max(ystar_train)
f_2_max = np.max(f_2_train)

np.savetxt('min-max-model-f_2.txt', [yplus_min,yplus_max,ystar_min,ystar_max,f_2_min,f_2_max])

########################## f_2 vs y
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus, f_2, 'bo',label='target')
plt.plot(yplus_test,f_2_NN, 'ro',label='NN')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$f_2$")
plt.legend(loc="best",fontsize=12)
plt.savefig('f_2-vs-yplus.png')

########################## f_2 vs ystar
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(ystar, f_2, 'bo',label='target')
plt.plot(ystar_test,f_2_NN, 'ro',label='NN')
plt.xlabel(r"$y^*$")
plt.ylabel(r"$f_2$")
plt.legend(loc="best",fontsize=12)
plt.savefig('f_2-vs-ystar.png')

########################## f_2 vs ystar zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(ystar, f_2, 'bo',label='target')
plt.plot(ystar_test,f_2_NN, 'ro',label='NN')
plt.xlabel(r"$y^*$")
plt.ylabel(r"$f_2$")
plt.xlim(9,100)
plt.legend(loc="best",fontsize=12)
plt.savefig('f_2-vs-ystar-zoom.png')

########################## f_2 vs y zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus, f_2, 'bo',label='target')
plt.plot(yplus_test,f_2_NN, 'ro',label='NN')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$f_2$")
plt.xlim(9,100)
plt.legend(loc="best",fontsize=12)
plt.savefig('f_2-vs-yplus-zoom.png')


########################## loss, error
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(loss_v, 'b-')
#plt.axis([100, len(loss_v),min(loss_v),loss_v[100]])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig('loss-f2.png')


