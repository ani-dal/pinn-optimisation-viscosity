import numpy as np
import torch
import time
from scipy import sparse
import sys
from scipy.sparse import spdiags,linalg,eye
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from joblib import dump, load


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.ll1 = nn.Linear(in_features=2,out_features=10) #axis 0: number of inputs
        self.tanh = nn.Tanh()
        self.ll2 = nn.Linear(in_features=10,out_features=10)
        self.ll3 = nn.Linear(in_features=10,out_features=10)
        self.output = nn.Linear(in_features=10,out_features=1) #axis 1: number of outputs

    def forward(self, x):
        out = self.ll1(x)
        out = self.tanh(out)
        out = self.ll2(out)
        out = self.tanh(out)
        out = self.ll3(out)
        out = self.output(out)

        return out



plt.rcParams.update({'font.size': 16})
plt.interactive(True)
plt.close('all')

#

# exemple of 1d Channel flow with a k-epsa model. Re=u_tau*h/nu=5200 (h=half 
# channel height).
#
# Discretization described in detail in
# https://www.cfd-sweden.se/lada/comp_fluid_dynamics/
#
# and the pyCALC-RANS report at 
#
# https://www.cfd-sweden.se/lada/pyCALC-RANS.html

# Use NN model or no
NN_bool = False
#NN_bool = True


# max number of iterations
maxit=25000
maxit=40000
#maxit=60000
#maxit=100

# convergence criteria
res_limit =  5e-14

# the grid has 70 interior cells (computatiional nodes), numbered from 0 to 69. No 
# computatiional nodes at the boundaries. The b.c. are implemented as source terms as
# in pyCALC-RANS and pyCALC-LES
#
nj=70
yfac=1.1 # stretching
viscos=1/5200
dy=0.1
ymax=1
yc=np.zeros(nj+1)
yc[0]=0.
# face (grid) coordinates
for j in range(1,nj+1):
    yc[j]=yc[j-1]+dy
    dy=yfac*dy

ymax_scale=yc[-1]

# make max yc = y_max
for j in range(0,nj+1):
   yc[j]=yc[j]/ymax_scale*ymax

# cell centres (computatiional nodes)
yp=np.zeros(nj)
for j in range(0,nj):
   yp[j]=0.5*(yc[j]+yc[j+1])


DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
u2DNS=DNS_stress[:,2];
v2DNS=DNS_stress[:,3];
w2DNS=DNS_stress[:,4];
uvDNS=DNS_stress[:,5];

k_DNS = 0.5*(u2DNS + v2DNS + w2DNS)





# under-relaxation
urf=0.5

# plot k for each iteration at node jmon
jmon=nj-1 

# turbulent constants 
c_eps_1=1.5
c_eps_2=1.9

prand_eps=1.4
prand_k=1.4

cmu=0.09

small=1.e-10
great=1.e10

# initialaze
u=np.ones(nj)*yp**(1/7)
vist=np.zeros(nj)
k=np.ones(nj)*1.e-4
y=np.zeros(nj)
eps=np.ones(nj)*1.e-5
k=np.ones(nj)*1.e-4
eps=np.ones(nj)*1.e-5
vist=np.ones(nj)*100.*viscos
fy=np.zeros(nj)
dn=np.zeros(nj)
ds=np.zeros(nj)
dy_s=np.zeros(nj)
delta_y=np.zeros(nj)
f2=np.zeros(nj)
fmu=np.zeros(nj)
tau_w=np.zeros(maxit)
k_iter=np.zeros(maxit)
eps_iter=np.zeros(maxit)


# load NN model
if NN_bool:
   NN = torch.load('model-neural-k-omega-f_2.pth',weights_only=False)
   scaler_yplus = load('scaler-yplus-k-omega-f_2.bin')
   scaler_ystar = load('scaler-ystar-k-omega-f_2.bin')


   yplus_min,yplus_max,ystar_min,ystar_max,f2_min,f2_max = np.loadtxt('min-max-model-f_2.txt')


# do a loop over all nodes 
for j in range(0,nj):
# compute deltay
   delta_y[j]=yc[j+1]-yc[j]

for j in range(0,nj):
   jm1 = max(j-1,0)
   jp1 = min(j+1,nj-1)
# compute dy_s
   dy_s=np.maximum(yp[j]-yp[jm1],1e-10)

# compute dy_n
   if j < nj:
      dy_n=np.maximum(yp[jp1]-yp[j],1e-10)

   dn[j]=1./dy_n
   ds[j]=1./dy_s

# interpolation factor
   if j < nj:
      del1=np.maximum(yc[jp1] - yp[j],1e-10)
      del2=np.maximum(yp[jp1]-yc[jp1],1e-10)
      fy[j]=del1/(del1+del2)  # Pe/(Pe+Ee)

fy[-1] = 0 # it is not used

su=np.zeros(nj)
sp=np.zeros(nj)
an=np.zeros(nj)
as1=np.zeros(nj)
ap=np.zeros(nj)

# do max. maxit iterations
for n in range(1,maxit):
    iter = n
    dudy=np.gradient(u,yp,edge_order = 2)
# fix dudy = 0 at center
    dudy[-1] = 0
    dudy2=dudy**2

    if NN_bool:
       ustar=(viscos*u[1]/yp[1])**0.5
       yplus = np.minimum(yp,2-yp)*ustar/viscos
       ueps=(eps*viscos)**0.25
       ystar=ueps*np.minimum(yp,2-yp)/viscos

# count values larger/smaller than max/min
       yplus_min_number= (yplus< yplus_min).sum()
       yplus_max_number= (yplus > yplus_max).sum()

       print('yplus_min_number',yplus_min_number)
       print('yplus_max_number',yplus_max_number)

       ystar_min_number= (ystar< ystar_min).sum()
       ystar_max_number= (ystar > ystar_max).sum()

       print('ystar_min_number',ystar_min_number)
       print('ystar_max_number',ystar_max_number)

# set limits
       yplus=np.minimum(yplus,yplus_max)
       yplus=np.maximum(yplus,yplus_min)
       ystar=np.minimum(ystar,ystar_max)
       ystar=np.maximum(ystar,ystar_min)

# re-shape
       yplus_shape= yplus.reshape(-1,1)
       ystar_shape= ystar.reshape(-1,1)

       X=np.zeros((len(dudy),2))
       X[:,0] = scaler_yplus.transform(yplus_shape)[:,0]
       X[:,1] = scaler_ystar.transform(ystar_shape)[:,0]

# convert the numpy arrays to PyTorch tensors with float32 data type
       X= torch.tensor(X, dtype=torch.float32)

       preds = NN(X)

#transform from tensor to numpy
       f2_NN = preds.detach().numpy()

# count values larger/smaller than max/min
       f2_min_number= (f2_NN <= f2_min).sum()
       f2_max_number= (f2_NN >= f2_max).sum()

       print('f2_min_number',f2_min_number)
       print('f2_max_number',f2_max_number)

# set limits
       f2_NN=np.minimum(f2_NN,f2_max)
       f2_NN=np.maximum(f2_NN,f2_min)


# solve u
    for j in range(0,nj):

      jm1 = max(j-1,0)
      jp1 = min(j+1,nj-1)
# source term
      su[j]=delta_y[j] 
      sp[j]= 0

# interpolate turbulent viscosity to faces
      vist_n=fy[j]*vist[jp1]+(1-fy[j])*vist[j]
      vist_s=fy[jm1]*vist[j]+(1-fy[jm1])*vist[jm1]

# compute an & as
      an[j]=(vist_n+viscos)*dn[j]
      as1[j]=(vist_s+viscos)*ds[j]


# as1 and an are not used at the boundaries (there are no nodes at the boundaries)
    as1[0]=0
    an[-1]=0

# boundary conditions for u at y=0
    sp[0]=sp[0]-viscos/(0.5*delta_y[0])

    res_u = 0
    for j in range(0,nj):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.0-urf)*ap[j]*u[j]

# use sparse-matrix solver
    m=nj
    A = sparse.diags([ap, -an[0:-1], -as1[1:]],  [0, 1, -1], format='csr')
    res_u=np.linalg.norm(A*u - su)
    u = linalg.spsolve(A,su)

# monitor the development of u_tau in node jmon
    tau_w[n]=viscos*u[0]/yp[0]

# print iteration info
    tau_target=1
    print(f"\n{'---iter: '}{n:2d}, {'wall shear stress: '}{tau_w[n]:.2e},{'  tau_w_target='}{tau_target:.2e}\n")

# check for convergence (when converged, the wall shear stress must be one)
    ntot=n
    if abs(tau_w[n]-1) < 0.001:
# do at least 1000 iter 
        if n > 1000:
           print('Converged!')
           break

# solve k
    for j in range(0,nj):

      jm1 = max(j-1,0)
      jp1 = min(j+1,nj-1)
# source term
      dist =  yp[j]
      ueps=(eps[j]*viscos)**0.25
      ystar=ueps*dist/viscos
      rt=k[j]**2/eps[j]/viscos
      if NN_bool:
         f2[j]=f2_NN[j,0]
      else:
         f2[j]=((1.-np.exp(-ystar/3.1))**2)*(1.-0.3*np.exp(-(rt/6.5)**2))
      fmu[j]=((1.-np.exp(-ystar/14.))**2)*(1.+5./rt**0.75*np.exp(-(rt/200.)**2))
      fmu[j]=np.minimum(fmu[j],1.)

# production term
      su[j]=vist[j]*dudy2[j]*delta_y[j]
   
# dissipation term
      sp[j]=-eps[j]/k[j]*delta_y[j]

# interpolate turbulent viscosity to faces
      vist_n=fy[j]*vist[jp1]+(1-fy[j])*vist[j]
      vist_s=fy[jm1]*vist[j]+(1-fy[jm1])*vist[jm1]

# compute an & as
      an[j]=(vist_n/prand_k+viscos)*dn[j]
      as1[j]=(vist_s/prand_k+viscos)*ds[j]

# as1 and an are not used at the boundaries (there are no nodes at the boundaries)
    as1[0]=0
    an[-1]=0

# boundary conditions for k
    sp[0]=sp[0]-viscos/(0.5*delta_y[0])

    res_k = 0
    for j in range(0,nj):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.0-urf)*ap[j]*k[j]

      jm1 = max(j-1,0)
      jp1 = min(j+1,nj-1)

# use sparse-matrix solver
    m=nj
    A = sparse.diags([ap, -an[0:-1], -as1[1:]],  [0, 1, -1], format='csr')
    res_k=np.linalg.norm(A*k - su)
    k = linalg.spsolve(A,su)

# monitor the development of k in node jmon
    k_iter[n]=k[jmon]

# solve eps
    for j in range(0,nj):

      jm1 = max(j-1,0)
      jp1 = min(j+1,nj-1)
# source term
      dist =  yp[j]
      ueps=(eps[j]*viscos)**0.25
      ystar=ueps*dist/viscos
      rt=k[j]**2/eps[j]/viscos
      if NN_bool:
         f2[j]=f2_NN[j,0]
      else:
         f2[j]=((1.-np.exp(-ystar/3.1))**2)*(1.-0.3*np.exp(-(rt/6.5)**2))
      fmu[j]=((1.-np.exp(-ystar/14.))**2)*(1.+5./rt**0.75*np.exp(-(rt/200.)**2))
      fmu[j]=np.minimum(fmu[j],1.)

# production term
      su[j]=c_eps_1*cmu*fmu[j]*dudy2[j]*k[j]*delta_y[j]

# dissipation term
      sp[j]=-c_eps_2*f2[j]*eps[j]*delta_y[j]/(k[j]+1.e-10)

# interpolate turbulent viscosity to faces
      vist_n=fy[j]*vist[jp1]+(1-fy[j])*vist[j]
      vist_s=fy[jm1]*vist[j]+(1-fy[jm1])*vist[jm1]

# compute an & as
      an[j]=(vist_n/prand_eps+viscos)*dn[j]
      as1[j]=(vist_s/prand_eps+viscos)*ds[j]

# as1 and an are not used at the boundaries (there are no nodes at the boundaries)
    as1[0]=0
    an[-1]=0

    res_eps = 0
    for j in range(0,nj):
# compute ap
      ap[j]=an[j]+as1[j]-sp[j]

# under-relaxate
      ap[j]= ap[j]/urf
      su[j]= su[j]+(1.0-urf)*ap[j]*eps[j]

# fix eps at first interior cell
    dy=yp[0]
    eps_wall=2*viscos*k[0]/yp[0]**2 
    ap[0]=1
    as1[0]=0
    an[0]=0
    su[0]=eps_wall

# use sparse-matrix solver
    m=nj
    A = sparse.diags([ap, -an[0:-1], -as1[1:]],  [0, 1, -1], format='csr')
    res_eps=np.linalg.norm(A*eps - su)
    eps = linalg.spsolve(A,su)

# monitor the development of eps in node jmon
    eps_iter[n]=eps[jmon]

# compute viscosity
    for j in range(0,nj):
      vist_new = cmu*fmu[j]*k[j]**2/eps[j]
      vist[j] = vist_new*urf + (1-urf)*vist[j]

# print residuals
    print(f"\n{'---iter: '}{n:2d}, {'res u: '}{res_u:.2e},{'  res k='}{res_k:.2e},{'  res eps='}{res_eps:.2e}\n")


   
# plot u log-scale
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ustar=tau_w[ntot]**0.5
uplus=u/ustar
plt.semilogx(yp*ustar/viscos,uplus,'b-',label="CFD")
plt.semilogx(yplus_DNS,u_DNS,'r-',label="DNS")
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 28])
plt.legend(loc="best",prop=dict(size=18))
plt.savefig('u_log-5200-NN-kom.png')


######################## plot u  linear scale
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(u/np.max(u),yp,'b-',label="CFD")
plt.plot(u_DNS/np.max(u_DNS),y_DNS,'r-',label="DNS")
plt.xlabel("$U$")
plt.ylabel("$y$")
plt.legend(loc="best",prop=dict(size=18))
plt.savefig('u_5200-NN-kom.png')


######################## plot k  vs iter
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(k_iter[0:iter],'b-')
plt.xlabel("iteration")
plt.ylabel("$k$")
plt.savefig('k-vs-iteration-5200-NN-kom.png')

######################## plot k 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(k,yp/viscos,'b-',label="CFD")
plt.plot(k_DNS,y_DNS/viscos,'r-',label="DNS")
plt.xlabel("$k$")
plt.ylabel("$y^+$")
plt.legend(loc="best",prop=dict(size=18))
plt.savefig('k_5200-NN-kom.png')


######################## plot k  zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(k,yp/viscos,'b-',label="CFD")
plt.plot(k_DNS,y_DNS/viscos,'r-',label="DNS")
plt.xlabel("$k$")
plt.ylabel("$y^+$")
plt.ylim(0,100)
plt.legend(loc="best",prop=dict(size=18))
plt.savefig('k_5200-NN-kom-zoom.png')

######################## plot vist  zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yp/viscos,vist/viscos,'b-',label="CFD")
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y^+$")
plt.xlim(0,50)
plt.ylim(0,20)
plt.legend(loc="best",prop=dict(size=18))
plt.savefig('vist-NN-kom-zoom.png')




