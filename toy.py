import numpy as np
import scipy
from scipy import linalg
from functions import*
from scipy.linalg import sqrtm
a = np.array(([50,1],[1,1]))
b = np.array(([1.01,1],[1,1])) #1.01,1
c = np.array(([1,1],[1,2]))

mlist=[a,b,c]

[weights,RVmat] = riem_weights(mlist)
frob_weights = frobenius_weights(RVmat)

#Compute frob barycenter (arithmetic mean)
Frob_bar = frobenius_avg(mlist,frob_weights)

#Compute Wass mean and Riemann barycenter
[Wassbar,Wtolvec]=wass_meanM(mlist,weights = weights,tollr=2e-8,imax = 10)
[BAR,IT,TOLVEC]=RiemBar(k_init=1,list_of_mat=mlist, max_iter=200, tollr=1e-12, weights=weights,corr_fact = 0)#problemino: mi viene una matrice che non è definita positiva

# Normalize barycenters s.t. the diagonal elements are equal to 1 (if similarity matrices are input, the Frobenius mean
# is already normalized)

norm_RiemBar = normalize_simmat(BAR)
norm_Wassbar = normalize_simmat(Wassbar)
norm_Frobbar = normalize_simmat(Frob_bar)

