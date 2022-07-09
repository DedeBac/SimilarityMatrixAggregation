#Functions for computing the barycenters
import numpy as np
import scipy
from scipy import linalg
import hoggorm as ho
import random
import matplotlib
from matplotlib import pyplot as plt

#Cosine similarity function

def cos_sim(B): #B is a rectangular matrix of features. Rows are nodes, columns are features
    B_square = B.dot(np.transpose(B))
    dim = B_square.shape[0]
    sim_mat = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            sim_mat[i,j] = B_square[i,j]/np.sqrt(B_square[i,i]*B_square[j,j])
    return [sim_mat,B_square] #Returns the square SIMILARITY matrix, and the matrix B_square

def normalize_simmat(sim_mat): #THIS MUST BE USED TO NORMALIZE VALUES OF BARYCENTERS
    sim_mat_norm = np.zeros(sim_mat.shape)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[0]):
            sim_mat_norm[i,j]=sim_mat[i,j]/((sim_mat[i,i]**(1/2))*(sim_mat[j,j]**(1/2)))
    return sim_mat_norm

def modif_mod(k,m): #returns module k%m, but sets null residuals to m
    res = k%m
    if(res==0):
        res=m
    return res

#Function to evaluate how well the weights represent matrices
def eval_weights(RVcoeff):
    eval, evec = np.linalg.eig(RVcoeff)
    idx = eval.argsort()[::-1]
    eigenValues = eval[idx]
    e_max = eigenValues[0]
    estimate = e_max/sum(eigenValues)
    return estimate

# Function used to perturb data matrix in case it is almost singular
#tol_eig is a small constant used to perturb eigenvalues, pert_tol is a tolerance on the matrix distance from the original
def eigenval_perturb(data_matrix,tol_eig = 2e-12,pert_tol= 2e-12):
    eva, evec = np.linalg.eigh(data_matrix)
    eva_neg = eva[eva<0]
    if(len(eva_neg)==0):
        reconstructed = data_matrix
    else:
        eva_t = eva + np.abs(min(eva_neg))+tol_eig
        reconstructed = np.dot(evec * eva_t, evec.conj().T)
        dist = np.allclose(reconstructed, data_matrix,rtol=pert_tol,atol=pert_tol)
    #print('Good representation:\t',dist)
    return reconstructed

def frobenius_weights(RV): #compute Frobenius weights for the computation of the weighted arithmetic mean, given the RV matrix
    eigenval, eigenvec = np.linalg.eig(RV)
    idx = eigenval.argsort()[::-1] # sort from largest to smaller eigenval e eigenvec
    eigenValues = eigenval[idx]
    eigenVectors = eigenvec[:, idx]
    e_max = eigenValues[0]  #extract largest eigenvalue and its index
    max_evec = eigenVectors[:, 0]
    weights_vector = max_evec / (sum(max_evec))
    return weights_vector #Returns vector of weights

def riem_weights(list_of_matrices):#Compute weights for the computation of the Riemannian barycenter using the Riemann metric
    n = len(list_of_matrices)
    corr_mat = np.identity(n)
    for i in range(n-1):
        mi = list_of_matrices[i]
        for j in range(i+1,n):
            mj = list_of_matrices[j]
            corr_mat[i, j] = np.trace(np.dot(np.transpose(mi),mj)) / ((np.linalg.norm(mi, 'fro'))*(np.linalg.norm(mj, 'fro')));
    corr_mat1 = corr_mat + np.transpose(corr_mat) - np.identity(n)
    w_denom = np.matmul(np.matmul(np.ones((1,n)),corr_mat1-np.identity(n)),np.ones((n,1)))
    w_num = np.matmul((corr_mat1-np.identity(n)),np.ones((len(list_of_matrices),1)))
    w = w_num/w_denom
    return [w,corr_mat1] #returns weights vector and correlation matrix used to compute weights

def frobenius_avg(list_of_similaritymat,weights): #list of similarity matrices as input, together with a weight vector
    sum = 0
    for i in range(0,len(list_of_similaritymat)):
        sum = sum + list_of_similaritymat[i]*weights[i]
    return sum #returns baricenter computed with choice of the Frobenius norm

# geommean_new computes the weighted geometric mean between two matrices A, B, with weight t (real)
def geommean_new(A, B, t,corr_fact = 0):
    #corr_fact set to small positive value if matrices are almost singular
    #check if A and B are positive definite: if not, return Error
    def check_input():
        check_input = 0
        def is_pos_def(x):
            return np.all(np.linalg.eigvalsh(x) > 0)

        if ((is_pos_def(A) and is_pos_def(B))=='False'):
            print('Input Error: all input matrices must be positive definite!')
            check_input = 1
            return
        if(t<0 or t>1):
            print('Input Error: t is a real number in range [0,1]')
            check_input = 1
        return check_input
    print('check_input',check_input())
    if(check_input()==0):
        #print('Inserted correct input, processing...')

        if(np.linalg.cond(A) >= np.linalg.cond(B)): #exchange A and B according to conditioning number, and turn t into (1-t)
            print('change roles of matrices')
            Anew = B
            Bnew = A
            A = Anew
            B = Bnew
            t = 1-t

        lowRA = np.linalg.cholesky(A)  #RA* lower triangular
        uppRA = np.transpose(lowRA)    #RA  upper triangular
        #uppRB = np.transpose(lowRB)    #RB
        invchol1 = np.linalg.inv(lowRA)     #RA*^(-1)
        invchol2 = np.linalg.inv(uppRA)     #RA^(-1)
        V = invchol1.dot(B).dot(invchol2)
        [U,diag_vec,U1]=np.linalg.svd(V,hermitian=True)
        #D,U = scipy.linalg.schur(V)
        D = np.diag(diag_vec)

        if(corr_fact!=0):
            pert_D = D + np.min(np.diagonal(D)) + corr_fact
        else:
            pert_D = D
        Dpower = np.identity(D.shape[0])
        np.fill_diagonal(Dpower, np.power(np.diagonal(pert_D), t))
        middle = U.dot(Dpower).dot(np.transpose(U))
        w_geometric_meanAB = lowRA.dot(middle).dot(uppRA)
        return w_geometric_meanAB
    else:
        return str('Process arrested: wrong input given.')

# RiemBar computes the Riemann barycenter via an iterative procedure
def RiemBar(k_init = int,list_of_mat=list, max_iter=int, tollr=float, weights=list,corr_fact = float):
    m = len(list_of_mat)
    tollist = list()   #list where storing tolerance values for each iteration
    #barlist = list() #uncomment and add to output if printing of barycenter at each iteration is required
    iter = 1
    k = k_init
    toll = tollr
    max_iter = max_iter
    jk = modif_mod(k,m)
    X_start = list_of_mat[jk - 1]
    flag = 1
    X_prec = X_start

    if(m==2): #if number of matrices =2, use closed form of the geometric weighted mean
        X_succ = geommean_new(list_of_mat[1],list_of_mat[0],weights[0])
        iter_counter=0
        flag=0

    while(flag):
        iter_counter = iter
        #print(iter_counter)
        jksucc = modif_mod(k+1,m)
        print('Index of matrix for geommean:\t',jksucc)
        w_exp = weights[jksucc - 1]
        print('',w_exp)
        S_succ = list_of_mat[jksucc - 1]
        denom = 0;
        indices = np.array(list(range(k+1)))+1

        for i in indices:
            denom = denom + weights[modif_mod(i,m)-1]
        #print('Num:\t',w_exp,'\tDenom:\t',denom)
        #print('exp:\t',w_exp/denom,'\n')
        X_succ = geommean_new(X_prec, S_succ, w_exp/denom, corr_fact= corr_fact) #provo con la media nuova
        #print('Current barycenter, first 5x5 submatrix:\n',X_succ[0:5,0:5])
        #barlist.append(X_succ) #uncomment if printing of the barycenter at each iteration is required
        diff = X_succ-X_prec
        num_err1 = np.trace(diff.dot(np.transpose(diff)))
        num_err2 = np.linalg.norm(X_succ - X_prec,'fro')
        den_err1= np.trace(X_prec.dot(np.transpose(X_prec)))
        den_err2 = np.linalg.norm(X_prec,'fro')
        #print('Num1:\t', num_err1)
        #print('Num2:\t', num_err2)
        #print('Denom1:\t', den_err1)
        #print('Denom2:\t',den_err2)
        err_rel1 = num_err1/den_err1
        err_rel2 = num_err2/den_err2
        print('Old_Relative_error:\t',err_rel2)
        print('Current_Relative_error:\t', err_rel1)
        err_rel=err_rel1
        tollist.append(err_rel)

        if (err_rel <= toll):
            print('Current iteration:\t', iter_counter)
            flag = 0

        if (iter_counter >= max_iter):
            print('Reached max number of iterations\n')
            #baric = kx
            flag = 0

        else:
            iter = iter+1
            print('iter_n:\t',iter)
            k = k + 1
            #print(X_succ[0:5,0:5])
            X_prec = X_succ

    out = [X_succ,iter_counter,tollist]
    #out = [X_succ, iter_counter, tollist, barlist] use this if you need list of baricenters
    return out

## Define Wasserstein metric between two completely positive matrices.
def wass_metric(A,B):
    sum = A+B
    radA = scipy.linalg.sqrtm(A)
    double_prod = np.sqrt(2*(radA.dot(B).dot(radA)))
    norm = np.sqrt(np.trace(sum-double_prod))
    return norm

## Define Wasserstein barycenter between two completely positive matrices.
def wass_mean2(X,Y,weights):
    s1 = (weights[0]**2)*X
    s2 = (weights[1]**2)*Y
    s12 = weights[0]*weights[1]*(scipy.linalg.sqrtm(X.dot(Y))+scipy.linalg.sqrtm(Y.dot(X)))
    bar = sum([s1,s2,s12])
    return bar

def kx_compute(matrix,mat_list,weights): ##compute K(X_k)
    m = len(mat_list)
    rad = scipy.linalg.sqrtm(matrix)
    negrad = np.linalg.inv(rad)
    to_sum = [0]*m
    for i in range(m):
        a = rad.dot(mat_list[i]).dot(rad)
        sqa = scipy.linalg.sqrtm(a)
        to_sum[i]=weights[i]*sqa
    somma = sum(to_sum)
    #sq_somma = somma.dot(somma)
    sq_somma = somma**2
    out = negrad.dot(sq_somma).dot(negrad)
    return out

def square_root_matrix(A):
    [eigvalues,V] = np.linalg.eig(A)
    diag_root = [eigvalues[i]**(1/2) for i in range(len(eigvalues))]
    diagarray = np.diag(np.array(diag_root))
    root_matrix = V.dot(diagarray).dot(np.linalg.inv(V))
    return root_matrix




def wass_meanM(list_of_matrices,weights,tollr,imax):
    m = len(list_of_matrices)
    k = random.randint(0, m - 1) #choose randomly a start matrix
    iter_count = 1
    tolvec = list()
    X_k = list_of_matrices[k]
    flag = 0
    if (m == 2):
        baric = wass_mean2(list_of_matrices[0],list_of_matrices[1],weights)
        flag=1
    #def kx_compute(matrix,mat_list,weights): ##compute K(X_k)
    #    m = len(mat_list)
    #    #rad = scipy.linalg.sqrtm(matrix)
    #    rad = square_root_matrix(matrix)
    #    negrad = np.linalg.inv(rad)
    #    to_sum = [0]*m
    #    for i in range(m):
    #        a = rad.dot(mat_list[i]).dot(rad)
    #        #sqa = scipy.linalg.sqrtm(a)
    #        sqa = square_root_matrix(a)
    #        to_sum[i]=weights[i]*sqa
    #    somma = sum(to_sum)
    #    #sq_somma = somma.dot(somma)
    #    sq_somma = somma**2
    #    out = negrad.dot(sq_somma).dot(negrad)
    #    return out
    while(flag==0):
        kx = kx_compute(X_k, list_of_matrices, weights)
        ###OLDerr_rel = np.linalg.norm(X_k - kx,'fro')/np.linalg.norm(kx,'fro')
        diff = kx - X_k
        num_err1 = np.trace(diff.dot(np.transpose(diff)))
        #num_err2 = np.linalg.norm(kx - X_k, 'fro')
        den_err1 = np.trace(X_k.dot(np.transpose(X_k)))
        #den_err2 = np.linalg.norm(X_k, 'fro')
        # print('Num1:\t', num_err1)
        # print('Num2:\t', num_err2)
        # print('Denom1:\t', den_err1)
        # print('Denom2:\t',den_err2)
        err_rel1 = num_err1/den_err1
        #err_rel2 = num_err2/den_err2
        print('Numeratore:\t', num_err1)
        print('Denominatore:\t', den_err1)
        err_rel = err_rel1
        #############################
        #print(err_rel)
        tolvec.append(err_rel)
        if(err_rel<=tollr):
            print('Current iteration:\t',iter_count)
            flag=1
            baric = kx
        if(iter_count >= imax):
            print('Reached max number of iterations\n')
            baric = kx
            flag=1
        else:
            iter_count+=1
            print('iteration:\t',iter_count)
            X_k = kx
    return [baric,tolvec]

