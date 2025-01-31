import numpy as np
import sys
from tqdm import tqdm
from numba import njit
from scipy.integrate import dblquad
from cubature import cubature
from mpi4py import MPI


@njit
def softmax(x, beta=1):
    max_x = np.array([np.max(row) for row in x])

    x = x - max_x.reshape(-1,1)
    P = np.exp(beta*x)    
    return (P.T / np.sum(P,1)).T


@njit
def A(z1):
    return softmax(np.outer(z1,z1)) + np.eye(len(z1))


@njit
def g(z1, z2):
    A1 = A(z1) @ z2
    return softmax(np.outer(A1, A1))


@njit
def x1_func(y1, y2, eps=1e-6):
    return - np.log(1/y1 - 1) / np.sqrt(np.log((1/y1 - 1)*(1/y2 - 1)) + eps)


@njit
def x2_func(y1, y2, eps=1e-6):
    return   np.log(1/y2 - 1) / np.sqrt(np.log((1/y1 - 1)*(1/y2 - 1)) + eps)


@njit
def z2_inv(z1, y):
    return np.linalg.solve(A(z1), y)


@njit
def Z_out_int_plus(z1, y, omega, V):
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z2 = z2_inv(z1, u)
    Z = np.zeros((2, len(z1)))
    Z[0] = z1
    Z[1] = z2

    Z_minus_omega = Z - omega.T 
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(A(z1)))


@njit
def Z_out_int_minus(z1, y, omega, V):
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = -np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z2 = z2_inv(z1, u)
    Z = np.zeros((2, len(z1)))
    Z[0] = z1
    Z[1] = z2

    Z_minus_omega = Z - omega.T 
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(A(z1)))


@njit
def g_out_int_plus(z1, y, omega, V):
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z2 = z2_inv(z1, u)
    Z = np.zeros((2, len(z1)))
    Z[0] = z1
    Z[1] = z2

    Z_minus_omega = Z - omega.T
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(A(z1))) * np.linalg.inv(V) @ Z_minus_omega


@njit
def g_out_int_minus(z1, y, omega, V):
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = -np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z2 = z2_inv(z1, u)
    Z = np.zeros((2, len(z1)))
    Z[0] = z1
    Z[1] = z2

    Z_minus_omega = Z - omega.T
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(A(z1))) * np.linalg.inv(V) @ Z_minus_omega


def Z_out(y, omega, V, inf=6):

    int = dblquad(lambda a,b: Z_out_int_plus(np.array([a,b]), y, omega, V) + Z_out_int_minus(np.array([a,b]), y, omega, V), -inf, inf, -inf, inf)[0]
    return int


def g_out(y, omega, V, inf=3):
    def func(x):
        return (g_out_int_plus(x, y, omega, V) + g_out_int_minus(x, y, omega, V)).flatten()
    
    result, _ = cubature(func, 2, 4, [-inf, -inf], [inf, inf])
    return result.reshape((2,2)) 


def dg_out_numerical_derivative(y, omega, V, inf=3, epsilon=1e-5):
    
    omega_shape = omega.shape
    dg_out_numerical = np.zeros((2,2,2,2))

    for i in range(omega_shape[0]):
        for j in range(omega_shape[1]):
            omega_plus = omega.copy()
            omega_minus = omega.copy()
            omega_plus[i, j] += epsilon
            omega_minus[i, j] -= epsilon
            
            g_plus = g_out(y, omega_plus, V, inf)
            Z_plus = Z_out(y, omega_plus, V, inf) 
            g_plus = g_plus / Z_plus

            g_minus = g_out(y, omega_minus, V, inf)
            Z_minus = Z_out(y, omega_minus, V, inf)
            g_minus = g_minus / Z_minus
            
            dg_out_numerical[i, j] = (g_plus - g_minus) / (2 * epsilon)
    
    return np.einsum("plmk->pkml", dg_out_numerical)



def compute_g_out_d_gout(y, omega, V, inf=3):
    V = V + 1e-3*np.eye(V.shape[0])
    V_inv_tensor = np.zeros((2,2,2,2))

    for i in range(2):
        V_inv_tensor[i,i,:,:] = np.linalg.inv(V)



    Z_out_val = Z_out(y, omega, V, inf) 
    g_out_val = g_out(y, omega, V, inf).T / Z_out_val
    dg_out_val =  dg_out_numerical_derivative(y, omega, V, inf)

    return g_out_val, dg_out_val

def main(MPI, d, alpha, max_iter=40, p=1, L=2, information=0.1, damping=0.9):
    rank = MPI.Get_rank()
    size = MPI.Get_size()


    if rank == 0:

        n = int(d*alpha)

        X = np.random.normal(0,1, (d, L, n))

        w = np.random.normal(0,1, (p, d))
        z = np.einsum("pi,iln->pln", w, X) / np.sqrt(d)

        y = np.zeros((L,L, n))
        for i in range(n):
            y[:,:, i] = g(z[0,:,i], z[1,:,i])

        chat = np.eye(p)

        what = np.random.normal(0,1, (p, d))
        g_out_val = np.zeros((L,p,n))
        dg_out_val = np.zeros((L,L,p,p,n))

        M_prev = - np.eye(p)
        M_iter = []
        e_gen_iter = []
        for iter in range(max_iter):
            V = chat
            omega = np.einsum("iln,pi->lpn", X, what)/np.sqrt(d) - np.einsum("pm,lmn->lpn", V, g_out_val)        


            omega_send = omega.reshape((L, p, n//size, size))
            V_send = V
            y_send = y.reshape((L, L, n//size, size))

            for i in range(1, size):
                MPI.send(omega_send[:, :, :, i], dest=i)
                MPI.send(V_send, dest=i)
                MPI.send(y_send[:, :, :, i], dest=i)

            g_out_local = np.zeros((L, p, n//size))
            dg_out_local = np.zeros((L,L, p,p, n//size))
            for mu in tqdm(range(n//size)):
                g_out_local[:,:, mu], dg_out_local[:,:, :,:, mu] = compute_g_out_d_gout(y_send[:,:,mu,0], omega_send[:,:,mu,0], V_send[:,:], inf=3)
            
            g_out_collect = np.zeros((L, p, n//size, size))
            dg_out_collect = np.zeros((L,L, p,p, n//size, size))

            for i in range(1, size):
                g_out_collect[:, :, :, i] = MPI.recv(source=i)
                dg_out_collect[:, :, :, :, :, i] = MPI.recv(source=i)
            g_out_collect[:, :, :, 0] = g_out_local
            dg_out_collect[:, :, :, :, :, 0] = dg_out_local

            g_out_val = g_out_collect.reshape((L, p, n))
            dg_out_val = dg_out_collect.reshape((L, L, p, p, n))
                                    
            A = - alpha / n * np.einsum("llpmn->pm", dg_out_val)
            b = np.einsum("dln,lpn->pd", X, g_out_val) / np.sqrt(d) + np.einsum("pm,md->pd", A, what)

            chat_proposal = np.zeros((p,p))
            what_proposal = np.zeros((p,d))
            chat_proposal[:,:] = np.linalg.solve(A[:,:]+np.eye(p), np.eye(p))
            for i in range(d):
                what_proposal[:,i] = np.linalg.solve(A[:,:]+np.eye(p), b[:,i])

            what = what_proposal * damping + what * (1-damping)
            chat = chat_proposal * damping + chat * (1-damping)

            Q = what @ what.T / d
            M = what @ w.T / d
            update_diff = np.linalg.norm(M - M_prev)

            M_iter.append(M)
            np.save(f"data_AMP_2w/AMP_BO_2w_d{d}_alpha{alpha}_samples{samples_average}_seed{int(sys.argv[2])}.npy", M_iter)

            n_gen = 1000

            X_gen = np.random.normal(0,1, (d, L, n_gen))

            z_gen = np.einsum("pi,iln->pln", w, X_gen) / np.sqrt(d)
            z_student = np.einsum("pi,iln->pln", what, X_gen) / np.sqrt(d)

            y_gen = np.zeros((L,L, n_gen))
            y_student = np.zeros((L,L, n_gen))
            for i in range(n_gen):
                y_gen[:,:, i] = g(z_gen[0,:,i], z_gen[1,:,i])
                y_student[:,:, i] = g(z_student[0,:,i], z_student[1,:,i])

            y_diff = y_gen - y_student
            e_gen = np.mean(np.linalg.norm(y_diff, axis=(0,1))**2)
            e_gen_iter.append(e_gen)

            print(f"\n\nIter {iter+1},\n Q = {Q},\n M = {M},\n diff = {update_diff}, \n e_gen = {e_gen}\n\n")
            np.save(f"data_AMP_2w/2w_e_gen_d{d}_alpha{alpha}_samples{samples_average}_seed{int(sys.argv[2])}.npy", e_gen_iter)


            M_prev = M
            
        return M_iter
    
    else:
        for iter in range(max_iter):
            omega = MPI.recv(source=0)
            V = MPI.recv(source=0)
            y = MPI.recv(source=0)

            n_worker = y.shape[2]
            L = y.shape[0]
            p = omega.shape[0]

            g_out_val = np.zeros((L,p,n_worker))
            dg_out_val = np.zeros((L,L,p,p,n_worker))

            for mu in range(n_worker):
                g_out_val[:,:, mu], dg_out_val[:,:, :,:, mu] = compute_g_out_d_gout(y[:,:,mu], omega[:,:,mu], V[:,:], inf=3)

            MPI.send(g_out_val, dest=0)
            MPI.send(dg_out_val, dest=0)

        return None
        

    
if __name__=="__main__":
    MPI = MPI.COMM_WORLD

    d = 1000
    alpha = float(sys.argv[1])
    samples_average = 1
    max_iter = 50
    information = 0.6
    damping = 0.8
    L = 2
    p = 2

    print(f"Escaping mediocrity in {np.log(d)} steps")

    M_list = np.zeros((samples_average))
    for s in range(samples_average):
        M_list[s] = main(MPI, d, alpha, max_iter, p, L, information, damping)


 