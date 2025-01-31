import numpy as np
from scipy.linalg import sqrtm
from numba import njit
from scipy.integrate import dblquad
from cubature import cubature
from tqdm import tqdm
from mpi4py import MPI
import sys



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

    Z_minus_omega = Z - omega 
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

    Z_minus_omega = Z - omega 
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

    Z_minus_omega = Z - omega 
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

    Z_minus_omega = Z - omega 
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(A(z1))) * np.linalg.inv(V) @ Z_minus_omega


def Z_out(y, omega, V, inf=3):
    int = dblquad(lambda a,b: Z_out_int_plus(np.array([a,b]), y, omega, V) + Z_out_int_minus(np.array([a,b]), y, omega, V), -inf, inf, -inf, inf)[0]
    return int


def g_out(y, omega, V, inf=5):
    def func(x):
        return (g_out_int_plus(x, y, omega, V) + g_out_int_minus(x, y, omega, V)).flatten()
    
    result, _ = cubature(func, 2, 4, [-inf, -inf], [inf, inf])
    return result.reshape((2,2)) / Z_out(y, omega, V)


def Q_func(Q_hat, eps=1e-2):
    return Q_hat @ np.linalg.inv(np.eye(Q_hat.shape[0]) + Q_hat)


def Q_hat_func_MCMC(alpha, Q, samples, L=2, eps=1e-3):
    Q_hat = np.zeros(Q.shape)
    for _ in tqdm(range(samples)):

        Z = np.random.normal(0,1, (Q.shape[0], L))
        U = np.random.normal(0,1, (Q.shape[0], L))

        sqrt_Q = sqrtm(Q)
        sqrt_one_minus_Q = sqrtm(np.eye(Q.shape[0]) - Q + eps*np.eye(Q.shape[0]))
        omega = sqrt_Q @ Z
        
        Z_true = sqrt_Q@Z + sqrt_one_minus_Q@U
        y = g(Z_true[0], Z_true[1])

        V = np.eye(Q.shape[0]) - Q + eps*np.eye(Q.shape[0])

        g_out_mat = g_out(y, omega, V)
        Q_hat += alpha * g_out_mat @ g_out_mat.T

    return Q_hat / samples


def main(alpha, Q, samples, iter, damping=.8):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        Q_list = []

    for it in range(iter):
        Q_hat = Q_hat_func_MCMC(alpha, Q, samples)
        
        if rank != 0:
            comm.send(Q_hat, dest=0)
            Q = comm.recv(source=0)
        
        if rank == 0:
            Q_hat_all = np.zeros((size, 2, 2), dtype=np.float64)

            Q_hat_all[0] = Q_hat
            for j in range(1, size):
                Q_hat_all[j] = comm.recv(source=j)

            Q_hat = np.mean(Q_hat_all, axis=0)
            Q = damping*Q_func(Q_hat) + (1-damping)*Q
            Q_list.append(Q)

            np.save(f"data_BO_2w_small_alpha/Q_list_alpha_{alpha}_samples_{int(size*samples)}.npy", Q_list)

            print(f"iter {it}: {Q}")

            Q = (Q + Q.T) / 2
            for j in range(1, size):
                comm.send(Q, dest=j)



        

if __name__=="__main__":
    alpha = float(sys.argv[1])
    iter = int(sys.argv[2])
    samples = int(sys.argv[3])
    Q = np.array([[.05,.0],[.0,.05]])



    main(alpha, Q, samples, iter)