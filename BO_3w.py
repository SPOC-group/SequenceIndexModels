import numpy as np
from scipy.linalg import sqrtm
from numba import njit
from scipy.integrate import dblquad
from cubature import cubature
from tqdm import tqdm
from mpi4py import MPI
import sys



skip = 1.0


@njit
def softmax(x, beta=1):
    max_x = np.array([np.max(row) for row in x])

    x = x - max_x.reshape(-1,1)
    P = np.exp(beta*x)    
    return (P.T / np.sum(P,1)).T


@njit
def A(z1):
    return softmax(np.outer(z1,z1)) + np.eye(len(z1))*skip


@njit
def B(z1, z2):
    A1 = A(z1)
    A2 = A(A1 @ z2)
    return A2 @ A1


@njit
def g(z1, z2, z3):
    B1 = B(z1, z2) @ z3
    return softmax(np.outer(B1, B1))


@njit
def x1_func(y1, y2, eps=1e-6):
    return - np.log(1/y1 - 1) / np.sqrt(np.log((1/y1 - 1)*(1/y2 - 1)) + eps)


@njit
def x2_func(y1, y2, eps=1e-6):
    return   np.log(1/y2 - 1) / np.sqrt(np.log((1/y1 - 1)*(1/y2 - 1)) + eps)


@njit
def z3_inv(z1, z2, y):
    return np.linalg.solve(B(z1, z2), y)



@njit
def Z_out_int_plus(z1, z2, y, omega, V):    
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z3 = z3_inv(z1, z2, u)
    Z = np.zeros((3, len(z1)))
    Z[0] = z1
    Z[1] = z2
    Z[2] = z3

    Z_minus_omega = Z - omega 
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(B(z1, z2)))


@njit
def Z_out_int_minus(z1, z2, y, omega, V):    
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = -np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z3 = z3_inv(z1, z2, u)
    Z = np.zeros((3, len(z1)))
    Z[0] = z1
    Z[1] = z2
    Z[2] = z3


    Z_minus_omega = Z - omega 
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(B(z1, z2)))


@njit
def g_out_int_plus(z1, z2, y, omega, V):
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z3 = z3_inv(z1, z2, u)
    Z = np.zeros((3, len(z1)))
    Z[0] = z1
    Z[1] = z2
    Z[2] = z3

    Z_minus_omega = Z - omega 
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(B(z1,z2))) * np.linalg.inv(V) @ Z_minus_omega


@njit
def g_out_int_minus(z1, z2, y, omega, V):
    y = np.abs(np.array([y[0,1], y[1,0]]))
    if y.sum() > 1 or y.sum() < 0:
        print(y)
        assert y.sum() < 1 and y.sum() > 0

    u = -np.array([x1_func(y[0], y[1]), x2_func(y[0], y[1])])
    z3 = z3_inv(z1, z2, u)
    Z = np.zeros((3, len(z1)))
    Z[0] = z1
    Z[1] = z2
    Z[2] = z3

    Z_minus_omega = Z - omega 
    return np.exp( - np.trace(Z_minus_omega.T @ np.linalg.inv(V) @ Z_minus_omega) / 2 ) / np.abs(np.linalg.det(B(z1,z2))) * np.linalg.inv(V) @ Z_minus_omega



def Z_out(y, omega, V, inf=3, samples_internal=int(1e4)):
    inf_a = inf * np.linalg.inv(sqrtm(V))[0,0]
    inf_b = inf * np.linalg.inv(sqrtm(V))[1,1]
    integral = 0
    for _ in range(samples_internal):
        a = np.random.uniform(-inf_a, inf_a, (2))
        b = np.random.uniform(-inf_b, inf_b, (2))
        integral += Z_out_int_plus(a, b, y, omega, V) + Z_out_int_minus(a, b, y, omega, V)

    result = integral/samples_internal
    return max(result, 1e-5)


def g_out(y, omega, V, inf=3, samples_internal=int(1e4)):
    def func(x, z):
        return (g_out_int_plus(x, z, y, omega, V) + g_out_int_minus(x, z, y, omega, V)).flatten()
    
    inf_a = inf * np.linalg.inv(sqrtm(V))[0,0]
    inf_b = inf * np.linalg.inv(sqrtm(V))[1,1]
    
    result = np.zeros((6))
    for _ in range(samples_internal):
        a = np.random.uniform(-inf_a, inf_a, (2))
        b = np.random.uniform(-inf_b, inf_b, (2))
        result += func(a, b)

    return result.reshape((3,2)) / Z_out(y, omega, V) / samples_internal



def Q_func(Q_hat, eps=1e-2):
    return Q_hat @ np.linalg.inv(np.eye(Q_hat.shape[0]) + Q_hat)


def Q_hat_func_MCMC(alpha, Q, samples, L=2, eps=1e-3):
    Q_hat = np.zeros(Q.shape)

    real_samples = 0
    for _ in tqdm(range(samples)):

        try:
            Z = np.random.normal(0,1, (Q.shape[0], L))
            U = np.random.normal(0,1, (Q.shape[0], L))

            sqrt_Q = sqrtm(Q)
            sqrt_one_minus_Q = sqrtm(np.eye(Q.shape[0]) - Q + eps*np.eye(Q.shape[0]))
            omega = sqrt_Q @ Z
            
            Z_true = sqrt_Q@Z + sqrt_one_minus_Q@U
            y = g(Z_true[0], Z_true[1], Z_true[2])

            V = np.eye(Q.shape[0]) - Q + eps*np.eye(Q.shape[0])

            g_out_mat = g_out(y, omega, V)
            value = alpha * g_out_mat @ g_out_mat.T

            Q_hat += value
            real_samples += 1
        except:
            pass
    
    if real_samples == 0:
        raise ValueError("No valid samples")

    return Q_hat / real_samples


def main(alpha, Q, samples, iter, damping=0.6):
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
            Q_hat_all = np.zeros((size, 3, 3), dtype=np.float64)

            Q_hat_all[0] = Q_hat
            for j in range(1, size):
                Q_hat_all[j] = comm.recv(source=j)

            Q_hat = np.mean(Q_hat_all, axis=0)

            Q = damping*Q_func(Q_hat) + (1-damping)*Q

            Q_list.append(Q)

            np.save(f"data_BO_3w/Q_list_alpha_{alpha}_samples_{int(size*samples)}_skip_{skip}.npy", Q_list)

            print(f"iter {it}: {Q}")

            Q = (Q + Q.T) / 2
            for j in range(1, size):
                comm.send(Q, dest=j)



        

if __name__=="__main__":
    alpha = float(sys.argv[1])
    iter = int(sys.argv[2])
    samples = int(sys.argv[3])

    Q = np.eye(3) * 0.05

    main(alpha, Q, samples, iter)