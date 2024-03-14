"""Create and diagonalize the Hamiltonian matrix for two coupled Morse oscillators using the finite differences method."""

import numpy as np
from aux_functions import morse_pot
# from scipy import sparse
# from scipy.sparse.linalg import eigsh
import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import eigsh
import sys

from pathlib import Path
Path("data/coupled_MO").mkdir(parents=True, exist_ok=True)


def create_H_m_0(x, y, xy_grid, k, D, a, symm=0, decoup=0):
    """
    Hamiltonian matrix for two uncoupled Morse oscillators: H = H_1 + H_2
    using the finite differences method
    where H_i = P_i^2/2 + V_mor_i(X_i)
    and Morse potential V_mor(x) = D*(1 - exp(-a*x))^2
    """
    
    len_x = len(x); dx = x[1] - x[0]
    len_y = len(y); dy = y[1] - y[0]
    len_xy = len(xy_grid)
    T_m_x = (-2*sparse.eye(len_x) + sparse.eye(len_x, k=1) + sparse.eye(len_x, k=-1))/dx**2
    
    if decoup == 1 and symm == 1:
        T_m_y = ((-2*sparse.eye(len_y) + sparse.eye(len_y, k=1) + sparse.eye(len_y, k=-1))/dy**2).tocsr()
        T_m_y[0,0] = T_m_y[0,0] + symm/dy**2
    else:
        T_m_y = (-2*sparse.eye(len_y) + sparse.eye(len_y, k=1) + sparse.eye(len_y, k=-1))/dy**2
    # T_m = (-5/2*sparse.eye(len_x) + 4/3*sparse.eye(len_x, k=1) + 4/3*sparse.eye(len_x, k=-1) - 1/12*sparse.eye(len_x, k=2) - 1/12*sparse.eye(len_x, k=-2))/dx**2
    
    if decoup == 0:
        H_x = -T_m_x/2 + sparse.spdiags(cp.array(morse_pot(x, D, a)), 0, len_x, len_x)
        H_y = -T_m_y/2 + sparse.spdiags(cp.array(morse_pot(y, D, a)), 0, len_y, len_y)
        return sparse.kron(H_y, sparse.eye(len_x)) + sparse.kron(sparse.eye(len_y), H_x)
    elif decoup == 1:
        T_x = sparse.kron(sparse.eye(len_y), -(1 - k)*T_m_x/2)
        T_y = sparse.kron(-(1 + k)*T_m_y/2, sparse.eye(len_x))
        V_m = sparse.spdiags(cp.array(morse_pot((xy_grid[:,0] - xy_grid[:,1])/np.sqrt(2), D, a) + morse_pot((xy_grid[:,0] + xy_grid[:,1])/np.sqrt(2), D, a)), 0, len_xy, len_xy)
        return T_x + T_y + V_m


def create_H_m_12(x, y):
    """
    Hamiltonian matrix for the kinetic coupling term H_12 = -P_1*P_2
    using the finite differences method
    """
    
    len_x = len(x); dx = x[1] - x[0]
    len_y = len(y); dy = y[1] - y[0]
    P_m_x = (0.5*sparse.eye(len_x, k=1) - 0.5*sparse.eye(len_x, k=-1))/dx # d/dx
    P_m_y = (0.5*sparse.eye(len_y, k=1) - 0.5*sparse.eye(len_y, k=-1))/dy # d/dy
    # P_m_x = (2/3*sparse.eye(len_x, k=1) - 2/3*sparse.eye(len_x, k=-1) - 1/12*sparse.eye(len_x, k=2) + 1/12*sparse.eye(len_x, k=-2))/dx
    # P_m_y = (2/3*sparse.eye(len_y, k=1) - 2/3*sparse.eye(len_y, k=-1) - 1/12*sparse.eye(len_y, k=2) + 1/12*sparse.eye(len_y, k=-2))/dy
    return sparse.kron(P_m_y, P_m_x)



def main():
    # Input parameters
    n_matrices = 251 # Number of Hamiltonian matrices
    symm = 1 # 1 = use symmetrized basis ; 2 = use anti-symmetrized basis
    decoup = 1 # kinetically decoupled by rotation

    k = np.linspace(0, 0.9, n_matrices)#[:-1] # Coupling parameter 1/M

    # Morse parameters
    # D = 32.0; a = 1.0/12; max_eigvals = 50; min_x = -7; max_x = 9
    D = 32.0; a = 1.0/5; max_eigvals = 300; min_x = -5; max_x = 7
    # D = 32.0; a = 1.0/3; max_eigvals = 35; min_x = -4; max_x = 7
    
    # Grid parameters
    if decoup == 0:
        x = np.linspace(min_x, max_x, 500)
        y = np.copy(x)
    elif decoup == 1:
        # x = np.linspace(-9, 12, 500)
        x = np.linspace(-5, 11, 2000)
        y_max = 12
        y_min = -y_max
        y_grid_size = 500 # even!
        if symm == 1 or symm == 2:
            y = np.linspace(y_min, y_max, 2*y_grid_size)[y_grid_size:]
        else:
            y = np.linspace(y_min, y_max, 2*y_grid_size)
            
    grid = np.meshgrid(x, y)
    xy_grid = np.asarray(list(zip(grid[0].ravel(), grid[1].ravel())))
    
    # Create Hamiltonian matrices
    H_0 = create_H_m_0(x, y, xy_grid, k[0], D, a, symm, decoup)
    if decoup == 0:
        H_12 = create_H_m_12(x, y)
    # print(H_0.shape)
    # print(np.sum(np.abs(H_0 - H_0.T)))


    # True Morse oscillator eigenenergies
    # ew, ev = np.linalg.eigh(H_0)
    # ew, ev = eigh(H_0, subset_by_index=[0,max_eigvals-1])
    # ew, ev = eigsh(H_0, k=max_eigvals, which="SM")
    # print(ew[ew < D])
    # plt.plot(x, morse_pot(x, D, a))
    # for i in range(max_eigvals):
    #     plt.plot(x, ev[:,i]+ew[i])
    # plt.show()

    # print(np.sort(np.sum(a*np.sqrt(2*D)*(np.asarray(basis[:n_max]) + 0.5) - a**2/2*(np.asarray(basis[:n_max]) + 0.5)**2, axis=-1)))
    # morse_energies = a*np.sqrt(2*D)*(np.arange(n_max) + 0.5) - a**2/2*(np.arange(n_max) + 0.5)**2
    # print(morse_energies)


    # Diagonalize Hamiltonian matrices
    spectra = np.zeros([min(H_0.shape[0], max_eigvals), len(k)])
    # eigenvecs = np.zeros([H_0.shape[0], min(H_0.shape[0], max_eigvals), len(k)])
    for i in range(len(k)):
        sys.stdout.write(f"\rk = {k[i]}"); sys.stdout.flush()
        if decoup == 0:
            H_m = H_0 + k[i]*H_12
        elif decoup == 1:
            H_m = create_H_m_0(x, y, xy_grid, k[i], D, a, symm, decoup)
        ew = eigsh(H_m, k=max_eigvals, which="SA", tol=1e-3, return_eigenvectors=False).get()
        # ew, ev = eigsh(H_m, k=max_eigvals, which="SA", tol=1e-3); ew = ew.get(); ev = ev.get()
        # eigenvecs[:,:,i] = ev # np.real()
        spectra[:,i] = np.sort(ew)

    if symm == 0:
        np.save("data/coupled_MO/spectra_grid.npy", spectra)
        # np.save("data/coupled_MO/eigenvecs_grid.npy", eigenvecs)
    elif symm == 1:
        np.save("data/coupled_MO/spectra_grid_sym.npy", spectra)
        # np.save("data/coupled_MO/eigenvecs_grid_sym.npy", eigenvecs)
    elif symm == 2:
        np.save("data/coupled_MO/spectra_grid_asym.npy", spectra)
        # np.save("data/coupled_MO/eigenvecs_grid_asym.npy", eigenvecs)


    plot_ = 0
    if plot_:
        n = 200

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.plot_surface(grid[0], grid[1], ev[:,n].reshape(len(y), len(x)))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel(r'$\psi$')

        ax2 = fig.add_subplot(222, projection='3d')
        if decoup == 0:
            ax2.plot_surface(grid[0], grid[1], (morse_pot(xy_grid[:,0], D, a) + morse_pot(xy_grid[:,1], D, a)).reshape(len(y), len(x)))
        elif decoup == 1:
            ax2.plot_surface(grid[0], grid[1], (morse_pot((xy_grid[:,0] - xy_grid[:,1])/np.sqrt(2), D, a) + morse_pot((xy_grid[:,0] + xy_grid[:,1])/np.sqrt(2), D, a)).reshape(len(y), len(x)))
        # ax2.set_zlim([-1,0])
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')
        ax2.set_zlabel('$E$')

        ax3 = fig.add_subplot(223)
        if decoup == 0:
            ax3.plot(x, ev[:,n].reshape(len(y), len(x))[round(0.5*len(y)),:])
        elif decoup == 1:
            ax3.plot(x, ev[:,n].reshape(len(y), len(x))[0,:])
        ax3.set_xlabel('$x$')
        ax3.set_ylabel(r'$\psi$')
        
        ax4 = fig.add_subplot(224)
        ax4.plot(y, ev[:,n].reshape(len(y), len(x))[:,round(0.5*len(x))])
        ax4.set_xlabel('$y$')
        ax4.set_ylabel(r'$\psi$')
        plt.show()


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid[0], grid[1], ev[:,n].reshape(len(y), len(x)))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel(r'$\psi$')
        plt.show()


main()

