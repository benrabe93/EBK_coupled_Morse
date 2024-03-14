"""Create and diagonalize the Hamiltonian matrix for two coupled Morse oscillators in a Morse oscillator eigenbasis."""

import numpy as np
from aux_functions import sqrt_divide_factorials, sqrt_divide_product_of_factorials, morse_pot
import sys

from pathlib import Path
Path("data/coupled_MO").mkdir(parents=True, exist_ok=True)


def create_H_0(basis, D, a, k, symm=0):
    """
    Hamiltonian matrix for two uncoupled Morse oscillators: H = H_1 + H_2
    in the Morse oscillator eigenbasis
    where H_i = P_i^2/2 + V_mor_i(X_i)
    and Morse potential V_mor(x) = D*(1 - exp(-a*x))^2
    Phys. Rev. A 32, 538 (1985)
    """
    
    if symm == 0:
        v_1 = basis[:,0]; v_2 = basis[:,1]
        e_v1 = D*(2*(v_1 + 0.5)/k - (v_1 + 0.5)**2/k**2)
        e_v2 = D*(2*(v_2 + 0.5)/k - (v_2 + 0.5)**2/k**2)
        H_0 = np.diag(e_v1 + e_v2)
        
    else:
        len_basis = len(basis)
        H_0 = np.zeros([len_basis, len_basis])
        
        for i in range(len_basis):
            v_1, v_2 = basis[i]
            e_v1 = D*(2*(v_1 + 0.5)/k - (v_1 + 0.5)**2/k**2)
            e_v2 = D*(2*(v_2 + 0.5)/k - (v_2 + 0.5)**2/k**2)
            
            for j in range(len_basis):
                w_1, w_2 = basis[j]
                
                if (v_1 == w_1) and (v_2 == w_2):
                    H_12_12 = e_v1 + e_v2
                else:
                    H_12_12 = 0.0
                
                if (v_1 == w_2) and (v_2 == w_1):
                    H_12_21 = e_v1 + e_v2
                else:
                    H_12_21 = 0.0
                
                if symm == 1:
                    if (v_1 == v_2) and (w_1 == w_2):
                        H_0[i,j] = H_12_12
                    elif (v_1 == v_2) or (w_1 == w_2):
                        H_0[i,j] = (H_12_12 + H_12_21)/np.sqrt(2)
                    else:
                        H_0[i,j] = H_12_12 + H_12_21
                
                elif symm == 2:
                    # if (v_1 != v_2) and (w_1 != w_2):
                    H_0[i,j] = H_12_12 - H_12_21
        
    return H_0


def create_H_12(basis, D, a, k, symm=0):
    """
    Hamiltonian matrix for the kinetic coupling term H_12 = -P_1*P_2
    in the Morse oscillator eigenbasis
    Phys. Rev. A 32, 538 (1985)
    """
    
    len_basis = len(basis)
    H_12 = np.zeros([len_basis, len_basis])

    for i in range(len_basis):
        v_1, v_2 = basis[i]
        if isinstance(k, int):
            N_v1 = np.sqrt(2*k - 2*v_1 - 1)
            N_v2 = np.sqrt(2*k - 2*v_2 - 1)
        # else:
        #     N_v1 = np.sqrt((2*k - 2*v_1 - 1) * np.math.factorial(v_1) / gamma(2*k - v_1))
        #     N_v2 = np.sqrt((2*k - 2*v_2 - 1) * np.math.factorial(v_2) / gamma(2*k - v_2))

        # for j in range(len_basis):
        for j in range(i+1): # Only lower triangle
            w_1, w_2 = basis[j]
            
            if v_1 > w_1: # Ensure minus sign coming from adjoint operator d/dx
                sign_1 = 1.0
            else:
                sign_1 = -1.0
                
            if v_2 > w_2:
                sign_2 = 1.0
            else:
                sign_2 = -1.0
            
            if isinstance(k, int):
                N_w1 = np.sqrt(2*k - 2*w_1 - 1)
                N_w2 = np.sqrt(2*k - 2*w_2 - 1)
            # else:
            #     N_w1 = np.sqrt((2*k - 2*w_1 - 1) * np.math.factorial(w_1) / gamma(2*k - w_1))
            #     N_w2 = np.sqrt((2*k - 2*w_2 - 1) * np.math.factorial(w_1) / gamma(2*k - w_2))
            
            # if (v_1 != w_1) and (v_2 != w_2):
            #     if isinstance(k, int):
            #         v1_d_dr_w1 = N_v1*N_w1/2 * (-1.0)**(v_1-w_1) * sqrt_divide_factorials(v_1, w_1) * sqrt_divide_factorials(2*k - v_1 - 1, 2*k - w_1 - 1) * a
            #         v2_d_dr_w2 = N_v2*N_w2/2 * (-1.0)**(v_2-w_2) * sqrt_divide_factorials(v_2, w_2) * sqrt_divide_factorials(2*k - v_2 - 1, 2*k - w_2 - 1) * a
            #     else:
            #         v1_d_dr_w1 = N_v1*N_w1 * (-1.0)**(v_1-w_1) * gamma(2*k - v_1) / (2.0*np.math.factorial(w_1)) * a
            #         v2_d_dr_w2 = N_v2*N_w2 * (-1.0)**(v_2-w_2) * gamma(2*k - v_2) / (2.0*np.math.factorial(w_2)) * a
            #     H_12[i,j] = v1_d_dr_w1 * v2_d_dr_w2
            
            if (v_1 != w_1) and (v_2 != w_2):
                m_1, n_1 = np.sort([v_1, w_1]); m_2, n_2 = np.sort([v_2, w_2])
                if isinstance(k, int):
                    # v1_d_dr_w1 = N_v1*N_w1/2 * (-1.0)**(n_1-m_1) * sqrt_divide_factorials(n_1, m_1) * sqrt_divide_factorials(2*k - n_1 - 1, 2*k - m_1 - 1) * a
                    # v2_d_dr_w2 = N_v2*N_w2/2 * (-1.0)**(n_2-m_2) * sqrt_divide_factorials(n_2, m_2) * sqrt_divide_factorials(2*k - n_2 - 1, 2*k - m_2 - 1) * a
                    # H_12_12 = sign_1 * sign_2 * v1_d_dr_w1 * v2_d_dr_w2
                    H_12_12 = (sign_1 * sign_2 * N_v1*N_w1/2 * (-1.0)**(n_1-m_1+n_2-m_2) * N_v2*N_w2/2 * a**2 * 
                               sqrt_divide_product_of_factorials(np.array([n_1, n_2, 2*k - n_1 - 1, 2*k - n_2 - 1]), np.array([m_1, m_2, 2*k - m_1 - 1, 2*k - m_2 - 1])))
                # else:
                #     v1_d_dr_w1 = N_v1*N_w1 * (-1.0)**(n_1-m_1) * gamma(2*k - n_1) / (2.0*np.math.factorial(m_1)) * a
                #     v2_d_dr_w2 = N_v2*N_w2 * (-1.0)**(n_2-m_2) * gamma(2*k - n_2) / (2.0*np.math.factorial(m_2)) * a
                #     H_12_12 = sign_1 * sign_2 * v1_d_dr_w1 * v2_d_dr_w2
            else:
                H_12_12 = 0
            
            if symm == 0:
                H_12[i,j] = H_12_12
            else:
                if (v_1 != w_2) and (v_2 != w_1):
                    
                    if v_1 > w_2: # Ensure minus sign coming from adjoint operator d/dx
                        sign_1_a = 1.0
                    else:
                        sign_1_a = -1.0
                        
                    if v_2 > w_1:
                        sign_2_a = 1.0
                    else:
                        sign_2_a = -1.0
                    
                    m_1, n_1 = np.sort([v_1, w_2]); m_2, n_2 = np.sort([v_2, w_1])
                    if isinstance(k, int):
                        # v1_d_dr_w2 = N_v1*N_w2/2 * (-1.0)**(n_1-m_1) * sqrt_divide_factorials(n_1, m_1) * sqrt_divide_factorials(2*k - n_1 - 1, 2*k - m_1 - 1) * a
                        # v2_d_dr_w1 = N_v2*N_w1/2 * (-1.0)**(n_2-m_2) * sqrt_divide_factorials(n_2, m_2) * sqrt_divide_factorials(2*k - n_2 - 1, 2*k - m_2 - 1) * a
                        # H_12_21 = sign_1_a * sign_2_a * v1_d_dr_w2 * v2_d_dr_w1
                        H_12_21 = (sign_1_a * sign_2_a * N_v1*N_w2/2 * (-1.0)**(n_1-m_1+n_2-m_2) * N_v2*N_w1/2 * a**2 * 
                                   sqrt_divide_product_of_factorials(np.array([n_1, n_2, 2*k - n_1 - 1, 2*k - n_2 - 1]), np.array([m_1, m_2, 2*k - m_1 - 1, 2*k - m_2 - 1])))
                else:
                    H_12_21 = 0
                    
                if symm == 1:
                    if (v_1 == v_2) and (w_1 == w_2):
                        H_12[i,j] = H_12_12
                    elif (v_1 == v_2) or (w_1 == w_2):
                        H_12[i,j] = (H_12_12 + H_12_21)/np.sqrt(2)
                    else:
                        H_12[i,j] = H_12_12 + H_12_21
                    
                elif symm == 2:
                    # if (v_1 != v_2) and (w_1 != w_2):
                    H_12[i,j] = H_12_12 - H_12_21
            
    return H_12



def main():
    # Input parameters
    n_matrices = 1001 # Number of Hamiltonian matrices
    n_max = 100 # Maximum vibrational quantum number
    symm = 1 # 1 = use symmetrized basis ; 2 = use anti-symmetrized basis

    k = np.linspace(0, 1, n_matrices)[:-1] # Coupling parameter 1/M
    # np.save("data/coupled_MO/k.npy", k)

    # Morse parameters
    # D = 150; a = 0.288
    # D = 30; a = 0.08
    # k_param = np.sqrt(2*D)/a

    # D = 32; a = 1/15; k_param = 120
    # D = 32.0; a = 1.0/12; k_param = 96; max_eigvals = 100
    D = 32.0; a = 1.0/5; k_param = 40; max_eigvals = 100
    # D = 32.0; a = 1.0/3; k_param = 24; max_eigvals = 70
    # D = 32; a = 1/2; k_param = 16
    # D = 32; a = 1; k_param = 8
    # D = 24.5; a = 1/15; k_param = 105
    print(f"Max quantum number: {int(k_param - 0.5)}")

    # Create Morse oscillator basis quantum numbers
    from itertools import product
    basis = np.array(list(product(np.arange(min(n_max, int(k_param - 0.5)) + 1), repeat=2)))
    if symm == 1:
        basis = np.unique(np.sort(basis, axis=-1), axis=0)
    elif symm == 2:
        basis = np.unique(np.sort(basis, axis=-1), axis=0)
        basis = np.delete(basis, np.where(basis[:,0] == basis[:,1])[0], axis=0)
    # print(len(basis))
    # print(basis)

    # Create Hamiltonian matrices
    H_0 = create_H_0(basis, D, a, k_param, symm)
    H_12 = create_H_12(basis, D, a, k_param, symm)
    print(H_12.shape)
    # print(np.sum(np.abs(H_0 - H_0.T)))
    # print(np.sum(np.abs(H_12 - H_12.T)))
    # print(np.max(H_12), np.min(H_12))
    

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
    spectra = np.zeros([min(H_12.shape[0], max_eigvals), len(k)])
    eigenvecs = np.zeros([H_12.shape[0], min(H_12.shape[0], max_eigvals), len(k)])
    for i in range(len(k)):
        sys.stdout.write(f"\rk = {k[i]}"); sys.stdout.flush()
        H_m = H_0 + k[i]*H_12
        ew, ev = np.linalg.eigh(H_m)
        spectra[:,i] = ew[:max_eigvals]
        eigenvecs[:,:,i] = ev[:,:max_eigvals]
    
    if symm == 0:
        np.save("data/coupled_MO/spectra.npy", spectra)
        np.save("data/coupled_MO/eigenvecs.npy", eigenvecs)
    elif symm == 1:
        np.save("data/coupled_MO/spectra_sym.npy", spectra)
        np.save("data/coupled_MO/eigenvecs_sym.npy", eigenvecs)
    elif symm == 2:
        np.save("data/coupled_MO/spectra_asym.npy", spectra)
        np.save("data/coupled_MO/eigenvecs_asym.npy", eigenvecs)


main()

