'''
Run many classical trajectories for varying initial conditions
for the system of two kinetically coupled Morse oscillators with Hamiltonian
H = p1^2/2 + p2^2/2 + V_Mor(x1) + V_Mor(x2) - k p1 p2,
compute the actions S_1 and S_2 for each trajectory, and save the actions
as well as the Poincare surfaces of section.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path
from aux_functions import compute_energy, harm_pot, morse_pot
from run_trajectory import run_trajectory_and_get_sections, run_trajectory_and_converge_sos
from plot_trajectory import plot_trajectory


def bounds_morse(x, slope, intsec, D1, a1, D2, a2, E):
    """Auxiliary function for finding initial conditions at specified angle for Morse potential"""
    y = slope*x + intsec
    return morse_pot(x, D1, a1) + morse_pot(y, D2, a2) - E



def main():
    # system = 'harmonic'
    system = 'morse'

    coupling = 'kinetic'
    # coupling = 'potential'

    if system == 'harmonic': # Harmonic oscillator
        path = 'data/coupled_HO/'
        
        # Morse parameters
        # D = 30.0; a = 0.08
        # D1 = 32.0; a1 = 1/12; dt = 0.01
        # D1 = 32.0; a1 = 1/3; dt = 0.001
        D1 = 32.0; a1 = 1.0/5; dt = 0.001
        D2 = D1; a2 = a1
        omega1 = a1*np.sqrt(2*D1)
        omega2 = a2*np.sqrt(2*D2)
        # omega1 = 1.0; omega2 = np.sqrt(2)
        
        max_n_t = 1e6 # Maximum time steps for trajectory
        n_k = 51
        n_E = 21
        n_x = 20
        symmetry = 1 # 0 = no symmetry ; 1 = symmetry (non-resonant)
        k = np.linspace(0, 1, n_k)[1:-1] # Coupling parameter
        E_total = np.linspace(0.5, 7.0, n_E) # Total energy
        angle_ini = (np.linspace(0.25, 1.25, n_x+2)[1:-1] * np.pi) # Angle of initial starting point for p_1 = p_2 = 0
        
        # k = [0.6]; E_total = np.linspace(3.0, 20.0, 21)[5:]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[4:] # 2:1 resonance
        # k = [0.8]; E_total = np.linspace(3.0, 20.0, 21)[5:]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[5:] # 3:1 resonance
        # k = [5.0/13]; E_total = np.linspace(3.0, 20.0, 21)[5:]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[5:] # 3:2 resonance

    elif system == 'morse': # Morse oscillator
        path = 'data/coupled_MO/'
        
        # Morse parameters
        # D1 = 30.0; a1 = 0.08
        # D1 = 32.0; a1 = 1/12; dt = 0.01
        # D1 = 32.0; a1 = 1/3; dt = 0.001
        # D1 = 32.0; a1 = 1/2; dt = 0.0005
        # D1 = 32.0; a1 = 1.0; dt = 0.0005
        D1 = 32.0; a1 = 1.0/5; dt = 0.001
        D2 = D1; a2 = a1
        omega1 = a1*np.sqrt(2*D1); omega2 = a2*np.sqrt(2*D2)
        
        max_n_t = 1e7 # Maximum time steps for trajectory
        n_k = 51
        n_E = 21
        n_x = 21
        symmetry = 0 # 0 = no symmetry ; 1 = symmetry (non-resonant)
        k = np.linspace(0, 1, n_k)[1:-1] # Coupling parameter
        E_total = np.linspace(1.0, 16.0, n_E) # Total energy
        # E_total = np.linspace(16.0, 31.0, n_E)[1:]
        angle_ini = (np.linspace(0.25, 1.25, n_x+2)[1:-1] * np.pi) # Angle of initial starting point for p_1 = p_2 = 0
        
        # k = [0.6]; E_total = np.linspace(3.0, 20.0, 21)[5:]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[2:] # Non-resonant trajectory
        # k = [0.6]; E_total = np.linspace(3.0, 20.0, 21)[5:]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[4:] # Resonant trajectory
        # k = [0.6]; E_total = np.linspace(20.0, 32.0, 21)[1:-1]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[4:] # Chaotic trajectory
        # k = [0.8]; E_total = np.linspace(3.0, 20.0, 21)[5:]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[5:] # 3:1 resonance
        # k = [5.0/13]; E_total = np.linspace(3.0, 20.0, 21)[5:]; angle_ini = (np.linspace(0.25, 1.25, 21+2)[1:-1] * np.pi)[5:] # 3:2 resonance
        
    Path(path).mkdir(parents=True, exist_ok=True)


    # Collect data for different initial conditions: [k, E, S_1, S_2, S_close, S_close_corr]
    data = np.zeros([len(k), 5, len(E_total), len(angle_ini)])
    slope_data = np.zeros([len(k), len(E_total), len(angle_ini), 2])
    
    for k_ in range(len(k)):
        S_1_list, S_2_list, S_close_list, S_close_corr_list = np.zeros([4, len(E_total), len(angle_ini)])

        for e_ in range(len(E_total)):
            for x_ in range(len(angle_ini)):
                
                # Find initial conditions
                if angle_ini[x_] < 0.5*np.pi:
                    if system == 'harmonic':
                        x_ini = np.sqrt(2*E_total[e_]/(omega1**2 + omega2**2*np.tan(angle_ini[x_])**2))
                    elif system == 'morse':
                        x_ini = brentq(bounds_morse, 0.0, -np.log(1 - np.sqrt(E_total[e_]/D1))/a1, args=(np.tan(angle_ini[x_]), 0.0, D1, a1, D2, a2, E_total[e_]))
                    y_ini = x_ini * np.tan(angle_ini[x_]) # y = x*m + n
                elif angle_ini[x_] == 0.5*np.pi:
                    x_ini = 0.0
                    if system == 'harmonic':
                        y_ini = np.sqrt(2*E_total[e_])/omega2
                    elif system == 'morse':
                        y_ini = -np.log(1 - np.sqrt(E_total[e_]/D2))/a2
                elif angle_ini[x_] > 0.5*np.pi:
                    if system == 'harmonic':
                        x_ini = -np.sqrt(2*E_total[e_]/(omega1**2 + omega2**2*np.tan(angle_ini[x_])**2))
                    elif system == 'morse':
                        x_ini = brentq(bounds_morse, -np.log(1 + np.sqrt(E_total[e_]/D1))/a1, 0.0, args=(np.tan(angle_ini[x_]), 0.0, D1, a1, D2, a2, E_total[e_]))
                    y_ini = x_ini * np.tan(angle_ini[x_]) # y = x*m + n
                
                # Find surfaces of section
                px_ini = 0.0; py_ini = 0.0; energy_ini = compute_energy(x_ini, y_ini, px_ini, py_ini, k[k_], omega1, omega2, D1, a1, D2, a2, system, coupling)
                print(f'{system, coupling}: k = {k[k_]:.4f}; x_ = {x_}; angle_ini = {angle_ini[x_]/np.pi}; (x, y, p_x, p_y) = ({x_ini:.4f}, {y_ini:.4f}, {px_ini:.4f}, {py_ini:.4f}); E = {energy_ini}')
                x, y, slopes, corners = run_trajectory_and_get_sections(x_ini, y_ini, px_ini, py_ini, energy_ini, dt, k[k_], omega1, omega2, D1, a1, D2, a2, system, coupling, n_t=1e6, symmetric_only=symmetry)
                # slopes = np.array([1.0, 0.0, -1.0, 0.0]); corners = []
                    
                slope_data[k_, e_, x_] = np.array([slopes[0], slopes[2]])

                if np.sum(np.isnan(slopes)) == 4:
                    S_1_list[e_,x_], S_2_list[e_,x_], S_close_list[e_,x_], S_close_corr_list[e_,x_] = [np.nan]*4
                    
                    plot_ = 0
                    if plot_:
                        buffer = 0.1
                        if system == 'harmonic':
                            x_plot = np.linspace(-np.sqrt(2*E_total[e_])/omega1 - buffer, np.sqrt(2*E_total[e_])/omega1 + buffer, 100)
                            y_plot = np.linspace(-np.sqrt(2*E_total[e_])/omega2 - buffer, np.sqrt(2*E_total[e_])/omega2 + buffer, 100)
                            X, Y = np.meshgrid(x_plot, y_plot)
                            Z = harm_pot(X, omega1) + harm_pot(Y, omega2)
                        elif system == 'morse':
                            x_plot = np.linspace(-np.log(1 + np.sqrt(E_total[e_]/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total[e_]/D1))/a1 + buffer, 100)
                            y_plot = np.linspace(-np.log(1 + np.sqrt(E_total[e_]/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total[e_]/D2))/a2 + buffer, 100)
                            X, Y = np.meshgrid(x_plot, y_plot)
                            Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)
                        CS = plt.contour(X, Y, Z)
                        plt.clabel(CS, inline=True, fontsize=10)
                        plt.plot(x, y, lw=0.5)
                        plt.axis([np.min(x_plot), np.max(x_plot), np.min(y_plot), np.max(y_plot)])
                        plt.show()

                else:
                    slope1, intsec1, slope2, intsec2 = slopes
                    print('Slopes:', slopes)
                    
                    plot_ = 0
                    if plot_:
                        buffer = 0.5
                        if system == 'harmonic':
                            x_plot = np.linspace(-np.sqrt(2*E_total[e_])/omega1 - buffer, np.sqrt(2*E_total[e_])/omega1 + buffer, 100)
                            y_plot = np.linspace(-np.sqrt(2*E_total[e_])/omega2 - buffer, np.sqrt(2*E_total[e_])/omega2 + buffer, 100)
                            X, Y = np.meshgrid(x_plot, y_plot)
                            Z = harm_pot(X, omega1) + harm_pot(Y, omega2)
                        elif system == 'morse':
                            x_plot = np.linspace(-np.log(1 + np.sqrt(E_total[e_]/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total[e_]/D1))/a1 + buffer, 100)
                            y_plot = np.linspace(-np.log(1 + np.sqrt(E_total[e_]/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total[e_]/D2))/a2 + buffer, 100)
                            X, Y = np.meshgrid(x_plot, y_plot)
                            Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)
                        fig = plt.figure()
                        ax1 = fig.add_subplot(111)
                        ax1.set_aspect('equal', adjustable='box')
                        CS = plt.contour(X, Y, Z)
                        plt.clabel(CS, inline=True, fontsize=10)
                        plt.plot(x, y, lw=0.5, marker='x')
                        plt.scatter(x[0], y[0], s=100, marker='x', color='black', label='Starting point')
                        if np.isnan(slope1):
                            plt.axhline(y=intsec1, color='black')
                            plt.axvline(x=intsec2, color='black')
                        else:
                            plt.plot(x_plot, slope1*x_plot + intsec1, color='black')
                            plt.plot(x_plot, slope2*x_plot + intsec2, color='black')
                        plt.axis([np.min(x_plot), np.max(x_plot), np.min(y_plot), np.max(y_plot)])
                        plt.show()

                    x, y, px, py, energy, sos1, sos2, sos1_area, sos2_area, S_close, S_close_corr = run_trajectory_and_converge_sos(x_ini, y_ini, px_ini, py_ini, energy_ini, dt, k[k_], omega1, omega2, D1, a1, D2, a2, system, coupling, slopes, max_n_t=max_n_t)
                    print(f'S_1 = {sos1_area:.4f} ; S_2 = {sos2_area:.4f} ; S_2_close = {((S_close + S_close_corr) - 1*sos1_area):.4f} ; S ratio = {sos1_area/sos2_area} ; n_sos1 = {len(sos1)} ; n_sos2 = {len(sos2)} ; n_t = {len(x)}')
                    S_1_list[e_,x_], S_2_list[e_,x_], S_close_list[e_,x_], S_close_corr_list[e_,x_] = [sos1_area, sos2_area, S_close, S_close_corr]
                    
                    plot_trajectory(x, y, px, py, energy, sos1, sos2, sos1_area, sos2_area, E_total[e_], k[k_], system, coupling, omega1, omega2, D1, a1, D2, a2, slopes, corners=corners)

        data[k_] = np.stack([np.tile(E_total, (len(angle_ini),1)).T, S_1_list, S_2_list, S_close_list, S_close_corr_list])

    if symmetry == 1:
        np.save(path+'ebk_traj_k_data_sym.npy', k)
        np.save(path+'ebk_traj_data_sym.npy', data)
        np.save(path+'ebk_traj_slopes_sym.npy', slope_data)
    else:
        np.save(path+'ebk_traj_k_data_asym.npy', k)
        np.save(path+'ebk_traj_data_asym.npy', data)
        np.save(path+'ebk_traj_slopes_asym.npy', slope_data)
    

main()
