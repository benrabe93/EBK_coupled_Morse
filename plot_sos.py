"""Create and plot various surfaces of section for the coupled Morse oscillators."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path
from aux_functions import morse_pot, compute_energy
from run_trajectory import run_trajectory_and_get_sos


def bounds_morse(x, slope, intsec, D1, a1, D2, a2, E):
    """Auxiliary function for finding initial conditions at specified angle for Morse potential"""
    y = slope*x + intsec
    return morse_pot(x, D1, a1) + morse_pot(y, D2, a2) - E



def main():
    system = 'morse'
    coupling = 'kinetic'

    if system == 'morse': # Morse oscillator
        path = 'data/coupled_MO/'
        
        # Morse parameters
        # D1 = 30.0; a1 = 0.08
        # D1 = 32.0; a1 = 1/12; dt = 0.01
        D1 = 32.0; a1 = 1.0/5; dt = 0.005
        # D1 = 32.0; a1 = 1/3; dt = 0.001
        # D1 = 32.0; a1 = 1/2; dt = 0.0005
        # D1 = 32.0; a1 = 1.0; dt = 0.0001
        D2 = D1; a2 = a1
        omega1 = a1*np.sqrt(2*D1); omega2 = a2*np.sqrt(2*D2)
        
        max_n_t = 1e7 # Maximum time steps for trajectory
        n_x = 100
        E_total = np.array([0.5]) * D1; k = np.linspace(0.0, 1.0, 9)[:-1] # Coupling parameter
        # k = [0.5]; E_total = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) * D1
        angle_ini = (np.linspace(0.25, 1.25, n_x+2)[1:-1] * np.pi) # Angle of initial starting point for p_1 = p_2 = 0
        # angle_ini = (np.linspace(0.25, 2.25, n_x+2)[1:-1] * np.pi)
        
    Path(path).mkdir(parents=True, exist_ok=True)


    sos_ = 0 # 1 - Create surfaces of section data; 0 - Load surfaces of section data
    if sos_:
        data = [] # Collect sos points for different initial conditions: (E/k, angle_ini)
        for k_ in range(len(k)):
            data_k = np.empty((0, 2))
            
            for e_ in range(len(E_total)):
                data_e = np.empty((0, 2))
                
                for x_ in range(len(angle_ini)):
                    
                    # Find initial conditions
                    if angle_ini[x_] < 0.5*np.pi or angle_ini[x_] > 1.5*np.pi:
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
                    elif 0.5*np.pi < angle_ini[x_] < 1.5*np.pi:
                        if system == 'harmonic':
                            x_ini = -np.sqrt(2*E_total[e_]/(omega1**2 + omega2**2*np.tan(angle_ini[x_])**2))
                        elif system == 'morse':
                            x_ini = brentq(bounds_morse, -np.log(1 + np.sqrt(E_total[e_]/D1))/a1, 0.0, args=(np.tan(angle_ini[x_]), 0.0, D1, a1, D2, a2, E_total[e_]))
                        y_ini = x_ini * np.tan(angle_ini[x_]) # y = x*m + n
                    elif angle_ini[x_] == 1.5*np.pi:
                        x_ini = 0.0
                        if system == 'harmonic':
                            y_ini = -np.sqrt(2*E_total[e_])/omega2
                        elif system == 'morse':
                            y_ini = -np.log(1 + np.sqrt(E_total[e_]/D2))/a2
                    
                    # Find surfaces of section
                    px_ini = 0.0; py_ini = 0.0; energy_ini = compute_energy(x_ini, y_ini, px_ini, py_ini, k[k_], omega1, omega2, D1, a1, D2, a2, system, coupling)
                    print(f'{system, coupling}: k = {k[k_]:.4f}; x_ = {x_}; angle_ini = {angle_ini[x_]/np.pi}; (x, y, p_x, p_y) = ({x_ini:.4f}, {y_ini:.4f}, {px_ini:.4f}, {py_ini:.4f}); E = {energy_ini}')
                    slopes = np.array([1.0, 0.0, -1.0, 0.0])
                    x, y, px, py, energy, sos1 = run_trajectory_and_get_sos(x_ini, y_ini, px_ini, py_ini, energy_ini, dt, k[k_], omega1, omega2, D1, a1, D2, a2, system, coupling, slopes, max_n_t=max_n_t)
        
                    plot_ = 0
                    if plot_:
                        slope1, intsec1, slope2, intsec2 = slopes
                        buffer = 0.5
                        x_plot = np.linspace(-np.log(1 + np.sqrt(E_total[e_]/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total[e_]/D1))/a1 + buffer, 100)
                        y_plot = np.linspace(-np.log(1 + np.sqrt(E_total[e_]/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total[e_]/D2))/a2 + buffer, 100)
                        X, Y = np.meshgrid(x_plot, y_plot)
                        Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)
                        fig = plt.figure()
                        ax1 = fig.add_subplot(111)
                        ax1.set_aspect('equal', adjustable='box')
                        CS = plt.contour(X, Y, Z)
                        plt.clabel(CS, inline=True, fontsize=10)
                        plt.plot(x, y, lw=0.5)#, marker='x')
                        plt.scatter(x[0], y[0], s=100, marker='x', color='black', label='Starting point')
                        if np.isnan(slope1):
                            plt.axhline(y=intsec1, color='black')
                            # plt.axvline(x=intsec2, color='black')
                        else:
                            plt.plot(x_plot, slope1*x_plot + intsec1, color='black')
                            # plt.plot(x_plot, slope2*x_plot + intsec2, color='black')
                        plt.axis([np.min(x_plot), np.max(x_plot), np.min(y_plot), np.max(y_plot)])
                        plt.show()

                    if len(sos1) > 0:
                        if len(E_total) > 1:
                            data_e = np.concatenate((data_e, np.asarray(sos1)), axis=0)
                        else:
                            data_k = np.concatenate((data_k, np.asarray(sos1)), axis=0)
                            
                if len(E_total) > 1:
                    data.append(data_e)
                    
            if len(E_total) == 1:
                data.append(data_k)
                
        np.save(path+'sos_data.npy', np.asarray(data, dtype=object))
        
    else:
        if len(E_total) > 1:
            data = np.load(path+'sos_data_k=0_6_1_5.npy', allow_pickle=True)
        else:
            data = np.load(path+'sos_data_E=0_5_1_5.npy', allow_pickle=True)


    plot_ = 1
    if plot_:
        from get_LaTex_plot_settings import get_tex_fonts, get_size
        plt.rcParams.update(get_tex_fonts('thesis'))

        subset_ind = 4
        markersize = 0.2
        markers = '.'
        
        if len(E_total) > 1:
            legend_x_shift = 1.01; legend_y_shift = 1.03
        else:
            legend_x_shift = 1.0; legend_y_shift = 1.0
        
        labels = []
        for i in range(len(data)):
            if len(E_total) > 1:
                labels.append(f'$E = {E_total[i]/D1:.1f} \, D_e$')
            else:
                labels.append(rf'$\lambda = {k[i]}$')

        fig, ax = plt.subplots(4, 2, figsize=get_size('thesis', fraction=1.0, subplots=(4, 2)))
        if len(data[6]) > 1:
            ax[0,0].scatter(data[6][::subset_ind,0]*a1, data[6][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[6])
            ax[0,0].scatter(data[6][::subset_ind,0]*a1, -data[6][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        # ax[0,0].set_xlabel(r'$\tilde{q} \, [1/a]$')
        ax[0,0].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        ax[0,0].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        if len(data[7]) > 1:
            ax[0,1].scatter(data[7][::subset_ind,0]*a1, data[7][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[7])
            ax[0,1].scatter(data[7][::subset_ind,0]*a1, -data[7][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        # ax[0,1].set_xlabel(r'$\tilde{q} \, [1/a]$')
        # ax[0,1].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        ax[0,1].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        if len(data[4]) > 1:
            ax[1,0].scatter(data[4][::subset_ind,0]*a1, data[4][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[4])
            ax[1,0].scatter(data[4][::subset_ind,0]*a1, -data[4][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        # ax[1,0].set_xlabel(r'$\tilde{q} \, [1/a]$')
        ax[1,0].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        ax[1,0].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        if len(data[5]) > 1:
            ax[1,1].scatter(data[5][::subset_ind,0]*a1, data[5][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[5])
            ax[1,1].scatter(data[5][::subset_ind,0]*a1, -data[5][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        # ax[1,1].set_xlabel(r'$\tilde{q} \, [1/a]$')
        # ax[1,1].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        ax[1,1].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        if len(data[2]) > 1:
            ax[2,0].scatter(data[2][::subset_ind,0]*a1, data[2][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[2])
            ax[2,0].scatter(data[2][::subset_ind,0]*a1, -data[2][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        # ax[2,0].set_xlabel(r'$\tilde{q} \, [1/a]$')
        ax[2,0].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        # ax[2,0].set_yticks([-5, 0, 5])
        ax[2,0].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        if len(data[3]) > 1:
            ax[2,1].scatter(data[3][::subset_ind,0]*a1, data[3][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[3])
            ax[2,1].scatter(data[3][::subset_ind,0]*a1, -data[3][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        # ax[2,1].set_xlabel(r'$\tilde{q} \, [1/a]$')
        # ax[2,1].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        # ax[2,1].set_yticks([-5, 0, 5])
        ax[2,1].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        if len(data[0]) > 1:
            ax[3,0].scatter(data[0][::subset_ind,0]*a1, data[0][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[0])
            ax[3,0].scatter(data[0][::subset_ind,0]*a1, -data[0][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        ax[3,0].set_xlabel(r'$\tilde{q} \, [1/a]$')
        ax[3,0].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        # ax[3,0].set_yticks([-5, 0, 5])
        ax[3,0].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        if len(data[1]) > 1:
            ax[3,1].scatter(data[1][::subset_ind,0]*a1, data[1][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0', label=labels[1])
            ax[3,1].scatter(data[1][::subset_ind,0]*a1, -data[1][::subset_ind,1]/D1*a1, marker=markers, s=markersize, edgecolors='none', color='C0')
        ax[3,1].set_xlabel(r'$\tilde{q} \, [1/a]$')
        # ax[3,1].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
        # ax[3,1].set_yticks([-5, 0, 5])
        ax[3,1].legend(loc=1, bbox_to_anchor=(legend_x_shift, legend_y_shift), frameon=False)
        
        fig.tight_layout()
        fig.savefig('plots/morse_sos.png', dpi=300, format='png')
        # plt.show()

main()
