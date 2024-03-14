"""Plot the trajectories and surfaces of section for the coupled Morse oscillators
as well as resonant trajectories of two coupled Morse and two coupled harmonic oscillators.
"""

import numpy as np
from aux_functions import morse_pot, harm_pot
import matplotlib.pyplot as plt
from matplotlib import cm
from get_LaTex_plot_settings import get_tex_fonts, get_size
plt.rcParams.update(get_tex_fonts('thesis'))


def main():
    path = 'data/coupled_MO/'
    
    # Morse parameters
    D1 = 32.0; a1 = 1.0/5; dt = 0.005
    D2 = D1; a2 = a1
    omega1 = a1*np.sqrt(2*D1); omega2 = a2*np.sqrt(2*D2)
    slopes = np.array([1.0, 0.0, -1.0, 0.0])
    slope1, intsec1, slope2, intsec2 = slopes
    
    # Load trajectory data
    x_nonres = np.load(path+'ebk_traj_x_nonres.npy'); y_nonres = np.load(path+'ebk_traj_y_nonres.npy'); sos1_nonres = np.array(np.load(path+'ebk_traj_sos1_nonres.npy'))
    x_res = np.load(path+'ebk_traj_x_res.npy'); y_res = np.load(path+'ebk_traj_y_res.npy'); sos1_res = np.array(np.load(path+'ebk_traj_sos1_res.npy'))
    x_chaos = np.load(path+'ebk_traj_x_chaos.npy'); y_chaos = np.load(path+'ebk_traj_y_chaos.npy'); sos1_chaos = np.array(np.load(path+'ebk_traj_sos1_chaos.npy'))
    E_total = np.array([np.linspace(3.0, 20.0, 21)[5], np.linspace(3.0, 20.0, 21)[5], np.linspace(20.0, 32.0, 21)[1]])
    
    buffer = 0.1
    linewidths = 2
    markersize = 2


    fig, ax = plt.subplots(2, 3, figsize=get_size('thesis', fraction=1.0, ratio='equal', subplots=(2, 3)))
    
    x_plot = np.linspace(-np.log(1 + np.sqrt(E_total[0]/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total[0]/D1))/a1 + buffer, 100)
    y_plot = np.linspace(-np.log(1 + np.sqrt(E_total[0]/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total[0]/D2))/a2 + buffer, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)
    CS = ax[0,0].contour(X*a1, Y*a1, Z/D1, linewidths=0.5, cmap=cm.Greens)
    ax[0,0].plot(x_nonres[:int(5e4)]*a1, y_nonres[:int(5e4)]*a1, lw=0.5, color='gray')
    # ax[0,0].plot(x_nonres[1500:15000]*a1, y_nonres[1500:15000]*a1, lw=1.0, color='black') # Show a few winds of trajectory
    if np.isnan(slope1):
        ax[0,0].axhline(y=intsec1, linestyle='--', lw=linewidths, color='C0')
    else:
        ax[0,0].plot(x_plot, slope1*x_plot + intsec1, linestyle='--', lw=linewidths, color='C0')
    ax[0,0].set_xlabel('$q_1 \, [1/a]$')
    ax[0,0].set_ylabel('$q_2 \, [1/a]$')
    ax[0,0].set_xlim([np.min(x_plot)*a1, np.max(x_plot)*a1])
    ax[0,0].set_ylim([np.min(y_plot)*a1, np.max(y_plot)*a1])
    ax[0,0].set_aspect('equal')

    x_plot = np.linspace(-np.log(1 + np.sqrt(E_total[1]/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total[1]/D1))/a1 + buffer, 100)
    y_plot = np.linspace(-np.log(1 + np.sqrt(E_total[1]/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total[1]/D2))/a2 + buffer, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)
    CS = ax[0,1].contour(X*a1, Y*a1, Z/D1, linewidths=0.5, cmap=cm.Greens)
    ax[0,1].plot(x_res[:int(5e4)]*a1, y_res[:int(5e4)]*a1, lw=0.5, color='gray')
    # ax[0,1].plot(x_res[1500:15000]*a1, y_res[1500:15000]*a1, lw=1.0, color='black') # Show a few winds of trajectory
    if np.isnan(slope1):
        ax[0,1].axhline(y=intsec1, linestyle='--', lw=linewidths, color='C0')
    else:
        ax[0,1].plot(x_plot, slope1*x_plot + intsec1, linestyle='--', lw=linewidths, color='C0')
    ax[0,1].set_xlabel('$q_1 \, [1/a]$')
    # ax[0,1].set_ylabel('$q_2 \, [1/a]$')
    ax[0,1].set_xlim([np.min(x_plot)*a1, np.max(x_plot)*a1])
    ax[0,1].set_ylim([np.min(y_plot)*a1, np.max(y_plot)*a1])
    ax[0,1].set_aspect('equal')

    x_plot = np.linspace(-np.log(1 + np.sqrt(E_total[2]/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total[2]/D1))/a1 + buffer, 100)
    y_plot = np.linspace(-np.log(1 + np.sqrt(E_total[2]/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total[2]/D2))/a2 + buffer, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)
    CS = ax[0,2].contour(X*a1, Y*a1, Z/D1, linewidths=0.5, cmap=cm.Greens)
    ax[0,2].plot(x_chaos[:int(8e4)]*a1, y_chaos[:int(8e4)]*a1, lw=0.5, color='gray')
    # ax[0,2].plot(x_chaos[1500:15000]*a1, y_chaos[1500:15000]*a1, lw=1.0, color='black') # Show a few winds of trajectory
    if np.isnan(slope1):
        ax[0,2].axhline(y=intsec1, linestyle='--', lw=linewidths, color='C0')
    else:
        ax[0,2].plot(x_plot, slope1*x_plot + intsec1, linestyle='--', lw=linewidths, color='C0')
    ax[0,2].set_xlabel('$q_1 \, [1/a]$')
    # ax[0,2].set_ylabel('$q_2 \, [1/a]$')
    ax[0,2].set_xlim([np.min(x_plot)*a1, np.max(x_plot)*a1])
    ax[0,2].set_ylim([np.min(y_plot)*a1, np.max(y_plot)*a1])
    ax[0,2].set_aspect('equal')

    ax[1,0].scatter(sos1_nonres[:,0]*a1, sos1_nonres[:,1]/D1*a1, marker='.', s=markersize, color='C0', edgecolors='none')
    ax[1,0].set_xlabel(r'$\tilde{q} \, [1/a]$')
    ax[1,0].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
    
    ax[1,1].scatter(sos1_res[:,0]*a1, sos1_res[:,1]/D1*a1, marker='.', s=markersize, color='C0', edgecolors='none')
    ax[1,1].set_xlabel(r'$\tilde{q} \, [1/a]$')
    # ax[1,1].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
    
    ax[1,2].scatter(sos1_chaos[:,0]*a1, sos1_chaos[:,1]/D1*a1, marker='.', s=markersize, color='C0', edgecolors='none')
    ax[1,2].set_xlabel(r'$\tilde{q} \, [1/a]$')
    # ax[1,2].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
    
    fig.tight_layout()
    fig.savefig('plots/morse_traj.png', dpi=300, format='png')
    
    
    
    x_mo_2_1 = np.load(path+'ebk_traj_x_res.npy'); y_mo_2_1 = np.load(path+'ebk_traj_y_res.npy')
    x_mo_3_1 = np.load(path+'ebk_traj_x_3_1_res.npy'); y_mo_3_1 = np.load(path+'ebk_traj_y_3_1_res.npy')
    x_mo_3_2 = np.load(path+'ebk_traj_x_3_2_res.npy'); y_mo_3_2 = np.load(path+'ebk_traj_y_3_2_res.npy')
    path = 'data/coupled_HO/'
    x_ho_2_1 = np.load(path+'ebk_traj_x_2_1_res.npy'); y_ho_2_1 = np.load(path+'ebk_traj_y_2_1_res.npy')
    x_ho_3_1 = np.load(path+'ebk_traj_x_3_1_res.npy'); y_ho_3_1 = np.load(path+'ebk_traj_y_3_1_res.npy')
    x_ho_3_2 = np.load(path+'ebk_traj_x_3_2_res.npy'); y_ho_3_2 = np.load(path+'ebk_traj_y_3_2_res.npy')
    E_total = np.linspace(3.0, 20.0, 21)[5]
    max_x_ind = int(5e4)
    
    fig, ax = plt.subplots(2, 3, figsize=get_size('thesis', fraction=1.0, ratio='equal', subplots=(2, 3)))
    
    x_plot = np.linspace(-np.log(1 + np.sqrt(E_total/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total/D1))/a1 + buffer, 100)
    y_plot = np.linspace(-np.log(1 + np.sqrt(E_total/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total/D2))/a2 + buffer, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)
    
    x_plot_ho = np.linspace(-np.sqrt(2*E_total)/omega1 - buffer, np.sqrt(2*E_total)/omega1 + buffer, 100)
    y_plot_ho = np.linspace(-np.sqrt(2*E_total)/omega2 - buffer, np.sqrt(2*E_total)/omega2 + buffer, 100)
    X_ho, Y_ho = np.meshgrid(x_plot_ho, y_plot_ho)
    Z_ho = harm_pot(X_ho, omega1) + harm_pot(Y_ho, omega2)
    
    ax[0,0].contour(X_ho*a1, Y_ho*a1, Z_ho/D1, linewidths=0.5, cmap=cm.Greens)
    ax[0,0].plot(x_ho_2_1[:max_x_ind]*a1, y_ho_2_1[:max_x_ind]*a1, lw=linewidths, color='gray')
    # ax[0,0].set_xlabel('$q_1 \, [1/a]$')
    ax[0,0].set_ylabel('$q_2 \, [1/a]$')
    ax[0,0].set_xlim([np.min(x_plot_ho)*a1, np.max(x_plot_ho)*a1])
    ax[0,0].set_ylim([np.min(y_plot_ho)*a1, np.max(y_plot_ho)*a1])
    ax[0,0].set_aspect('equal')
    
    ax[0,1].contour(X_ho*a1, Y_ho*a1, Z_ho/D1, linewidths=0.5, cmap=cm.Greens)
    ax[0,1].plot(x_ho_3_1[:max_x_ind]*a1, y_ho_3_1[:max_x_ind]*a1, lw=linewidths, color='gray')
    # ax[0,1].set_xlabel('$q_1 \, [1/a]$')
    # ax[0,1].set_ylabel('$q_2 \, [1/a]$')
    ax[0,1].set_xlim([np.min(x_plot_ho)*a1, np.max(x_plot_ho)*a1])
    ax[0,1].set_ylim([np.min(y_plot_ho)*a1, np.max(y_plot_ho)*a1])
    ax[0,1].set_aspect('equal')
    
    ax[0,2].contour(X_ho*a1, Y_ho*a1, Z_ho/D1, linewidths=0.5, cmap=cm.Greens)
    ax[0,2].plot(x_ho_3_2[:max_x_ind]*a1, y_ho_3_2[:max_x_ind]*a1, lw=linewidths, color='gray')
    # ax[0,2].set_xlabel('$q_1 \, [1/a]$')
    # ax[0,2].set_ylabel('$q_2 \, [1/a]$')
    ax[0,2].set_xlim([np.min(x_plot_ho)*a1, np.max(x_plot_ho)*a1])
    ax[0,2].set_ylim([np.min(y_plot_ho)*a1, np.max(y_plot_ho)*a1])
    ax[0,2].set_aspect('equal')
    
    ax[1,0].contour(X*a1, Y*a1, Z/D1, linewidths=0.5, cmap=cm.Greens)
    ax[1,0].plot(x_mo_2_1[:max_x_ind]*a1, y_mo_2_1[:max_x_ind]*a1, lw=0.5, color='gray')
    ax[1,0].set_xlabel('$q_1 \, [1/a]$')
    ax[1,0].set_ylabel('$q_2 \, [1/a]$')
    ax[1,0].set_xlim([np.min(x_plot)*a1, np.max(x_plot)*a1])
    ax[1,0].set_ylim([np.min(y_plot)*a1, np.max(y_plot)*a1])
    ax[1,0].set_aspect('equal')
    
    ax[1,1].contour(X*a1, Y*a1, Z/D1, linewidths=0.5, cmap=cm.Greens)
    ax[1,1].plot(x_mo_3_1*a1, y_mo_3_1*a1, lw=0.5, color='gray')
    ax[1,1].set_xlabel('$q_1 \, [1/a]$')
    # ax[1,1].set_ylabel('$q_2 \, [1/a]$')
    ax[1,1].set_xlim([np.min(x_plot)*a1, np.max(x_plot)*a1])
    ax[1,1].set_ylim([np.min(y_plot)*a1, np.max(y_plot)*a1])
    ax[1,1].set_aspect('equal')
    
    ax[1,2].contour(X*a1, Y*a1, Z/D1, linewidths=0.5, cmap=cm.Greens)
    ax[1,2].plot(x_mo_3_2[:max_x_ind]*a1, y_mo_3_2[:max_x_ind]*a1, lw=0.5, color='gray')
    ax[1,2].set_xlabel('$q_1 \, [1/a]$')
    # ax[1,2].set_ylabel('$q_2 \, [1/a]$')
    ax[1,2].set_xlim([np.min(x_plot)*a1, np.max(x_plot)*a1])
    ax[1,2].set_ylim([np.min(y_plot)*a1, np.max(y_plot)*a1])
    ax[1,2].set_aspect('equal')
    
    fig.tight_layout()
    fig.savefig('plots/res_traj.png', dpi=300, format='png')
    # plt.show()


main()

