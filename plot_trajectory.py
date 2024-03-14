"""Plot the trajectory and the surfaces of section."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from aux_functions import harm_pot, morse_pot
from run_trajectory import sort_vertices
from get_LaTex_plot_settings import get_tex_fonts, get_size
plt.rcParams.update(get_tex_fonts('thesis'))


def plot_trajectory(x, y, px, py, energy, sos1, sos2, sos1_area, sos2_area, E_total, k, system, coupling, omega1, omega2, D1, a1, D2, a2, slopes, corners=[], start_end_inds=[]):
    """Plot the trajectory and the surfaces of section."""
    
    sos1 = np.asarray(sos1); sos2 = np.asarray(sos2)
    slope1, intsec1, slope2, intsec2 = slopes
    buffer = 0.01
    linewidths = 2

    if system == 'harmonic':
        levels = [2.0]
        x_plot = np.linspace(-np.sqrt(2*E_total)/omega1 - buffer, np.sqrt(2*E_total)/omega1 + buffer, 100)
        y_plot = np.linspace(-np.sqrt(2*E_total)/omega2 - buffer, np.sqrt(2*E_total)/omega2 + buffer, 100)
        X, Y = np.meshgrid(x_plot, y_plot)
        if coupling == 'kinetic':
            Z = harm_pot(X, omega1) + harm_pot(Y, omega2)

    elif system == 'morse':
        levels = [1.0, 2.0, 3.0]
        x_plot = np.linspace(-np.log(1 + np.sqrt(E_total/D1))/a1 - buffer, -np.log(1 - np.sqrt(E_total/D1))/a1 + buffer, 100)
        y_plot = np.linspace(-np.log(1 + np.sqrt(E_total/D2))/a2 - buffer, -np.log(1 - np.sqrt(E_total/D2))/a2 + buffer, 100)
        X, Y = np.meshgrid(x_plot, y_plot)
        if coupling == 'kinetic':
            Z = morse_pot(X, D1, a1) + morse_pot(Y, D2, a2)

    fig, ax = plt.subplots(1, 2, figsize=get_size('thesis', fraction=1.0, ratio='equal', subplots=(1, 2)))
    ax[0].set_aspect('equal')
    CS = ax[0].contour(X, Y, Z, linewidths=0.5, cmap=cm.gray)#, levels=levels)
    # ax[0].clabel(CS, inline=True, fontsize=10)
    ax[0].plot(x, y, lw=0.5, color='gray')
    ax[0].plot(x[1500:15000], y[1500:15000], lw=1.0, color='black') # Show a few winds of trajectory
    ax[0].scatter(x[0], y[0], marker='x', color='black') # Starting point
    # ax[0].plot(x_plot, x_plot, linestyle='--', color='gray', label='Symmetric stretch')
    
    if np.isnan(slope1):
        ax[0].axhline(y=intsec1, linestyle='--', lw=linewidths, color='C0')
    else:
        ax[0].plot(x_plot, slope1*x_plot + intsec1, linestyle='--', lw=linewidths, color='C0')
        
    if np.isnan(slope2):
        ax[0].axvline(x=intsec2, linestyle='--', lw=linewidths, color='C1')
    else:
        ax[0].plot(x_plot, slope2*x_plot + intsec2, linestyle='--', lw=linewidths, color='C1')
    
    if len(corners) > 0:
        ax[0].scatter(corners[0][0], corners[1][0], s=100, color='C1')#, label='Corner #0')
        ax[0].scatter(corners[0][1], corners[1][1], s=100, color='C2')#, label='Corner #1')
        ax[0].scatter(corners[0][2], corners[1][2], s=100, color='C3')#, label='Corner #2')
        ax[0].scatter(corners[0][3], corners[1][3], s=100, color='C4')#, label='Corner #3')
    
    if len(start_end_inds) > 0:
        ax[0].plot(x[start_end_inds[0]:start_end_inds[1]], y[start_end_inds[0]:start_end_inds[1]], lw=1, color='black')
        ax[0].plot(x[start_end_inds[2]:start_end_inds[3]], y[start_end_inds[2]:start_end_inds[3]], lw=1, linestyle='--', color='red')
    
    ax[0].set_xlabel('$q_1$')
    ax[0].set_ylabel('$q_2$')
    ax[0].set_xlim([np.min(x_plot), np.max(x_plot)])
    ax[0].set_ylim([np.min(y_plot), np.max(y_plot)])
    
    
    if len(sos1) > 1:
        array1 = sort_vertices(sos1)
        ax[1].plot(array1[:,0]*a1, array1[:,1]/D1*a1, marker='.', ms=1, color='C0', linestyle='none')
        # ax[1].scatter(np.mean(sos1[:,0]), np.mean(sos1[:,1]), s=50, color='C0')
        
    if len(sos2) > 1:
        array2 = sort_vertices(sos2)
        ax[1].plot(array2[:,0]*a1, array2[:,1]/D1*a1, marker='.', ms=1, color='C1', linestyle='none')
        # ax[1].scatter(np.mean(sos2[:,0]), np.mean(sos2[:,1]), s=50, marker='x', color='C1')
        
    ax[1].set_xlabel(r'$\tilde{q} \, [1/a]$')
    ax[1].set_ylabel(r'$\tilde{p} \, [D_e/a]$')
    fig.tight_layout()
    fig.savefig('plots/traj_plot.pdf', format='pdf')


    fig, ax = plt.subplots(1, 1, figsize=get_size('thesis', fraction=1.0))
    ax.plot(np.arange(len(energy)), (energy-E_total)/energy)
    ax.set_xlabel('time step')
    ax.set_ylabel('relative error in E')
    plt.show()

