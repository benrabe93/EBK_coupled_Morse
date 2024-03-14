"""Plot the spectra of two kinetically coupled Morse oscillators and two coupled harmonic oscillators."""

import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from get_LaTex_plot_settings import get_tex_fonts, get_size
plt.rcParams.update(get_tex_fonts('thesis'))


# Morse parameters
D1 = 32.0; a1 = 1.0/5; k_param = 40

# Load quantum spectra
path = "data/coupled_MO/"
# k = np.load(path + "k.npy")
k = np.linspace(0, 1, 1001)[:-1]
k_grid = np.linspace(0, 0.9, 251)
spectra_sym = np.load(path + "spectra_grid_1_5_sym.npy")
spectra_asym = np.load(path + "spectra_grid_1_5_asym.npy")


### Compute spectra for two identically harmonic oscillators with kinetic coupling ###
omega1 = a1*np.sqrt(2*D1)
n_max = 40 # Maximum vibrational quantum number

basis = np.array(list(product(np.arange(min(n_max, int(k_param - 0.5)) + 1), repeat=2)))
inds_sym = (basis[:,1]%2 == 0)
basis_sym = basis[inds_sym]
basis_asym = basis[np.invert(inds_sym)]
# print(len(basis))

spectra_ho = np.zeros((len(basis), len(k)))
spectra_ho_sym = np.zeros((len(basis_sym), len(k)))
spectra_ho_asym = np.zeros((len(basis_asym), len(k)))
for i in range(len(spectra_ho[0])):
    spectra_ho[:,i] = np.sort(omega1*np.sqrt(1 - k[i])*(basis[:,0] + 0.5) + omega1*np.sqrt(1 + k[i])*(basis[:,1] + 0.5))
    spectra_ho_sym[:,i] = np.sort(omega1*np.sqrt(1 - k[i])*(basis_sym[:,0] + 0.5) + omega1*np.sqrt(1 + k[i])*(basis_sym[:,1] + 0.5))
    spectra_ho_asym[:,i] = np.sort(omega1*np.sqrt(1 - k[i])*(basis_asym[:,0] + 0.5) + omega1*np.sqrt(1 + k[i])*(basis_asym[:,1] + 0.5))


# Scale spectra
spectra_sym = spectra_sym/D1; spectra_asym = spectra_asym/D1
spectra_ho = spectra_ho/D1; spectra_ho_sym = spectra_ho_sym/D1; spectra_ho_asym = spectra_ho_asym/D1

xlims = [0.0, 0.9]; ylims = [0.3, 0.5]


fig, ax = plt.subplots(1, 2, figsize=get_size('thesis', fraction=1.0, subplots=(1, 2), ratio=0.8))
for i in range(len(spectra_sym)):
    ax[0].plot(k_grid, spectra_sym[i], lw=0.5, color='C0')
for i in range(len(spectra_asym)):
    ax[0].plot(k_grid, spectra_asym[i], lw=0.5, color='C1')
ax[0].set_xlabel(r"$\lambda$")
ax[0].set_ylabel(r"$E \, [D_e]$")
ax[0].set_xlim(xlims)
ax[0].set_ylim(ylims)

# for i in range(len(spectra_ho)):
#     ax[1].plot(k, spectra_ho[i], lw=0.5, color='gray')
for i in range(len(spectra_ho_sym)):
    ax[1].plot(k, spectra_ho_sym[i], lw=0.5, color='C0')
for i in range(len(spectra_ho_asym)):
    ax[1].plot(k, spectra_ho_asym[i], lw=0.5, color='C1')
ax[1].set_xlabel(r"$\lambda$")
# ax[1].set_ylabel(r"$E \, [D_e]$")
ax[1].set_xlim(xlims)
ax[1].set_ylim(ylims)

fig.tight_layout()
fig.savefig('plots/spectrum_mo_ho.pdf', format='pdf')
# plt.show()
