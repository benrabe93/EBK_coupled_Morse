"""Plot a 1D Morse potential with energy levels."""

import numpy as np
import matplotlib.pyplot as plt
from aux_functions import morse_pot
from get_LaTex_plot_settings import get_tex_fonts, get_size
plt.rcParams.update(get_tex_fonts('thesis'))


def morse_pot_inv(E, D, a):
    """Return the two roots of the Morse potential.

    Args:
        E (scalar): energy
        D (scalar): Dissociation energy of the Morse potential
        a (scalar): Stiffness parameter of the Morse potential

    Returns:
        np.ndarray: two roots of the Morse potential
    """
    
    x1 = -np.log(1 + np.sqrt(E/D))/a
    x2 = -np.log(1 - np.sqrt(E/D))/a
    return np.array([x1, x2])



D = 32.0; a = 1.0/5
omega = a*np.sqrt(2*D)
n = np.arange(37) # Number of energy levels
energies = omega*(n + 0.5) - omega**2/(4*D)*(n + 0.5)**2
x = np.linspace(-3.8, 30, 1000)

fig, ax = plt.subplots(1, 1, figsize=get_size('thesis', fraction=0.5))
for i in range(len(energies)):
    xi = morse_pot_inv(energies[i], D, a)
    ax.plot(xi*a, np.array([energies[i], energies[i]])/D, lw=0.7)
ax.plot(x*a, morse_pot(x, D, a)/D, color='black')
ax.set_xlabel('$x \, [1/a]$')
ax.set_ylabel('$E \, [D_e]$')
ax.set_yticks([0, 1])
fig.tight_layout()
fig.savefig('plots/morse_pot.pdf', format='pdf')

