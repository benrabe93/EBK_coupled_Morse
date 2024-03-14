"""Auxiliary functions for the computation of the quantum-classical dynamics of a triatomic molecule."""

import numpy as np
from numba import njit


def divide_factorials(arg_num, arg_denom):
    """Compute the ratio of two factorials efficiently by first reducing the fraction.

    Args:
        arg_num (int or np.ndarray): Numerator, positive integer
        arg_denom (int or np.ndarray): Denominator, positive integer

    Returns:
        float or np.ndarray: arg_num! / arg_denom!
    """
    
    if np.isscalar(arg_num) and np.isscalar(arg_denom):
        if arg_num > arg_denom:
            return np.prod(np.arange(arg_denom, arg_num) + 1.0)
        elif arg_denom > arg_num:
            return 1.0/np.prod(np.arange(arg_num, arg_denom) + 1.0)
        elif arg_num == arg_denom:
            return 1.0

    elif np.isscalar(arg_num):
        sol = np.zeros(len(arg_denom))
        for i in range(len(arg_denom)):
            if arg_num > arg_denom[i]:
                sol[i] = np.prod(np.arange(arg_denom[i], arg_num) + 1.0)
            elif arg_denom[i] > arg_num:
                sol[i] = 1.0/np.prod(np.arange(arg_num, arg_denom[i]) + 1.0)
            elif arg_num == arg_denom[i]:
                sol[i] = 1.0

    elif np.isscalar(arg_denom):
        sol = np.zeros(len(arg_num))
        for i in range(len(arg_num)):
            if arg_num[i] > arg_denom:
                sol[i] = np.prod(np.arange(arg_denom, arg_num[i]) + 1.0)
            elif arg_denom > arg_num[i]:
                sol[i] = 1.0/np.prod(np.arange(arg_num[i], arg_denom) + 1.0)
            elif arg_num[i] == arg_denom:
                sol[i] = 1.0

    elif len(arg_num) == len(arg_denom):
        sol = np.zeros(len(arg_num))
        for i in range(len(sol)):
            if arg_num[i] > arg_denom[i]:
                sol[i] = np.prod(np.arange(arg_denom[i], arg_num[i]) + 1.0)
            elif arg_denom[i] > arg_num[i]:
                sol[i] = 1.0/np.prod(np.arange(arg_num[i], arg_denom[i]) + 1.0)
            elif arg_num[i] == arg_denom[i]:
                sol[i] = 1.0
    
    return sol


def sqrt_divide_factorials(arg_num, arg_denom):
    """Compute the sqrt of the ratio of two factorials efficiently by first reducing the fraction.

    Args:
        arg_num (int or np.ndarray): Numerator, positive integer
        arg_denom (int or np.ndarray): Denominator, positive integer

    Returns:
        float or np.ndarray: sqrt(arg_num! / arg_denom!)
    """
    
    if np.isscalar(arg_num) and np.isscalar(arg_denom):
        if arg_num > arg_denom:
            # return np.sqrt(np.prod(np.arange(arg_denom, arg_num) + 1.0))
            return np.prod(np.sqrt(np.arange(arg_denom, arg_num) + 1.0))
        elif arg_denom > arg_num:
            # return 1.0/np.sqrt(np.prod(np.arange(arg_num, arg_denom) + 1.0))
            return 1.0/np.prod(np.sqrt(np.arange(arg_num, arg_denom) + 1.0))
        elif arg_num == arg_denom:
            return 1.0

    elif np.isscalar(arg_num):
        sol = np.zeros(len(arg_denom))
        for i in range(len(arg_denom)):
            if arg_num > arg_denom[i]:
                sol[i] = np.prod(np.sqrt(np.arange(arg_denom[i], arg_num) + 1.0))
            elif arg_denom[i] > arg_num:
                sol[i] = 1.0/np.prod(np.sqrt(np.arange(arg_num, arg_denom[i]) + 1.0))
            elif arg_num == arg_denom[i]:
                sol[i] = 1.0

    elif np.isscalar(arg_denom):
        sol = np.zeros(len(arg_num))
        for i in range(len(arg_num)):
            if arg_num[i] > arg_denom:
                sol[i] = np.prod(np.sqrt(np.arange(arg_denom, arg_num[i]) + 1.0))
            elif arg_denom > arg_num[i]:
                sol[i] = 1.0/np.prod(np.sqrt(np.arange(arg_num[i], arg_denom) + 1.0))
            elif arg_num[i] == arg_denom:
                sol[i] = 1.0

    elif len(arg_num) == len(arg_denom):
        sol = np.zeros(len(arg_num))
        for i in range(len(sol)):
            if arg_num[i] > arg_denom[i]:
                sol[i] = np.prod(np.sqrt(np.arange(arg_denom[i], arg_num[i]) + 1.0))
            elif arg_denom[i] > arg_num[i]:
                sol[i] = 1.0/np.prod(np.sqrt(np.arange(arg_num[i], arg_denom[i]) + 1.0))
            elif arg_num[i] == arg_denom[i]:
                sol[i] = 1.0
    
    return sol


def harm_pot(x, omega, d=0):
    """Harmonic potential and its derivatives.

    Args:
        x (float): Input/argument of the potential
        omega (float): Frequency of the harmonic potential
        d (int, optional): Order of the derivative of the potential. Defaults to 0.

    Returns:
        float: Value of the potential or its derivative
    """
    
    if d==0:
        V = 0.5*(omega*x)**2
    elif d==1:
        V = omega**2*x
    elif d==2:
        V = omega**2
    return V


@njit
def harm_pot_njit(x, omega, d=0):
    """Harmonic potential and its derivatives (jitted version).

    Args:
        x (float): Input/argument of the potential
        omega (float): Frequency of the harmonic potential
        d (int, optional): Order of the derivative of the potential. Defaults to 0.

    Returns:
        float: Value of the potential or its derivative
    """
    
    if d==0:
        V = 0.5*(omega*x)**2
    elif d==1:
        V = omega**2*x
    elif d==2:
        V = omega**2
    return V


def morse_pot(x, D, a, d=0):
    """Morse potential and its derivatives.

    Args:
        x (float): Input/argument of the potential
        D (float): Dissociation energy
        a (float): Stiffness parameter
        d (int, optional): Order of the derivative of the potential. Defaults to 0.

    Returns:
        float: Value of the potential or its derivative
    """
    
    if d==0:
        V = D*(1.0 - np.exp(-a*x))**2
    elif d==1:
        V = 2.0*D*a*(np.exp(-a*x) - np.exp(-2.0*a*x))
    elif d==2:
        V = 2.0*D*a**2*(np.exp(-2.0*a*x) + (np.exp(-a*x) - 1.0)*np.exp(-a*x))
    return V


@njit
def morse_pot_njit(x, D, a, d=0):
    """Morse potential and its derivatives (jitted version).

    Args:
        x (float): Input/argument of the potential
        D (float): Dissociation energy
        a (float): Stiffness parameter
        d (int, optional): Order of the derivative of the potential. Defaults to 0.

    Returns:
        float: Value of the potential or its derivative
    """
    
    if d==0:
        V = D*(1.0 - np.exp(-a*x))**2
    elif d==1:
        V = 2.0*D*a*(np.exp(-a*x) - np.exp(-2.0*a*x))
    elif d==2:
        V = 2.0*D*a**2*(np.exp(-2.0*a*x) + (np.exp(-a*x) - 1.0)*np.exp(-a*x))
    return V


@njit
def leap_frog(x0, y0, px0, py0, dt, k, omega1, omega2, D1, a1, D2, a2, system, coupling):
    """Leap frog integration of Hamiltonians equations of motion."""
    
    if coupling == 'kinetic':
        x1 = x0 + 0.5*dt*(px0 - k*py0)
        y1 = y0 + 0.5*dt*(py0 - k*px0)

        if system == 'harmonic':
            px2 = px0 - dt*harm_pot_njit(x1, omega1, d=1)
            py2 = py0 - dt*harm_pot_njit(y1, omega2, d=1)
        elif system == 'morse':
            px2 = px0 - dt*morse_pot_njit(x1, D1, a1, d=1)
            py2 = py0 - dt*morse_pot_njit(y1, D2, a2, d=1)

        x2 = x1 + 0.5*dt*(px2 - k*py2)
        y2 = y1 + 0.5*dt*(py2 - k*px2)
    
    # elif coupling == 'potential':
    #     x1 = x0 + 0.5*dt*px0
    #     y1 = y0 + 0.5*dt*py0

    #     px2 = px0 - dt*(V1(x1, d=1) + k*y1)
    #     py2 = py0 - dt*(V2(y1, d=1) + k*x1)

    #     x2 = x1 + 0.5*dt*px2
    #     y2 = y1 + 0.5*dt*py2

    return x2, y2, px2, py2


@njit
def compute_energy(x, y, px, py, k, omega1, omega2, D1, a1, D2, a2, system, coupling):
    """Compute the total energy of the system."""
    
    if coupling == 'kinetic':
        if system == 'harmonic':
            energy = 0.5*px**2 + harm_pot_njit(x, omega1) + 0.5*py**2 + harm_pot_njit(y, omega2) - k*px*py
        elif system == 'morse':
            energy = 0.5*px**2 + morse_pot_njit(x, D1, a1) + 0.5*py**2 + morse_pot_njit(y, D2, a2) - k*px*py
    
    # elif coupling == 'potential':
    #     return 0.5*px**2 + V1(x) + 0.5*py**2 + V2(y) + k*x*y

    return energy


@njit
def set_initial_cond(x_ini, E_total, k, omega1, omega2, D, a, system, coupling, px=False, negative=False):
    """Compute the full set of initial coordinates from the initial x-coordinate and the total energy."""
    
    if px:
        px_ini = x_ini
        x_ini, y_ini, py_ini, energy_ini = np.zeros(4)
    else:
        y_ini, px_ini, py_ini, energy_ini = np.zeros(4)
    
    if system == 'harmonic':
        condition = harm_pot_njit(x_ini, omega=omega1) >= E_total - harm_pot_njit(0, omega=omega2)
    elif system == 'morse':
        condition = morse_pot_njit(x_ini, D=D, a=a) >= E_total - morse_pot_njit(0, D=D, a=a)
    
    if condition:
        y_ini = 0.0
    else:
        if coupling == "kinetic":
            if system == 'harmonic':
                if negative:
                    y_ini = -np.sqrt(2*E_total - 2*harm_pot_njit(x_ini, omega=omega1))/omega2
                else:
                    y_ini = np.sqrt(2*E_total - 2*harm_pot_njit(x_ini, omega=omega1))/omega2
            elif system == 'morse':
                if negative:
                    y_ini = -np.log(1 + np.sqrt((E_total - morse_pot_njit(x_ini, D=D, a=a))/D))/a
                else:
                    y_ini = -np.log(1 - np.sqrt((E_total - morse_pot_njit(x_ini, D=D, a=a))/D))/a

        # elif coupling == "potential":
        #     if system == 'harmonic':
        #         y_ini[i] = -k*x_ini[i]/omega2**2 + np.sqrt((k*x_ini[i]/omega2**2)**2 + 2/omega2**2*(1 - harm_pot_njit(x_ini[i], omega=omega1)))
    
    energy_ini = compute_energy(x_ini, y_ini, px_ini, py_ini, k, omega1, omega2, D, a, system, coupling)
    
    return x_ini, y_ini, px_ini, py_ini, energy_ini

