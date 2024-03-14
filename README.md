# EBK_coupled_Morse

The Einstein–Brillouin–Keller (EBK) Quantization method is an improvement over the Bohr-Sommerfeld Quantization and allows for coordinate-independent semi-classical energy quantization using regular classical trajectories. In this repository I treat two identical, linearly aligned, kinetically coupled Morse oscillators, simulating a simplified tri-atomic molecule.

The system Hamiltonian is discussed in [[C. G. Schlier, “Collinear collisions in a double-Morse well”, Chem. Phys. 105, 361
(1986)]](https://doi.org/10.1016/0301-0104(86)80124-0) and writes

$$
H(\lambda) = p_1^2/2 + V_{\text{Mor}}(q_1) + p_2^2/2 + V_{\text{Mor}}(q_2) - \lambda p_1 p_2 ,
$$

where $\lambda$ is an adiabatic parameter corresponding to the mass ratio of the center atom and the mass of the outer atoms.
