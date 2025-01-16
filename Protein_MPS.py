import numpy as np
import matplotlib.pyplot as plt
from parser import extract_state_from_file
import MPS_in_Qiskit.prepare_MPS as mps
from qiskit_aer import AerSimulator
import numpy as np
from qiskit import QuantumCircuit, transpile
import time

n = 10

def split(M, bond_dim):
    """Split a matrix M via SVD and keep only the ``bond_dim`` largest entries."""
    U, S, Vd = np.linalg.svd(M, full_matrices=False)
    bonds = len(S)
    Vd = Vd.reshape(bonds, 2, -1)
    U = U.reshape((-1, 2, bonds))

    # keep only chi bonds
    chi = np.min([bonds, bond_dim])
    U, S, Vd = U[:, :, :chi], S[:chi], Vd[:chi]
    return U, S, Vd

def dense_to_mps(psi, bond_dim):
    """Turn a state vector ``psi`` into an MPS with bond dimension ``bond_dim``."""
    Ms = []
    Ss = []

    psi = np.reshape(psi, (2, -1))   # split psi[2, 2, 2, 2..] = psi[2, (2x2x2...)]
    U, S, Vd = split(psi, bond_dim)  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]

    Ms.append(U)
    Ss.append(S)
    bondL = Vd.shape[0]
    psi = np.tensordot(np.diag(S), Vd, 1)

    for _ in range(n-2):
        psi = np.reshape(psi, (2*bondL, -1)) # reshape psi[2 * bondL, (2x2x2...)]
        U, S, Vd = split(psi, bond_dim) # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]
        Ms.append(U)
        Ss.append(S)

        psi = np.tensordot(np.diag(S), Vd, 1)
        bondL = Vd.shape[0]

    # dummy step on last site
    psi = np.reshape(psi, (-1, 1))
    U, _, _ = np.linalg.svd(psi, full_matrices=False)

    U = np.reshape(U, (-1, 2, 1))
    Ms.append(U)
    reshaped_Ms = [np.transpose(M, (1, 0, 2)) for M in Ms] 
    return reshaped_Ms, Ss

# [print(f'M:{M.shape}') for M in Ms]
# [print(f'S:{S.shape}') for S in Ss]


def reshape_with_padding(A, x, y, z):
    """
    Reshape tensor A to shape (x, y, z) and pad with zeros if necessary.
    
    Parameters:
        A: Input tensor.
        x, y, z: Desired shape for the output tensor.
    
    Returns:
        Padded tensor of shape (x, y, z).
    """
    # Get the current shape of the input tensor
    x1,y1,z1 = A.shape
    if y1 < y:
        A = np.pad(A,((0,0), (0,y-y1), (0,0)), mode = 'constant', constant_values= (0,0))
     # Total number of elements in the original tensor
    if z1 < z:
        A = np.pad(A,((0,0), (0,0), (0,z-z1)), mode = 'constant', constant_values= (0,0))
    # Desired shape and the total number of elements in the new tensor
    target_size = x * y * z
    


    return A

def construct_mps(Ms, Ss, bond_dim):
    """
    Construct the MPS tensors A_i by combining M and S tensors.
    
    Parameters:
        Ms: List of M tensors
        Ss: List of S tensors
        
    Returns:
        A: List of MPS tensors A_1, A_2, ..., A_N
    """
    A = []  
    
    # Iterate over each M and S pair to create the corresponding A_i tensor
    for i in range(len(Ms)-1):
        M = Ms[i]  # Shape: (physical_dim, x, bond_dim)
        S = Ss[i]  # Shape: (bond_dim,)
        
        # Create a diagonal matrix from S
        S_diag = np.diag(S)  # Shape: (bond_dim, bond_dim)
        
        # Contract the M and S to get the A_i tensor
        A_i = np.tensordot(M, S_diag, axes=(2, 0))  
        
        # Reshape the result to match the MPS form (physical_dim, bond_dim, bond_dim)
        A_i_padded = reshape_with_padding(A_i, 2, bond_dim, bond_dim)  # Pad with zeros if necessary
        
        # Append the result to A
        A.append(A_i_padded)
    A.append(reshape_with_padding(Ms[i],2,bond_dim,bond_dim))
    
    return A

num_protein = 5
bond_dims = [2,4,8,16,32]
for bd in bond_dims:
  for i in range(num_protein):
    start_time = time.time()

    protein_state = extract_state_from_file(f'Protein_states_100/state{i}.txt',10)
    # print(protein_0_state)
    bond_dim = bd
    print(f'BOND_DIM: {bond_dim}')
    print(f'Protein Number: {i}')
    Ms, Ss = dense_to_mps(protein_state, bond_dim)
    # [print(f'M:{M.shape}') for M in Ms]
    # [print(f'S:{S.shape}') for S in Ss]
    As = construct_mps(Ms, Ss, bond_dim)
    # [[print(A.shape) for A in As]]



    # Create Random MPS with size 4, bond dimension 4 and physical dimension 2 (qubits)
    N = 10
    d = 2
    chi = bond_dim
    phi_final = np.random.rand(chi)
    phi_initial = np.random.rand(chi)



    # print('Finding MPS Circuit')
    # Create the circuit. The 'reg' register corresponds to the 'MPS' register
    circ, reg = mps.MPS_to_circuit(As, phi_initial, phi_final)
    # print('MPS CircuitCreated - Transpiling')
    basis_gates = ['cx', 'rx', 'ry', 'rz']
    decomposed_circ = transpile(circ, basis_gates=basis_gates)

    # print('Trasnpiled - Simulating')
    decomposed_circ.save_statevector()
    simulator = AerSimulator(method='statevector')
    result = simulator.run(decomposed_circ).result()
    psi_out = result.get_statevector(decomposed_circ)
    psi_out = psi_out.reshape(d**N, chi)

    exp = psi_out.dot(phi_final)
    thr, _ = mps.create_statevector(As, phi_initial, phi_final, qiskit_ordering=True)
    exp = mps.normalize(mps.extract_phase(exp))
    thr = mps.normalize(mps.extract_phase(thr))
    # print("The MPS is \n{}".format(thr))
    # print(thr.shape)
    # print("The statevector produced by the circuit is \n{}".format(exp))
    # print("The statevector produced by the Transpiled circuit is \n{}".format(exp))

    fidelity = np.abs(np.vdot(thr, exp))**2
    fidelity_truncation = np.abs(np.vdot(protein_state, exp))**2
    # print(f"BOND-Dimension:{bond_dim}")
    print(f"Fidelity_MPS_Circuit: {fidelity}")
    print(f"Fidelity from original state to truncated: {fidelity_truncation}")
    gate_counts = decomposed_circ.count_ops()

    # Extract total gate count and CX gate count
    total_gates = sum(gate_counts.values())  # Total number of gates
    cx_gates = gate_counts.get('cx', 0)  
    print(f"MPS Circuit Gate Count: {total_gates}")
    print(f"MPS Circuit CX Count: {cx_gates}")
    end_time = time.time()
    print(f"Runtime: {end_time - start_time} seconds")
