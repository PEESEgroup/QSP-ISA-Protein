import MPS_in_Qiskit.prepare_MPS as mps
from qiskit_aer import AerSimulator
import numpy as np
from qiskit import QuantumCircuit, transpile
import os
import qc
from parser import extract_state_from_file
import numpy as np
from datetime import datetime
import time
import csv

# Create Random MPS with size 4, bond dimension 4 and physical dimension 2 (qubits)
N = 6
d = 2
chi = 16
print(f"Bond Dimension: {chi}")
phi_final = np.random.rand(chi)
phi_initial = np.random.rand(chi)
A = mps.create_random_tensors(N, chi, d)
[print(f'M:{a.shape}') for a in A]


# Create the circuit. The 'reg' register corresponds to the 'MPS' register
circ, reg = mps.MPS_to_circuit(A, phi_initial, phi_final)
circ.save_statevector()

circ_t, reg_t = mps.MPS_to_circuit(A, phi_initial, phi_final)
basis_gates = ['cx', 'rx', 'ry', 'rz']
decomposed_circ = transpile(circ_t, basis_gates=basis_gates)
# print(decomposed_circ.draw())
decomposed_circ.save_statevector()
# Run the circuit on the statevector simulator using AerSimulator
simulator = AerSimulator(method='statevector')
# qc = transpile(qc, simulator)   # The 'statevector' method is the default
result = simulator.run(circ).result()
psi_out = result.get_statevector(circ)
simulator_t = AerSimulator(method='statevector')
# qc = transpile(qc, simulator)   # The 'statevector' method is the default
result_t = simulator_t.run(decomposed_circ).result()
psi_out_t = result_t.get_statevector(decomposed_circ)
# Contract out the ancilla with the known state
psi_out = psi_out.reshape(d**N, chi)
exp = psi_out.dot(phi_final)

psi_out_t = psi_out_t.reshape(d**N, chi)
exp_t = psi_out_t.dot(phi_final)

# Prepare the MPS classically
thr, _ = mps.create_statevector(A, phi_initial, phi_final, qiskit_ordering=True)

# Compare the resulting vectors (fixing phase and normalization)
exp = mps.normalize(mps.extract_phase(exp))
exp_t = mps.normalize(mps.extract_phase(exp_t))
thr = mps.normalize(mps.extract_phase(thr))

# Print the results
print("The MPS is \n{}".format(thr))
print(thr.shape)
# print("The statevector produced by the circuit is \n{}".format(exp))
print("The statevector produced by the Transpiled circuit is \n{}".format(exp_t))

print(circ.draw())
basis_gates = ['cx', 'rx', 'ry', 'rz']
decomposed_circ = transpile(circ_t, basis_gates=basis_gates)
# print(decomposed_circ.draw())

fidelity = np.abs(np.vdot(thr, exp_t))**2
print(f"Fidelity_MPS_Circuit: {fidelity}")
gate_counts = decomposed_circ.count_ops()

# Extract total gate count and CX gate count
total_gates = sum(gate_counts.values())  # Total number of gates
cx_gates = gate_counts.get('cx', 0)      # Count of CX gates (0 if not present)

# Print the results
print(f"MPS Circuit Gate Count: {total_gates}")
print(f"MPS Circuit CX Count: {cx_gates}")


def vector_to_state_txt_file(embedding, output_file):
    n = int(np.log2(len(embedding)))  # Number of qubits
    if 2**n != len(embedding):
        raise ValueError("Length of the embedding vector must be a power of 2.")

    # Step 1: Calculate amplitudes and phases
    amplitudes = np.abs(embedding)
    phases = np.angle(embedding)  # Compute phases in radians

    # Step 2: Normalize the amplitudes
    normalization_factor = np.sqrt(np.sum(amplitudes**2))
    normalized_amplitudes = amplitudes / normalization_factor

    # Step 3: Write to file
    with open(output_file, 'w') as f:
        for i in range(len(embedding)):
            binary_representation = format(i, f'0{n}b')  # Convert index to binary
            amplitude = normalized_amplitudes[i]
            phase = phases[i]
            f.write(f"{binary_representation} {amplitude:.6g} {phase:.6g}\n")


vector_to_state_txt_file(thr, 'mps_state')

#Compile cpp code to executable, if it hasn't been compiled already
if "isa_cpp" not in os.listdir():
    os.system("./build_isa")
#In the below command,
#  [output.txt] is the output file (can be named anything)
#  [state.txt] is the input file
#  [5] is the number of qubits
#  [0.95] is the target fidelity. The target fidelity is an optional argument
#    with 0.95 as default value
state_file = 'mps_state'
os.system(f"./isa_cpp output_cpp.txt {state_file} {N} 0.999999")
state = extract_state_from_file(state_file,N)

#Now read the gates from the output file
gate_sequence = []
with open("output_cpp.txt") as f:
    while True:
        line = f.readline()
        if len(line) == 0: break
        if len(line) < 3: continue
        [gate_type, a1, a2] = line.split()
        if gate_type == "rx": 
            gate_sequence.append(qc.Gate.RX(int(a1), float(a2), N ))
        elif gate_type == "ry":
            gate_sequence.append(qc.Gate.RY(int(a1), float(a2), N))
        elif gate_type == "rz":
            gate_sequence.append(qc.Gate.RZ(int(a1), float(a2), N))
        elif gate_type == "cx" or gate_type == "xc":
            gate_sequence.append(qc.Gate.CX(int(a1), int(a2), N))
        else: raise

#[gate_sequence] is the same as 
# gate_sequence = isa_prepare(state, target_fidelity=0.95)
# from py_demo.py
# Getting the gate sequence using the C++ executable is a bit more convoluted
# but the C++ code is much much faster than the Python version.

CX_count = 0
for gate in gate_sequence:
    # print(gate.to_string())
    if gate.gate_type == "cx" or gate.gate_type == "xc":
        CX_count += 1
print(f"ISA Circuit Number of Gates : {len(gate_sequence)}")
print(f"ISA Circuit CX Count: {CX_count}")
print(state_file)
state_c = state
for gate in gate_sequence:
    state_c = qc.apply_gate(gate, state_c)

fidelity_ISA = abs(state_c[0]) ** 2 #Should be greater than 0.95
print(f"Fidelity_ISA_Circuit: {fidelity_ISA}")

