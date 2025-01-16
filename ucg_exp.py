import numpy as np
import numpy as np
import os
from qclib.state_preparation import UCGInitialize
from datetime import datetime
import time
import qc
import qiskit
# from qiskit import execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import state_fidelity
from qiskit.providers.basic_provider import BasicProvider
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy
import random
import cmath
from math import atan


from qiskit.quantum_info import state_fidelity

n_qubits = 10
CX_ucg_counts = []
Runtimes_ucg = []
num_protein  = 20

coupling_map = []
for i in range(n_qubits - 1):
    coupling_map.append([i, i + 1])
    coupling_map.append([i + 1, i])
print(coupling_map)

for i in range(num_protein):
    state_file = f"./States/state{i}.txt"
    state = []
    with open(state_file, 'r') as file:
        for line in file:
            columns = line.split()  
            if len(columns) >= 3:  

                amp = float(columns[1])  
                phase = float(columns[2])  
                if phase == 0.0:  
                    state.append(amp)
                else:
                    state.append(-1*amp)


        state = state/np.linalg.norm(state)
        ucg_time = time.time()
       


        # qclib ucg
        circuit = UCGInitialize(state).definition


        backend = AerSimulator()



        lnn = True
        if lnn:
            transpiled = qiskit.transpile(circuit, backend, basis_gates=['u', 'cx'], coupling_map = coupling_map, optimization_level=0)
        else:
            transpiled = qiskit.transpile(circuit, backend, basis_gates=['u', 'cx'], optimization_level=0)
        print(f"LNN: {lnn}")
        print(f"N={n_qubits}")
        # transpiled = transpile(circuit, backend)
        ucg_runtime = time.time() - ucg_time
        Runtimes_ucg.append(ucg_runtime)
        cx = transpiled.count_ops().get('cx', 0)
        CX_ucg_counts.append(cx)
        # provider = BasicProvider()
        # backend = provider.get_backend("basic_simulator")
        # statevector = Statevector(transpiled)
        backend = AerSimulator()
        transpiled.save_statevector()
        state_vector = backend.run(transpiled).result().get_statevector()

        # # print(f"Input State : State Vector")
        # for i in range(len(state)):
        #     # print((state[i],state_vector.data[i]))
      
        fidelity = state_fidelity(state,state_vector)

        print(f"Fidelity: {fidelity}")


        
# Extract gate information
       
# Now you can access the gate sequence
# print(CX_ucg_counts)
# print(Runtimes_ucg)
print(f"Avg CX: {sum(CX_ucg_counts)/len(CX_ucg_counts)}")
print(f"Avg Runtime : {sum(Runtimes_ucg)/len(Runtimes_ucg)}")

