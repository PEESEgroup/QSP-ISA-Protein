from data import Gate, encode_state, decode_state, generate_dataset

#Generate training and testing datasets. Each dataset is a list of sequences.
# Each sequence is formatted as:
#   reward, state, gate, reward, state, gate, ...
# Reward is a 2-tuple (fidelity to go, number of CX remaining)
# State is encoded version of the current quantum state
# Gate is an encoded version of the gate to apply to that quantum state
# The first state is the target state to be prepared, the last state is
# approximately the basis state |0>
#
# @param n_qubits: number of qubits in the system
# @param count: number of sequences for each CX count
# @param frac_testing: fraction of sequences put in the testing set
# @param noise: parameter representing how well the gate sequences prepare
#     the target state. noise=0 means perfect preparation, noise > 1 means
#     mostly noise. Empirically, noise=0.1 works pretty well. 
training, testing = generate_dataset(n_qubits=3, count=100, frac_testing=0.2, noise=0.1)

#Example for how to interpret the output datasets.
seq0 = training[0]
(fidelity_to_go, num_cx), state, gate = seq0[0:3]
print("Fidelity to go: ", fidelity_to_go)
print("Number of CX gates remaining: ", num_cx)
print("Current quantum state: ", decode_state(state))
print("Best gate to apply: ", Gate.decode(gate).to_string())
(fidelity_to_go, num_cx), state, gate = seq0[4:6]
print("Resultant quantum state: ", decode_state(state))
print("Next gate to apply: ", Gate.decode(gate).to_string())
