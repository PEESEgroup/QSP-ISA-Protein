import cirq

def run(gates, bit=None):
    circuit = cirq.Circuit()
    for g in gates:
        circuit.append(g)
    s = cirq.Simulator()
    output = s.simulate(circuit)
    print("Raw output: ", output)
    print("Qubit mapping: ", output.qubit_map)
    print("State vector: ", output.final_state_vector)

    if bit != None:
        print("Expectation of Z on specified bit is: ")
        z = cirq.Z(bit)
        print(z.expectation_from_state_vector(
            output.final_state_vector, output.qubit_map))

