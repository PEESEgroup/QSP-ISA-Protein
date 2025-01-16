
import os
import qc

#Compile cpp code to executable, if it hasn't been compiled already
if "isa_mod_cpp" not in os.listdir():
    os.system("./build_mod_isa")

#In the below command,
#  [output.txt] is the output file (can be named anything)
#  [state.txt] is the input file
#  [lnn.txt] is the file describing the connectivity graph
#  [5] is the number of qubits
#  [0.95] is the target fidelity. The target fidelity is an optional argument
#    with 0.95 as default value
state_file = 'state.txt'
os.system(f"./isa_mod_cpp outputNLNN.txt {state_file} lnn.txt 5 0.95")

#Now read the gates from the output file
gate_sequence = []
with open("outputNLNN.txt") as f:
    while True:
        line = f.readline()
        if len(line) == 0: break
        if len(line) < 3: continue
        [gate_type, a1, a2] = line.split()
        if gate_type == "rx": 
            gate_sequence.append(qc.Gate.RX(int(a1), float(a2), 5))
        elif gate_type == "ry":
            gate_sequence.append(qc.Gate.RY(int(a1), float(a2), 5))
        elif gate_type == "rz":
            gate_sequence.append(qc.Gate.RZ(int(a1), float(a2), 5))
        elif gate_type == "cx" or gate_type == "xc":
            gate_sequence.append(qc.Gate.CX(int(a1), int(a2), 5))
        else: raise

CX_count = 0
for gate in gate_sequence:
    # print(gate.to_string())
    if gate.gate_type == "cx" or gate.gate_type == "xc":
        CX_count += 1
print(f"Number of Gates : {len(gate_sequence)}")
print(f"CX Count: {CX_count}")
print(state_file)
