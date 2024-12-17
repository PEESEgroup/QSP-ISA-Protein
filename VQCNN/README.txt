
This folder represents the main scripts used in developing the project. Here, we
give a high level overview of the function of each of the scripts.

The CSTR folder is the parsed version of the CSTR dataset. Each data point is in
a separate file, with data point 0 in the file "0", data point 1 in the file "1"
and so on. The Train and Test folders separate the training and testing folders,
the Norm, F1, F2, and F3 folders separate the normal data, fault 1 data, fault 2
data, and fault 3 data. The meta.txt file in the Train folder is the file
storing the computed means and standard deviations across the entire training
dataset.

The vqc_model script is a script handling the construction of the variational
quantum circuit model, while the vqc_train script is a script handling functions
needed for VQC training. The layer, layers, nn, and util scripts are adapted 
from a past project where I implemented classical neural networks from scratch.
I implemented the VQC training mostly from scratch (using Cirq to simulate
quantum circuits) because the pruning method I used in this project was adapted
from a March 2021 paper so I did not expect any deep learning libraries to 
support this training method. The classical.py script was used to develop the 
classical benchmark.

The pure_vqc_two_cat script is the main script used for VQC training.
The Saves folder contains some of the final parameter states that were generated
from this script and may be loaded from pure_vqc_two_cat script if _save_prefix
is set to the appropriate file name, and _from_save is set to true.
