import numpy as np
import h5py

#
def embedding_to_state_txt_file(embedding, output_file):
    if len(embedding) != 1024:
        raise ValueError("Embedding list must have exactly 1024 elements.")
    
    # Step 1: Calculate amplitudes and phases
    amplitudes = np.abs(embedding)
    phases = np.where(np.sign(embedding) == -1, -3.14159, 0)
    
    # Step 2: Normalize the amplitudes
    normalization_factor = np.sqrt(np.sum(amplitudes ** 2))
    normalized_amplitudes = amplitudes / normalization_factor
    
    # Step 3: Write to file
    with open(output_file, 'w') as f:
        for i in range(1024):
            binary_representation = format(i, '010b')
            amplitude = normalized_amplitudes[i]
            phase = phases[i]
            f.write(f"{binary_representation} {amplitude:.6g} {phase}\n")


#homo_sapien
num_proteins = 100
#path_to_per-protein.h5 from https://www.uniprot.org/help/downloads#embeddings
with h5py.File("/Users/rodrofougaran/Downloads/QSP_Experiments/Uniprot/per-protein.h5", "r") as file:
    print(f"number of entries: {len(file.items())}")
    items  = list(file.items())
    for i, (sequence_id, embedding) in enumerate(items[:num_proteins]):
        # print(
        #     f"  id: {sequence_id}, "
        #     f"  embeddings shape: {embedding.shape}, "
        #     f"  embedding: {np.array(embedding)}"
        # )
        embedding_to_state_txt_file(embedding,f"state{i}")

