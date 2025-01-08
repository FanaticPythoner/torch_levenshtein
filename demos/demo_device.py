"""
Description:
    Demonstrates how to instantiate TorchLevenshtein on one device, then
    move all internal tensors to another device (e.g., GPU <-> CPU).
    
    This scenario can be useful if you initially prepare everything on CPU,
    then want to push computations to GPU without re-initializing.
    
Parameters:
    None

Returns:
    None
"""

import torch
from torch_levenshtein import TorchLevenshtein

def main():
    # --------------------------------------------------------------
    # STEP 1: Define concepts for some entities (CPU).
    # --------------------------------------------------------------
    concepts_per_entity = [
        ["cat", "catch", "category"],
        ["cut", "cute", "acute"],
    ]
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda") if torch.cuda.is_available() else cpu_device

    # --------------------------------------------------------------
    # STEP 2: Instantiate on CPU.
    # --------------------------------------------------------------
    tl = TorchLevenshtein(concepts_per_entity, device=cpu_device)

    # Print synergy on CPU.
    synergy_0_1_before = tl.query_entity_synergy(0, 1)
    print(f"Synergy(0,1) on CPU = {synergy_0_1_before}")

    # --------------------------------------------------------------
    # STEP 3: Move all data to the GPU (if available).
    # --------------------------------------------------------------
    if cuda_device != cpu_device:
        print("Moving TorchLevenshtein data to GPU...")
        tl.move_all_to(cuda_device)

        # Re-check synergy on GPU.
        synergy_0_1_after = tl.query_entity_synergy(0, 1)
        print(f"Synergy(0,1) on GPU = {synergy_0_1_after}")
    else:
        print("CUDA not available; staying on CPU.")

if __name__ == "__main__":
    main()
