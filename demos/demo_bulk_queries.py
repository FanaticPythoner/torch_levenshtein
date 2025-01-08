"""
Description:
    Demonstrates bulk distance and synergy queries for a larger set
    of concept strings. Illustrates how to leverage the batch methods
    to reduce Python overhead and keep computations on the GPU.
    
Parameters:
    None

Returns:
    None
"""

import torch
from torch_levenshtein import TorchLevenshtein

def main():
    # --------------------------------------------------------------
    # STEP 1: Create synthetic data for multiple entities.
    # --------------------------------------------------------------
    entities = [
        ["king", "ring", "ping"],
        ["song", "long", "gong"],
        ["ping", "pong", "pang"],
        ["ming", "sing", "ding"],
    ]

    # --------------------------------------------------------------
    # STEP 2: Instantiate TorchLevenshtein (assume GPU if available).
    # --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl = TorchLevenshtein(entities, device=device)

    # --------------------------------------------------------------
    # STEP 3: Prepare concept pairs for bulk distance queries.
    # --------------------------------------------------------------
    c1_list = ["king", "song", "ping", "apple"]  # apple doesn't exist => distance=0
    c2_list = ["ring", "gong", "pang", "banana"] # banana doesn't exist => distance=0

    # Query distances in bulk.
    dist_bulk = tl.query_concept_distances(c1_list, c2_list)
    print("Bulk distances for concept pairs:")
    for i in range(len(c1_list)):
        print(f"  Distance({c1_list[i]}, {c2_list[i]}) = {dist_bulk[i].item()}")

    # --------------------------------------------------------------
    # STEP 4: Prepare entity pairs for synergy queries in bulk.
    # --------------------------------------------------------------
    idxs1 = [0, 0, 1, 2, 2, 3]
    idxs2 = [1, 2, 2, 0, 3, 3]
    synergy_vals = tl.query_entity_synergies(idxs1, idxs2)

    print("\nBulk synergy values for entity pairs:")
    for i in range(len(idxs1)):
        print(f"  Synergy({idxs1[i]}, {idxs2[i]}) = {synergy_vals[i].item()}")

if __name__ == "__main__":
    main()
