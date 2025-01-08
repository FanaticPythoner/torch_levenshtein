"""
Description:
    Demonstrates a basic usage scenario for TorchLevenshtein:
      1) Create a few entities, each with a small list of concepts.
      2) Construct a TorchLevenshtein instance.
      3) Query pairwise distances and synergies.
    
    This serves as a simple introductory example to confirm the class
    behaves as expected on small input.
    
Parameters:
    None

Returns:
    None
"""

import torch
from torch_levenshtein import TorchLevenshtein

def main():
    # --------------------------------------------------------------
    # STEP 1: Define the concepts per entity.
    # --------------------------------------------------------------
    concepts_per_entity = [
        ["apple", "banana"],
        ["banana", "bandana"],
        ["fruit", "snack"],
    ]

    # --------------------------------------------------------------
    # STEP 2: Instantiate TorchLevenshtein (on CPU for simplicity).
    # --------------------------------------------------------------
    device = torch.device("cpu")
    tl = TorchLevenshtein(concepts_per_entity, device=device)

    # --------------------------------------------------------------
    # STEP 3: Query some concept distances.
    # --------------------------------------------------------------
    d_apple_banana = tl.query_concept_distance("apple", "banana")
    d_banana_bandana = tl.query_concept_distance("banana", "bandana")
    d_apple_unknown = tl.query_concept_distance("apple", "xyz")
    print(f"Distance (apple, banana) = {d_apple_banana}")
    print(f"Distance (banana, bandana) = {d_banana_bandana}")
    print(f"Distance (apple, 'xyz') = {d_apple_unknown} (unknown => 0.0)")

    # --------------------------------------------------------------
    # STEP 4: Query entity synergy.
    # --------------------------------------------------------------
    # synergy(0,1), synergy(0,2), synergy(1,2).
    s01 = tl.query_entity_synergy(0, 1)
    s02 = tl.query_entity_synergy(0, 2)
    s12 = tl.query_entity_synergy(1, 2)
    print(f"Synergy(0, 1) = {s01}")
    print(f"Synergy(0, 2) = {s02}")
    print(f"Synergy(1, 2) = {s12}")

if __name__ == "__main__":
    main()
