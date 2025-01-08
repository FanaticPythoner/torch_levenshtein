import unittest
import torch

from torch_levenshtein import TorchLevenshtein


class TorchLevenshteinTests(unittest.TestCase):
    """
    Description:
        Test suite for TorchLevenshtein, ensuring robust coverage
        of edge cases, typical use cases, and internal method correctness.
    
    Parameters:
        None

    Returns:
        None
    """

    # --------------------------------------------------------------
    # STEP 1: Test initialization and basic class setup.
    # --------------------------------------------------------------

    def test_init_empty_entities(self):
        """
        Description:
            Tests that providing an empty list of entities works gracefully.
            No mathematical formula is used in this test.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 1.1: Create the TorchLevenshtein object with no entities.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[])  # Create object with empty entity list.

        # ===============
        # Sub step 1.2: Check that all_concepts is empty.
        # ===============
        self.assertEqual(len(wfl.all_concepts), 0,
                         "all_concepts should be empty for no entities.")

    def test_init_single_entity_no_concepts(self):
        """
        Description:
            Tests that a single entity with no concepts doesn't break initialization.
            No mathematical formula is used in this test.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 1.1: Create the TorchLevenshtein object with one entity, no concepts.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[[]])  # Single entity, no concepts.

        # ===============
        # Sub step 1.2: Check that all_concepts is empty.
        # ===============
        self.assertEqual(len(wfl.all_concepts), 0,
                         "No concepts in a single entity means empty vocabulary.")

    def test_init_with_some_concepts(self):
        """
        Description:
            Tests initialization with multiple entities that do have concepts.
            No mathematical formula is used in this test.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 1.1: Prepare test data with multiple entities.
        # ===============
        concepts_per_entity = [
            ["apple", "banana", "orange"],
            ["banana", "kiwi"],
            ["grape", "apple"],
        ]

        # ===============
        # Sub step 1.2: Create the TorchLevenshtein object.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=concepts_per_entity)

        # ===============
        # Sub step 1.3: Check that expected concepts are in the final vocabulary.
        # ===============
        self.assertIn("apple", wfl.all_concepts, "Vocabulary must include 'apple'.")
        self.assertIn("banana", wfl.all_concepts, "Vocabulary must include 'banana'.")
        self.assertIn("orange", wfl.all_concepts, "Vocabulary must include 'orange'.")
        self.assertIn("kiwi", wfl.all_concepts, "Vocabulary must include 'kiwi'.")
        self.assertIn("grape", wfl.all_concepts, "Vocabulary must include 'grape'.")

    # --------------------------------------------------------------
    # STEP 2: Test the distance matrix build process.
    # --------------------------------------------------------------

    def test_build_distance_matrix_empty_vocab(self):
        """
        Description:
            Verifies that building a distance matrix with an empty vocabulary
            completes without error and results in a 0x0 matrix.
            No mathematical formula is used in this test.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 2.1: Create the object with no entities.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[])  # Create object with empty entity list.

        # ===============
        # Sub step 2.2: Build the distance matrix.
        # ===============
        wfl.build_distance_matrix()  # Build an empty distance matrix.

        # ===============
        # Sub step 2.3: Check the distance_matrix shape.
        # ===============
        self.assertIsNotNone(wfl.distance_matrix,
                             "distance_matrix should not be None after building.")
        self.assertEqual(wfl.distance_matrix.shape, (0, 0),
                         "distance_matrix must be (0, 0) for empty vocab.")

    def test_build_distance_matrix_single_concept(self):
        """
        Description:
            Tests distance matrix construction when there's only one concept total.
            Verifies the diagonal is 0 for distance to itself.

            Levenshtein distance formula (dp approach):
                dp(i,j) = min of [
                    dp(i-1, j) + 1,
                    dp(i, j-1) + 1,
                    dp(i-1, j-1) + cost
                ]

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 2.1: Create the object with a single concept.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[["hello"]])  # Single concept.

        # ===============
        # Sub step 2.2: Build distance matrix.
        # ===============
        wfl.build_distance_matrix()  # Build the distance matrix for 1 concept.

        # ===============
        # Sub step 2.3: Check shape and diagonal distance.
        # ===============
        self.assertEqual(wfl.distance_matrix.shape, (1, 1),
                         "Distance matrix must be (1,1) if we have one concept.")
        self.assertAlmostEqual(float(wfl.distance_matrix[0, 0].item()), 0.0,
                               msg="Distance to itself must be 0.")

    def test_build_distance_matrix_multiple_concepts(self):
        """
        Description:
            Tests distance matrix creation with multiple concepts and verifies
            distances for known pairs (cat, cut, cast).

            Example Levenshtein distances:
                dist(cat, cat)  = 0
                dist(cat, cut)  = 1
                dist(cat, cast) = 1
                dist(cut, cast) = 2

            Generic Levenshtein distance formula (dp approach):
                dp(i,j) = min of [
                    dp(i-1, j) + 1,
                    dp(i, j-1) + 1,
                    dp(i-1, j-1) + cost
                ]

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 2.1: Create the object with multiple concepts (cat, cut, cast).
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[["cat"], ["cut"], ["cast"]])

        # ===============
        # Sub step 2.2: Build the distance matrix.
        # ===============
        wfl.build_distance_matrix()  # Build matrix for cat, cut, cast.
        dist_mat = wfl.distance_matrix  # Store reference for checks.

        # ===============
        # Sub step 2.3: Check shape => (3,3).
        # ===============
        self.assertEqual(dist_mat.shape, (3, 3),
                         "Must have 3 concepts => distance matrix of shape (3,3).")

        # ===============
        # Sub step 2.4: Validate diagonal distances (0).
        # ===============
        self.assertEqual(float(dist_mat[0, 0]), 0.0, "cat->cat must be 0.")
        self.assertEqual(float(dist_mat[1, 1]), 0.0, "cut->cut must be 0.")
        self.assertEqual(float(dist_mat[2, 2]), 0.0, "cast->cast must be 0.")

        # ===============
        # Sub step 2.5: Validate off-diagonal distances (cat->cut=1, cat->cast=1, cut->cast=2).
        # ===============
        self.assertEqual(float(dist_mat[0, 1]), 1.0, "cat->cut must be distance 1.")
        self.assertEqual(float(dist_mat[1, 0]), 1.0, "cut->cat must be distance 1.")
        self.assertEqual(float(dist_mat[0, 2]), 1.0, "cat->cast must be distance 1.")
        self.assertEqual(float(dist_mat[2, 0]), 1.0, "cast->cat must be distance 1.")
        self.assertEqual(float(dist_mat[1, 2]), 2.0, "cut->cast must be distance 2.")
        self.assertEqual(float(dist_mat[2, 1]), 2.0, "cast->cut must be distance 2.")

    # --------------------------------------------------------------
    # STEP 3: Test the synergy matrix build process.
    # --------------------------------------------------------------

    def test_build_synergy_matrix_no_entities(self):
        """
        Description:
            Ensures building synergy with zero entities results in a 0x0 synergy matrix.
            No mathematical formula is used in this test.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 3.1: Create the object with no entities.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[])  # Empty entities.

        # ===============
        # Sub step 3.2: Build synergy matrix.
        # ===============
        wfl.build_synergy_matrix()  # Build synergy matrix for no entities.

        # ===============
        # Sub step 3.3: Check synergy matrix shape => (0,0).
        # ===============
        self.assertIsNotNone(wfl.synergy_matrix, "synergy_matrix should be allocated.")
        self.assertEqual(wfl.synergy_matrix.shape, (0, 0),
                         "synergy_matrix must be (0,0) with no entities.")

    def test_build_synergy_matrix_no_concepts(self):
        """
        Description:
            Tests synergy matrix creation when there are entities but no concepts.

            The synergy formula (ASCII) for entities i, j is:
                synergy(i,j) = 1.0 - [ (E[i] * distance_matrix * E[j]^T) / (|i| * |j| * dist_max) ]

            If no concepts, synergy=0.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 3.1: Create with 3 entities, each has no concepts.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[[], [], []])  # 3 empty entities.

        # ===============
        # Sub step 3.2: Build synergy matrix.
        # ===============
        wfl.build_synergy_matrix()  # Build synergy with no concepts.

        # ===============
        # Sub step 3.3: Check shape => (3,3) and that all are zeros.
        # ===============
        self.assertEqual(wfl.synergy_matrix.shape, (3, 3),
                         "We have 3 entities => synergy_matrix shape (3,3).")
        self.assertTrue(torch.all(wfl.synergy_matrix.eq(0)),
                        "No concepts => synergy=0 across entire matrix.")

    def test_build_synergy_matrix_partial_overlap(self):
        """
        Description:
            Checks synergy for partial overlap of concepts with distance-based scaling.
            
            The synergy formula (ASCII) is:
                synergy(i,j) = 1.0 - [ (E[i] * distance_matrix * E[j]^T) / (|i| * |j| * dist_max) ]

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 3.1: Define entities for partial overlap.
        # ===============
        # E0 => [apple, banana]
        # E1 => [banana, kiwi]
        # E2 => [grape]
        wfl = TorchLevenshtein(
            concepts_per_entity=[
                ["apple", "banana"],
                ["banana", "kiwi"],
                ["grape"]
            ]
        )

        # ===============
        # Sub step 3.2: Build distance and synergy matrices.
        # ===============
        wfl.build_distance_matrix()  # Build distance matrix for all concepts.
        wfl.build_synergy_matrix()   # Build synergy matrix using that distance.

        # ===============
        # Sub step 3.3: Check synergy matrix shape and values.
        # ===============
        s = wfl.synergy_matrix  # Reference to synergy matrix.
        self.assertEqual(s.shape, (3, 3), "We have 3 entities => synergy matrix (3,3).")
        self.assertAlmostEqual(float(s[0, 0]), 1.0, "Diagonal synergy=1 if entity has concepts.")
        self.assertAlmostEqual(float(s[1, 1]), 1.0, "Diagonal synergy=1 if entity has concepts.")
        self.assertAlmostEqual(float(s[2, 2]), 1.0, "Diagonal synergy=1 if entity has concepts.")

        # ===============
        # Sub step 3.4: Check off-diagonal synergy => in [0,1].
        # ===============
        for i in range(3):
            for j in range(3):
                if i != j:
                    val = float(s[i, j])
                    self.assertGreaterEqual(val, 0.0, "Synergy must be >=0.")
                    self.assertLessEqual(val, 1.0, "Synergy must be <=1.")

    def test_build_synergy_matrix_exact_distances(self):
        """
        Description:
            Verifies synergy is in [0,1] when distances are known (cat, cut, cast).

            Synergy formula (ASCII):
                synergy(i,j) = 1.0 - [ (E[i] * distance_matrix * E[j]^T ) / (|i| * |j| * dist_max ) ]

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 3.1: Create with 3 entities: [cat], [cut], [cast, cat].
        # ===============
        wfl = TorchLevenshtein(
            concepts_per_entity=[["cat"], ["cut"], ["cast", "cat"]]
        )

        # ===============
        # Sub step 3.2: Build distance and synergy matrices.
        # ===============
        wfl.build_distance_matrix()  # Build distance matrix for cat, cut, cast.
        wfl.build_synergy_matrix()   # Build synergy matrix.

        # ===============
        # Sub step 3.3: Check synergy matrix shape and diagonal synergy.
        # ===============
        s = wfl.synergy_matrix  # Reference synergy matrix.
        self.assertEqual(s.shape, (3, 3),
                         "We have 3 entities => synergy_matrix shape (3,3).")
        self.assertAlmostEqual(float(s[0, 0]), 1.0)
        self.assertAlmostEqual(float(s[1, 1]), 1.0)
        self.assertAlmostEqual(float(s[2, 2]), 1.0)

        # ===============
        # Sub step 3.4: Check off-diagonal synergy => [0,1].
        # ===============
        for i in range(3):
            for j in range(3):
                if i != j:
                    val = float(s[i, j])
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)

    # --------------------------------------------------------------
    # STEP 4: Test query functions.
    # --------------------------------------------------------------

    def test_query_concept_distance_unknown(self):
        """
        Description:
            Tests query_concept_distance when one or both concepts are unknown.
            
            Levenshtein distance would normally apply if both concepts are known:
            dp(i,j) = min of [
                dp(i-1, j) + 1,
                dp(i, j-1) + 1,
                dp(i-1, j-1) + cost
            ]
            
            If either concept is missing (unknown), distance=0.0 by current definition.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 4.1: Create the object with known concepts.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[["foo", "bar"]])  # Known: foo, bar.

        # ===============
        # Sub step 4.2: Build distance matrix.
        # ===============
        wfl.build_distance_matrix()  # Build distances for foo, bar.

        # ===============
        # Sub step 4.3: Query distances.
        # ===============
        known_dist = wfl.query_concept_distance("foo", "bar")  # Known concepts distance.
        unknown_dist = wfl.query_concept_distance("foo", "baz")  # One unknown.
        both_unknown = wfl.query_concept_distance("aaa", "bbb")  # Both unknown.

        # ===============
        # Sub step 4.4: Validate results.
        # ===============
        self.assertGreaterEqual(known_dist, 1.0,
                                "Distance between 'foo' and 'bar' must be >= 1.")
        self.assertEqual(unknown_dist, 0.0,
                         "If either concept is missing, distance=0.0 by definition.")
        self.assertEqual(both_unknown, 0.0,
                         "If both concepts are unknown, distance=0.0 by definition.")

    def test_query_entity_synergy_unknown_index(self):
        """
        Description:
            Tests query_entity_synergy with invalid entity indices, ensuring it raises IndexError.
            No mathematical formula is used in this test.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 4.1: Create with 2 entities => indices 0, 1 only.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[["cat"], ["dog"]])

        # ===============
        # Sub step 4.2: Build synergy matrix.
        # ===============
        wfl.build_synergy_matrix()  # Build synergy for cat, dog.

        # ===============
        # Sub step 4.3: Try to query out-of-range indices.
        # ===============
        with self.assertRaises(IndexError):
            _ = wfl.query_entity_synergy(2, 0)  # entity index 2 does not exist

        with self.assertRaises(IndexError):
            _ = wfl.query_entity_synergy(0, 2)  # entity index 2 does not exist

    def test_query_entity_synergy_ok(self):
        """
        Description:
            Tests query_entity_synergy with valid indices after synergy is built.
            
            synergy(i,i) = 1 if entity i has >=1 concept, else 0.
            synergy(i,j) = 0 if there is no concept overlap and there's at least some distance cost.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 4.1: Create with 2 entities, each has one distinct concept => synergy=0 between them.
        # ===============
        wfl = TorchLevenshtein(concepts_per_entity=[["cat"], ["dog"]])

        # ===============
        # Sub step 4.2: Build synergy matrix.
        # ===============
        wfl.build_synergy_matrix()  # Build synergy for cat, dog.

        # ===============
        # Sub step 4.3: Check self synergy => 1 if entity has concepts.
        # ===============
        self.assertAlmostEqual(wfl.query_entity_synergy(0, 0), 1.0,
                               msg="Self synergy must be 1.")
        self.assertAlmostEqual(wfl.query_entity_synergy(1, 1), 1.0,
                               msg="Self synergy must be 1.")

        # ===============
        # Sub step 4.4: Check synergy(0,1) => no overlap => synergy=0.
        # ===============
        synergy_01 = wfl.query_entity_synergy(0, 1)
        synergy_10 = wfl.query_entity_synergy(1, 0)
        self.assertAlmostEqual(synergy_01, 0.0,
                               msg="cat & dog => no shared => synergy=0.")
        self.assertAlmostEqual(synergy_10, 0.0,
                               msg="Symmetry => synergy=0.")

    def test_query_entity_synergies_bulk(self):
        """
        Description:
            Tests the bulk synergy query (query_entity_synergies) with valid and invalid indices.
            Checks that out-of-range indices yield 0 synergy, while valid pairs match synergy_matrix.

            synergy(i,j) = synergy_matrix[i,j].
            If i or j is invalid, synergy=0 in the returned result.

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 4.1: Create scenario with 3 entities: [cat], [cut], [cast, cat].
        # ===============
        wfl = TorchLevenshtein(
            concepts_per_entity=[["cat"], ["cut"], ["cast", "cat"]]
        )

        # ===============
        # Sub step 4.2: Build distance and synergy matrices.
        # ===============
        wfl.build_distance_matrix()  # Build distances for cat, cut, cast.
        wfl.build_synergy_matrix()   # Build synergy matrix.

        # ===============
        # Sub step 4.3: Define pairs of entity indices for query.
        # ===============
        idxs1 = [0, 1, 2, 2, 999, -1]
        idxs2 = [0, 2, 2, 1, 1,  1]

        # ===============
        # Sub step 4.4: Bulk synergy query.
        # ===============
        synergies = wfl.query_entity_synergies(idxs1, idxs2)  # Torch operation returning synergy values.

        # ===============
        # Sub step 4.5: Build expected synergy values (or 0 if out-of-range).
        # ===============
        expected = []
        n = wfl.num_entities
        for i, j in zip(idxs1, idxs2):
            if i < 0 or j < 0 or i >= n or j >= n:
                expected.append(0.0)
            else:
                expected.append(float(wfl.synergy_matrix[i, j]))

        # ===============
        # Sub step 4.6: Compare results with expected.
        # ===============
        expected_t = torch.tensor(expected, dtype=torch.float32, device=wfl.device)  # Create tensor for comparison.
        self.assertEqual(synergies.shape, expected_t.shape,
                         "Output shape must match number of pairs.")
        self.assertTrue(torch.allclose(synergies, expected_t, atol=1e-5, rtol=1e-5),
                        "Bulk synergy results must match synergy_matrix or 0 for invalid indices.")

    def test_query_concept_distances_bulk(self):
        """
        Description:
            Tests the bulk concept distance query (query_concept_distances) with known and unknown concepts.
            Ensures unknown concepts yield distance=0, while known pairs match computed distances.

            Levenshtein distance formula (dp approach):
                dp(i,j) = min of [
                    dp(i-1, j) + 1,
                    dp(i, j-1) + 1,
                    dp(i-1, j-1) + cost
                ]

        Parameters:
            None

        Returns:
            None
        """
        # ===============
        # Sub step 4.1: Create scenario with cat, cut, cast. Build distances.
        # ===============
        wfl = TorchLevenshtein(
            concepts_per_entity=[["cat", "cut", "cast"]]
        )
        wfl.build_distance_matrix()  # Build distance matrix for cat, cut, cast.

        # ===============
        # Sub step 4.2: Pairs of concepts for query (some unknown).
        # ===============
        c1s = ["cat", "cat", "cut", "cast", "foo", "cat"]
        c2s = ["cat", "cut", "cat", "cut",  "bar",  "zzz"]  # "foo","bar","zzz" => unknown

        # ===============
        # Sub step 4.3: Bulk distance query.
        # ===============
        distances = wfl.query_concept_distances(c1s, c2s)  # Torch operation returning distances.

        # ===============
        # Sub step 4.4: Compute expected distances individually.
        # ===============
        expected = []
        for x, y in zip(c1s, c2s):
            expected.append(wfl.query_concept_distance(x, y))  # Use single query to confirm.

        # ===============
        # Sub step 4.5: Compare the results with expected.
        # ===============
        expected_t = torch.tensor(expected, dtype=torch.float32, device=wfl.device)  # Create comparison tensor.
        self.assertEqual(distances.shape, expected_t.shape,
                         "Output shape must match the number of concept pairs.")
        self.assertTrue(torch.allclose(distances, expected_t, atol=1e-5, rtol=1e-5),
                        "Bulk distances must match individual query_concept_distance results.")

if __name__ == "__main__":
    unittest.main()
