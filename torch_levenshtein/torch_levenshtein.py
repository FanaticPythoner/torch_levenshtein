from typing import *
import torch
import torch.nn.functional as F

# https://github.com/FanaticPythoner/torch_string_tensors/blob/main/torch_string_tensors/torch_string_tensors.py
from torch_string_tensors import *
patch_functional()


class TorchLevenshtein:
    """
    Description:
        A GPU-accelerated class to compute pairwise Levenshtein distances for
        a global set of concepts, then build an entity-level synergy matrix
        using purely tensor-based lookups (no Python dictionaries) and minimal Python overhead.
        
        The class preserves the ability to query concept distance by either string or index, and
        the same for entity synergy (by index or by a hypothetical non-integer reference).
        
        The core includes:
            1) Flattening concepts for all entities.
            2) Building membership matrix E (N x T).
            3) Building a distance matrix (T x T) via a wavefront DP approach in a batch manner.
            4) Building a synergy matrix (N x N).
        
        The synergy measure formula used is (for i != j and entities non-empty):
            synergy(i,j) = 1.0 - [ (E[i] * distance_matrix * E[j]^T) / (|i| * |j| * dist_max) ]
        
        The Levenshtein distance wavefront DP uses (for each pair of strings a, b):
            dp(i,0) = i
            dp(0,j) = j
            dp(i,j) = min( dp(i-1, j-1) + cost_sub(i,j), dp(i, j-1) + 1, dp(i-1, j) + 1 )
        where cost_sub(i,j) = 1 if a_i != b_j else 0.
        
    Parameters:
        concepts_per_entity (List[List[str]]): Nested list of concepts for each entity.
        device (torch.device): The PyTorch device on which to allocate all data.

    Returns:
        None
    """

    def __init__(self,
                 concepts_per_entity: List[List[str]],
                 device: torch.device = torch.device("cuda")) -> None:
        # --------------------------------------------------------------
        # STEP 1: Initialize instance variables and flatten concepts.
        # --------------------------------------------------------------
        # Count the number of entities.
        self.concepts_per_entity = concepts_per_entity
        # Assign the device.
        self.device = device
        # Store the number of entities.
        self.num_entities = len(concepts_per_entity)

        # ===============
        # Sub step 1.1: Prepare flattened concept list, membership indices, offsets, and unique concepts.
        # ===============
        # Prepare an empty list to collect all flattened concepts.
        flattened: List[str] = []
        # Prepare lists for membership matrix row/column IDs.
        row_ids = []
        col_ids = []
        # Prepare offsets with an initial 0.
        offsets = [0]
        # Prepare to track unique concepts.
        unique_concepts = []
        seen = set()

        # ===============
        # Sub step 1.2: Single nested loop over entities & concepts to fill structures.
        # ===============
        total_count = 0  # Will track total concept count across all entities.
        for i, cset in enumerate(concepts_per_entity):
            # For each concept in this entity's set
            for c in cset:
                # Append concept to flattened.
                flattened.append(c)
                # Record membership => row=i, col=total_count.
                row_ids.append(i)
                col_ids.append(total_count)

                # Check if concept is unique; if so, record it.
                if c not in seen:
                    seen.add(c)
                    unique_concepts.append(c)

                # Increment total_count.
                total_count += 1

            # At entity boundary => record offset in offsets list.
            offsets.append(total_count)

        # Store all unique concepts (for test compatibility).
        self.all_concepts = unique_concepts
        # Store total concept count.
        self.num_concepts_total = total_count

        # --------------------------------------------------------------
        # STEP 2: Build codes_all_2D & lengths_all from 'flattened'.
        # --------------------------------------------------------------
        # If no concepts in total, build zero-sized placeholders on device.
        if self.num_concepts_total == 0:
            # Create empty placeholders
            self.codes_all_2D = torch.zeros((0, 0), dtype=torch.uint8, device=self.device)
            self.lengths_all = torch.zeros((0,), dtype=torch.long, device=self.device)
        else:
            # Convert list of strings to PyTorch-coded strings.
            self.codes_all_2D, self.lengths_all = F.list_to_tensor(flattened, device=self.device)

        # --------------------------------------------------------------
        # STEP 3: Build membership matrix E => (N x T).
        # --------------------------------------------------------------
        self.E = torch.zeros((self.num_entities, self.num_concepts_total), dtype=torch.float32, device=self.device)
        # If there are concepts, fill the membership matrix using row/col indices.
        if row_ids:
            # Convert row_ids to a tensor.
            row_cat = torch.tensor(row_ids, dtype=torch.long, device=self.device)
            # Convert col_ids to a tensor.
            col_cat = torch.tensor(col_ids, dtype=torch.long, device=self.device)
            # Mark membership matrix entries with 1.0.
            self.E[row_cat, col_cat] = 1.0

        # Build entity offsets as a tensor.
        self.entity_offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        # Build entity sizes => difference between consecutive offsets.
        self.entity_sizes = self.entity_offsets[1:] - self.entity_offsets[:-1]

        # --------------------------------------------------------------
        # STEP 4: Prepare placeholders for distance/synergy/DP buffer and build them.
        # --------------------------------------------------------------
        self.distance_matrix: Optional[torch.Tensor] = None
        self.synergy_matrix: Optional[torch.Tensor] = None
        self.dp_buffer: Optional[torch.Tensor] = None

        # Build distance matrix immediately.
        self.build_distance_matrix()
        # Build synergy matrix immediately.
        self.build_synergy_matrix()

    # ----------------------------------------------------------------
    # Distance Matrix Construction
    # ----------------------------------------------------------------
    def build_distance_matrix(self) -> None:
        """
        Description:
            Builds the (T x T) distance matrix among all concept strings in codes_all_2D,
            using a wavefront DP in a vectorized manner. 
            The matrix is computed only on the upper triangle (off-diagonal),
            and then mirrored to form a symmetric distance matrix.

            For each pair (i, j), the Levenshtein distance is computed via _levenshtein_batch.

        Parameters:
            None

        Returns:
            None
        """
        # --------------------------------------------------------------
        # STEP 1: Edge cases for T < 2.
        # --------------------------------------------------------------
        T = self.num_concepts_total
        # If there's fewer than 2 concepts total, distance matrix is trivially zero.
        if T < 2:
            self.distance_matrix = torch.zeros((T, T), dtype=torch.float32, device=self.device)
            return

        # ===============
        # Sub step 1.1: Allocate dist_mat and compute upper triangle distances in batches.
        # ===============
        dist_mat = torch.zeros((T, T), dtype=torch.float32, device=self.device)
        # Generate indices for the upper triangle (excluding diagonal).
        idx = torch.triu_indices(T, T, offset=1, device=self.device)
        idxA, idxB = idx[0], idx[1]
        # Compute the distances for those pairs.
        dvals = self._levenshtein_batch(idxA, idxB)
        # Fill the upper triangle.
        dist_mat[idxA, idxB] = dvals
        # Mirror the values to the lower triangle.
        dist_mat[idxB, idxA] = dvals
        # Store the computed distance matrix.
        self.distance_matrix = dist_mat

    def _levenshtein_batch(self, idxA: torch.Tensor, idxB: torch.Tensor) -> torch.Tensor:
        """
        Description:
            Performs the wavefront DP for Levenshtein distance in a batch manner,
            given two sets of concept indices (idxA, idxB). 
            Each index pair (idxA[i], idxB[i]) corresponds to a pair of concept strings.
            
            The DP recurrence is:
            
                dp(i,0) = i
                dp(0,j) = j
                dp(i,j) = min(
                    dp(i-1, j) + 1,     # Deletion
                    dp(i, j-1) + 1,     # Insertion
                    dp(i-1, j-1) + cost_sub(i,j)  # Substitution cost
                )
            
            where cost_sub(i,j) = 1 if the characters differ, else 0.

        Parameters:
            idxA (torch.Tensor): 1D tensor of concept indices (for strings).
            idxB (torch.Tensor): 1D tensor of concept indices (for strings).

        Returns:
            dist_vals (torch.Tensor): 1D float32 tensor of distances, shape=(batch_size,).
        """
        # --------------------------------------------------------------
        # STEP 1: Handle empty batch case.
        # --------------------------------------------------------------
        batch_size = idxA.size(0)
        # If no pairs, return empty distance tensor.
        if batch_size == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        # --------------------------------------------------------------
        # STEP 2: Gather lengths and find max lengths.
        # --------------------------------------------------------------
        # Gather lengths for the A-side strings.
        la = self.lengths_all[idxA]
        # Gather lengths for the B-side strings.
        lb = self.lengths_all[idxB]
        # Determine max lengths.
        maxA = la.max()
        maxB = lb.max()

        # --------------------------------------------------------------
        # STEP 3: Prepare / resize the DP buffer.
        # --------------------------------------------------------------
        self._maybe_resize_dp_buffer(batch_size, maxA, maxB)
        # Slice the relevant portion from the DP buffer for this batch.
        dp = self.dp_buffer[:batch_size, :maxA+1, :maxB+1]
        # Zero out the DP buffer region to ensure a clean start.
        dp.zero_()

        # --------------------------------------------------------------
        # STEP 4: Extract the coded strings for the batch and clamp invalid positions to 0.
        # --------------------------------------------------------------
        # Clone the coded strings for A side up to maxA.
        matA = self.codes_all_2D[idxA, :maxA].clone()
        # Clone the coded strings for B side up to maxB.
        matB = self.codes_all_2D[idxB, :maxB].clone()

        # ===============
        # Sub step 4.1: Build masks to mark valid positions in each row.
        # ===============
        row_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)  # Shape (batch_size, 1).
        colA = torch.arange(maxA, device=self.device).unsqueeze(0)          # Shape (1, maxA).
        validA = colA < la[row_idx]  # Shape (batch_size, maxA). True if within length.
        colB = torch.arange(maxB, device=self.device).unsqueeze(0)          # Shape (1, maxB).
        validB = colB < lb[row_idx]  # Shape (batch_size, maxB). True if within length.

        # Clamp invalid positions to 0 to avoid random data at those places.
        matA[~validA] = 0
        matB[~validB] = 0

        # --------------------------------------------------------------
        # STEP 5: Initialize dp for base cases.
        # --------------------------------------------------------------
        # dp[:, i, 0] = i in range(maxA+1)
        dp[:, :, 0] = torch.arange(maxA+1, device=self.device, dtype=torch.int32)
        # dp[:, 0, j] = j in range(maxB+1)
        dp[:, 0, :] = torch.arange(maxB+1, device=self.device, dtype=torch.int32)

        # --------------------------------------------------------------
        # STEP 6: Wavefront DP iteration over diagonals (s in 1..maxA+maxB).
        # --------------------------------------------------------------
        for s in range(1, maxA + maxB + 1):
            # i_start = max(1, s - maxB), i_end = min(s, maxA).
            i_start = max(1, s - maxB)
            i_end = min(s, maxA)
            if i_start > i_end:
                continue
            i_seq = torch.arange(i_start, i_end+1, device=self.device, dtype=torch.int64)
            j_seq = s - i_seq
            # cost_sub => 1 if matA[:, i_seq-1] != matB[:, j_seq-1]
            cost_sub = (matA[:, i_seq-1] != matB[:, j_seq-1]).int()

            # Calculate substitution cost => dp[:, i-1, j-1] + cost_sub
            rep = dp[:, i_seq-1, j_seq-1] + cost_sub
            # Calculate insertion => dp[:, i, j-1] + 1
            ins = dp[:, i_seq, j_seq-1] + 1
            # Calculate deletion => dp[:, i-1, j_seq] + 1
            dele = dp[:, i_seq-1, j_seq] + 1
            # Final dp => min of the three.
            dp[:, i_seq, j_seq] = torch.minimum(rep, torch.minimum(ins, dele))

        # --------------------------------------------------------------
        # STEP 7: Collect final distances from dp.
        # --------------------------------------------------------------
        row_idx_final = torch.arange(batch_size, device=self.device)
        la_clamped = la.clamp_max(maxA)
        lb_clamped = lb.clamp_max(maxB)
        dist_vals = dp[row_idx_final, la_clamped, lb_clamped]
        # Convert to float32 and return.
        return dist_vals.to(torch.float32)

    def _maybe_resize_dp_buffer(self, bs: int, maxA: int, maxB: int) -> None:
        """
        Description:
            Ensures the DP buffer is large enough for a given batch size and
            maximum string lengths (maxA, maxB). If not large enough,
            allocates a new DP buffer of shape (bs, maxA+1, maxB+1).

        Parameters:
            bs (int): The batch size (number of pairs).
            maxA (int): The maximum length for A-side strings.
            maxB (int): The maximum length for B-side strings.

        Returns:
            None
        """
        # Check if dp_buffer is None or not big enough.
        if (self.dp_buffer is None
            or self.dp_buffer.size(0) < bs
            or self.dp_buffer.size(1) < maxA+1
            or self.dp_buffer.size(2) < maxB+1):
            # Allocate a new dp_buffer with needed shape.
            shape_new = (bs, maxA+1, maxB+1)
            self.dp_buffer = torch.zeros(shape_new, dtype=torch.int32, device=self.device)

    # ----------------------------------------------------------------
    # Synergy Matrix Construction
    # ----------------------------------------------------------------
    def build_synergy_matrix(self) -> None:
        """
        Description:
            Builds the synergy matrix among N entities in the range [0..1].
            
            The synergy formula for two entities i, j (if both non-empty) is:
                synergy(i,j) = 1.0 - [ (E[i] * distance_matrix * E[j]^T) / (|i| * |j| * dist_max) ]
            
            Where:
                E[i] = membership row for entity i,
                |i| = number of concepts in entity i,
                dist_max = the maximum distance in distance_matrix.
                
            Diagonal entries are set to 1.0 if the entity is non-empty, else 0.0.
            If either entity is empty, synergy(i,j) = 0 for i != j.

        Parameters:
            None

        Returns:
            None
        """
        # --------------------------------------------------------------
        # STEP 1: Handle edge cases for no entities or no concepts.
        # --------------------------------------------------------------
        n = self.num_entities
        synergy_mat = torch.zeros((n, n), dtype=torch.float32, device=self.device)

        if n == 0:
            self.synergy_matrix = synergy_mat
            return

        T = self.num_concepts_total
        # If no concepts or no distance matrix, synergy is trivial zeros (except possibly diagonals).
        if T == 0 or self.distance_matrix is None or self.distance_matrix.numel() == 0:
            self.synergy_matrix = synergy_mat
            return

        # --------------------------------------------------------------
        # STEP 2: Compute dist_max and handle the case where distances ~ 0.
        # --------------------------------------------------------------
        dist_max = self.distance_matrix.max()
        if dist_max < 1e-12:
            # All distances ~0 => synergy=1 if entity has any concepts, else 0.
            cvec = self.E.sum(dim=1)
            diag_mask = cvec.gt(0)
            synergy_mat[diag_mask, diag_mask] = 1.0
            self.synergy_matrix = synergy_mat
            return

        # --------------------------------------------------------------
        # STEP 3: Build synergy numerator => E * distance_matrix * E^T.
        # --------------------------------------------------------------
        synergy_numer = self.E @ self.distance_matrix @ self.E.T

        # --------------------------------------------------------------
        # STEP 4: Build synergy denominator => (|i| * |j| * dist_max).
        # --------------------------------------------------------------
        cvec = self.E.sum(dim=1)
        denom = (cvec.unsqueeze(1) * cvec.unsqueeze(0)) * dist_max

        # --------------------------------------------------------------
        # STEP 5: synergy_raw => synergy_numer / denom, if denom > 0, else 0.
        # --------------------------------------------------------------
        synergy_raw = torch.where(
            denom > 0,
            synergy_numer / denom,
            torch.zeros(1, device=self.device, dtype=torch.float32)
        )
        synergy_scaled = 1.0 - synergy_raw

        # --------------------------------------------------------------
        # STEP 6: Handle empty entities => synergy=0 for off-diagonal, diagonal=1 if non-empty else 0.
        # --------------------------------------------------------------
        nonempty_mask = cvec.gt(0)
        # For the diagonal, set synergy to 1 if non-empty, else 0.
        for i in range(n):
            if nonempty_mask[i]:
                synergy_scaled[i, i] = 1.0
            else:
                synergy_scaled[i, i] = 0.0

        # offdiag_mask => identifies off-diagonal positions.
        offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
        # If either entity is empty, synergy=0 for off-diagonal.
        empty_mask_2d = (~nonempty_mask.unsqueeze(1)) | (~nonempty_mask.unsqueeze(0))
        synergy_scaled = torch.where(
            empty_mask_2d & offdiag_mask,
            torch.zeros_like(synergy_scaled),
            synergy_scaled
        )

        # Clamp synergy to [0.0, 1.0].
        synergy_mat = synergy_scaled.clamp_(0.0, 1.0)
        self.synergy_matrix = synergy_mat

    # ----------------------------------------------------------------
    # Private helper to locate a concept index by string
    # ----------------------------------------------------------------
    def _find_concept_index(self, c: str) -> Optional[int]:
        """
        Description:
            Searches for the global index of a concept string 'c' among
            all T concepts. If not found, returns None. The search is
            purely tensor-based:
              1) Compare length to filter potential matches.
              2) Compare coded bytes to find exact row match.

        Parameters:
            c (str): The concept string to look up.

        Returns:
            global_row (Optional[int]): The index of the concept in the global list, or None if not found.
        """
        # --------------------------------------------------------------
        # STEP 1: If no concepts, short-circuit.
        # --------------------------------------------------------------
        T = self.num_concepts_total
        if T == 0:
            return None

        # ===============
        # Sub step 1.1: Convert the string to codes, compare lengths with lengths_all.
        # ===============
        code_t = F.string_to_tensor(c, device=self.device)  # Convert to tensor-coded string.
        c_len = code_t.size(0)
        length_mask = (self.lengths_all == c_len)
        if not torch.any(length_mask):
            return None
        candidate_rows = torch.nonzero(length_mask, as_tuple=True)[0]
        if candidate_rows.numel() == 0:
            return None

        # ===============
        # Sub step 1.2: Compare rowwise among candidate_rows.
        # ===============
        sel_codes = self.codes_all_2D[candidate_rows, :c_len]
        eq_matrix = sel_codes.eq(code_t.unsqueeze(0))  # Broadcast
        row_matches = eq_matrix.all(dim=1)
        match_indices = torch.nonzero(row_matches, as_tuple=True)[0]
        if match_indices.numel() == 0:
            return None
        global_row = candidate_rows[match_indices[0]]
        return global_row

    # ----------------------------------------------------------------
    # Concept distance queries
    # ----------------------------------------------------------------
    def query_concept_distance_by_index(self, idx1: int, idx2: int):
        """
        Description:
            Looks up the distance between two concepts by their global indices (idx1, idx2).
            If either index is out of range, returns 0.0.

        Parameters:
            idx1 (int): Global concept index of the first concept.
            idx2 (int): Global concept index of the second concept.

        Returns:
            distance (float): The Levenshtein distance. Returns 0.0 if out of range.
        """
        T = self.num_concepts_total
        # If out of range, distance=0.0.
        if not (0 <= idx1 < T and 0 <= idx2 < T):
            return 0.0
        # Otherwise, lookup in distance_matrix.
        return self.distance_matrix[idx1, idx2]

    def query_concept_distance(self, c1: str, c2: str):
        """
        Description:
            Looks up the distance between two concept strings by name.
            If either concept is not found, returns 0.0.

        Parameters:
            c1 (str): First concept string.
            c2 (str): Second concept string.

        Returns:
            distance (float): The Levenshtein distance. Returns 0.0 if concept unknown.
        """
        idx1 = self._find_concept_index(c1)
        idx2 = self._find_concept_index(c2)
        if idx1 is None or idx2 is None:
            return 0.0
        return self.query_concept_distance_by_index(idx1, idx2)

    # ----------------------------------------------------------------
    # Entity synergy queries
    # ----------------------------------------------------------------
    def query_entity_synergy(self, i: int, j: int):
        """
        Description:
            Retrieves synergy(i, j) from the synergy matrix.
            Raises IndexError if either i or j is out of range.

        Parameters:
            i (int): Index of first entity.
            j (int): Index of second entity.

        Returns:
            synergy (float): The synergy value for the (i, j) pair. 
                             In [0..1], or 0.0 if either entity is empty.
        """
        n = self.num_entities
        if i < 0 or j < 0 or i >= n or j >= n:
            raise IndexError(f"Invalid synergy query i={i}, j={j}.")
        return self.synergy_matrix[i, j]

    def query_entity_synergies(self, idxs1: List[int], idxs2: List[int]) -> torch.Tensor:
        """
        Description:
            Bulk synergy queries for multiple entity pairs. 
            Returns a 1D tensor of shape (n,) with synergy values for each pair.

        Parameters:
            idxs1 (List[int]): List of entity indices for the first slot.
            idxs2 (List[int]): List of entity indices for the second slot.

        Returns:
            out (torch.Tensor): 1D tensor of synergy values (float32).
        """
        # Ensure matching lengths.
        if len(idxs1) != len(idxs2):
            raise ValueError("idxs1, idxs2 must have same length.")
        # If synergy_matrix is not built, build it.
        if self.synergy_matrix is None:
            self.build_synergy_matrix()

        n = self.num_entities
        # Convert to tensors on device.
        i_t = torch.tensor(idxs1, device=self.device, dtype=torch.long)
        j_t = torch.tensor(idxs2, device=self.device, dtype=torch.long)

        # Build a validity mask => in range [0..n-1].
        valid_i = (i_t >= 0) & (i_t < n)
        valid_j = (j_t >= 0) & (j_t < n)
        valid_mask = valid_i & valid_j

        # Allocate output (float32) with zeros.
        out = torch.zeros_like(i_t, dtype=torch.float32)
        # For valid pairs, gather synergy from synergy_matrix.
        out[valid_mask] = self.synergy_matrix[i_t[valid_mask], j_t[valid_mask]]
        return out

    # ----------------------------------------------------------------
    # Bulk concept distance queries
    # ----------------------------------------------------------------
    def query_concept_distances(self, c1s: List[str], c2s: List[str]) -> torch.Tensor:
        """
        Description:
            Computes distances in a bulk manner for multiple concept string pairs.
            Returns a 1D tensor of shape (n,). If a concept is unknown, the distance is 0.

        Parameters:
            c1s (List[str]): List of concept strings for the first slot.
            c2s (List[str]): List of concept strings for the second slot.

        Returns:
            distances (torch.Tensor): 1D float32 tensor of distances for each pair.
        """
        # Ensure matching lengths.
        if len(c1s) != len(c2s):
            raise ValueError("c1s and c2s must have same length.")

        # Make sure distance_matrix is available.
        if self.distance_matrix is None:
            self.build_distance_matrix()

        # ===============
        # Sub step 1.1: Convert c1s, c2s to concept indices.
        # ===============
        idx1_t = self._find_concepts_indices(c1s)
        idx2_t = self._find_concepts_indices(c2s)

        # ===============
        # Sub step 1.2: Query distances by indices in a bulk manner.
        # ===============
        return self.query_concept_distances_by_indices(idx1_t, idx2_t)

    def _find_concepts_indices(self, c_list: List[str]) -> torch.Tensor:
        """
        Description:
            Converts a list of concept strings c_list to their global indices
            in a purely parallel GPU approach. If a concept is not found, 
            that position is set to -1.

            The steps are:
              1) Convert c_list to a 2D codes representation.
              2) Compare lengths with self.lengths_all to narrow down candidates.
              3) Compare coded bytes in a broadcast manner to find the first match.

        Parameters:
            c_list (List[str]): List of concept strings.

        Returns:
            chosen (torch.Tensor): 1D tensor of shape (len(c_list),) with the 
                                   matched indices or -1 if not found.
        """
        # --------------------------------------------------------------
        # STEP 1: Edge cases / early exits.
        # --------------------------------------------------------------
        n = len(c_list)
        if n == 0 or self.num_concepts_total == 0:
            return torch.full((n,), -1, dtype=torch.long, device=self.device)

        # --------------------------------------------------------------
        # STEP 2: Convert c_list -> codes2d, lengths
        # --------------------------------------------------------------
        c_codes, c_lens = F.list_to_tensor(c_list, device=self.device)

        # --------------------------------------------------------------
        # STEP 3: Build length_mask => compare with self.lengths_all
        # --------------------------------------------------------------
        T = self.num_concepts_total
        len_mask = (self.lengths_all.unsqueeze(1) == c_lens.unsqueeze(0))  # shape(T,n)

        # --------------------------------------------------------------
        # STEP 4: Rowwise content check in a fully parallel manner
        # --------------------------------------------------------------
        max_len_self = self.codes_all_2D.shape[1]
        max_len_query = c_codes.shape[1]
        max_len = max(max_len_self, max_len_query)

        # ===============
        # Sub step 4.1: Zero-pad A_exp, B_exp to shape(T,n,max_len) for eq comparison.
        # ===============
        A_exp = self.codes_all_2D.unsqueeze(1)  # shape (T,1,max_len_self)
        B_exp = c_codes.unsqueeze(0)           # shape (1,n,max_len_query)

        # Create function to zero-pad to new_len along last dimension.
        def zero_pad(src: torch.Tensor, new_len: int) -> torch.Tensor:
            # If current length >= new_len, just slice.
            cur_len = src.shape[2]
            if cur_len >= new_len:
                return src[..., :new_len]
            # Otherwise, pad the difference with zeros.
            pad_width = new_len - cur_len
            pad_shape = list(src.shape)
            pad_shape[2] = pad_width
            z = torch.zeros(pad_shape, dtype=src.dtype, device=src.device)
            return torch.cat([src, z], dim=2)

        # Apply zero_pad to shapes (T,1,big_len) and (1,n,big_len).
        A_big = zero_pad(A_exp, max_len)
        B_big = zero_pad(B_exp, max_len)

        # eq_matrix => shape(T,n,max_len) after broadcast.
        eq_matrix = A_big + 0 == B_big + 0

        # rowwise_match => shape(T,n)
        rowwise_match = eq_matrix.all(dim=2)

        # Combine length mask => final_mask => rowwise_match & len_mask
        final_mask = rowwise_match & len_mask

        # --------------------------------------------------------------
        # STEP 5: For each column, pick the first (minimal) row that matches.
        # --------------------------------------------------------------
        row_ids = torch.arange(T, device=self.device, dtype=torch.long).unsqueeze(1).expand(T, n)
        INF = T + 10000
        inf_mask = torch.where(final_mask, row_ids, torch.full_like(row_ids, INF))
        chosen = inf_mask.min(dim=0)[0]
        chosen = torch.where(chosen == INF, torch.full_like(chosen, -1), chosen)

        return chosen

    def query_concept_distances_by_indices(self, idxs1: List[int], idxs2: List[int]) -> torch.Tensor:
        """
        Description:
            Queries distances for lists of concept indices in a bulk manner.
            Out-of-range or negative indices result in 0.

        Parameters:
            idxs1 (List[int]): List of concept indices for the first slot.
            idxs2 (List[int]): List of concept indices for the second slot.

        Returns:
            out (torch.Tensor): 1D float32 tensor of distances for each pair.
        """
        # Ensure matching lengths.
        if len(idxs1) != len(idxs2):
            raise ValueError("idxs1, idxs2 must have same length.")
        # If distance_matrix is not built, build it.
        if self.distance_matrix is None:
            self.build_distance_matrix()

        # Convert to tensors on device.
        idx1_t = torch.tensor(idxs1, device=self.device, dtype=torch.long)
        idx2_t = torch.tensor(idxs2, device=self.device, dtype=torch.long)

        T = self.num_concepts_total
        valid1 = (idx1_t >= 0) & (idx1_t < T)
        valid2 = (idx2_t >= 0) & (idx2_t < T)
        valid_mask = valid1 & valid2

        # Prepare output distances, defaulting to 0.
        out = torch.zeros_like(idx1_t, dtype=torch.float32)
        # For valid pairs, gather from distance_matrix.
        out[valid_mask] = self.distance_matrix[idx1_t[valid_mask], idx2_t[valid_mask]]
        return out

    # ---------------------------------------------------------
    # Device movement
    # ---------------------------------------------------------
    def move_all_to(self, device: torch.device):
        """
        Description:
            Moves all relevant tensors (codes_all_2D, lengths_all, E, offsets, distance_matrix, synergy_matrix, dp_buffer) to a new device.

        Parameters:
            device (torch.device): The target device to move to.

        Returns:
            None
        """
        self.device = device
        # If codes_all_2D has data, move it.
        if self.codes_all_2D.numel() > 0:
            self.codes_all_2D = self.codes_all_2D.to(device)
        # If lengths_all has data, move it.
        if self.lengths_all.numel() > 0:
            self.lengths_all = self.lengths_all.to(device)
        # If E has data, move it.
        if self.E.numel() > 0:
            self.E = self.E.to(device)
        # Move offsets.
        self.entity_offsets = self.entity_offsets.to(device)
        # Move sizes.
        self.entity_sizes = self.entity_sizes.to(device)
        # If distance_matrix has data, move it.
        if self.distance_matrix is not None and self.distance_matrix.numel() > 0:
            self.distance_matrix = self.distance_matrix.to(device)
        # If synergy_matrix has data, move it.
        if self.synergy_matrix is not None and self.synergy_matrix.numel() > 0:
            self.synergy_matrix = self.synergy_matrix.to(device)
        # If dp_buffer has data, move it.
        if self.dp_buffer is not None and self.dp_buffer.numel() > 0:
            self.dp_buffer = self.dp_buffer.to(device)

    def subset_from_entities_instance(self, entity_indices: List[int]) -> "TorchLevenshtein":
        """
        Description:
            Creates a new TorchLevenshtein instance restricted to a subset of entities.
            1) Only the specified entities are kept (duplicates removed, out-of-range removed).
            2) Only the concepts used by those entities are included.
            3) Distance matrix is sliced accordingly.
            4) Synergy matrix is rebuilt for the sub-membership.

        Parameters:
            entity_indices (List[int]): Indices of entities to keep.

        Returns:
            new_tl (TorchLevenshtein): A new TorchLevenshtein object with only those entities and concepts.
        """
        e_t_raw = torch.tensor(entity_indices, device=self.device, dtype=torch.long)
        n_all = self.num_entities
        valid_mask = (e_t_raw >= 0) & (e_t_raw < n_all)
        e_t_valid = e_t_raw[valid_mask]
        e_t_uniq = torch.unique(e_t_valid, sorted=True)
        if e_t_uniq.numel() == 0:
            return TorchLevenshtein([], device=self.device)

        sub_rows = e_t_uniq
        E_mask = torch.nonzero(self.E, as_tuple=True)
        E_rows, E_cols = E_mask[0], E_mask[1]
        row_in_subset = torch.isin(E_rows, sub_rows)
        E_rows_sub = E_rows[row_in_subset]
        E_cols_sub = E_cols[row_in_subset]
        used_concepts = torch.unique(E_cols_sub, sorted=True)

        if used_concepts.numel() == 0:
            new_concepts_empty = [[] for _ in range(sub_rows.numel())]
            return TorchLevenshtein(new_concepts_empty, device=self.device)

        sub_codes_all_2D = self.codes_all_2D[used_concepts, :]
        sub_lengths_all = self.lengths_all[used_concepts]
        sub_distance = self.distance_matrix[used_concepts][:, used_concepts]

        # Build a map from old entity index -> new row index
        new_entity_index_map = {}
        for i, ent_idx in enumerate(sub_rows.tolist()):
            new_entity_index_map[ent_idx] = i

        # Build a map from old concept index -> new concept index
        new_concept_index_map = {}
        for i, c_idx in enumerate(used_concepts.tolist()):
            new_concept_index_map[c_idx] = i

        # Build sub-E
        new_num_entities = sub_rows.numel()
        new_num_concepts = used_concepts.numel()
        E_sub = torch.zeros((new_num_entities, new_num_concepts), dtype=torch.float32, device=self.device)

        old_rows_list = E_rows_sub.tolist()
        old_cols_list = E_cols_sub.tolist()
        for k in range(len(old_rows_list)):
            old_r = old_rows_list[k]
            old_c = old_cols_list[k]
            new_r = new_entity_index_map[old_r]
            new_c = new_concept_index_map[old_c]
            E_sub[new_r, new_c] = 1.0

        # Build new concepts_per_entity
        new_concepts_list: List[List[str]] = []
        for new_r_idx in range(new_num_entities):
            idx_used = torch.nonzero(E_sub[new_r_idx], as_tuple=True)[0]
            cset_local = [self.all_concepts[used_concepts[c].item()] for c in idx_used]
            new_concepts_list.append(cset_local)

        # Build offsets exactly the same way:
        new_offsets = [0]
        total_count = 0
        for ent_list_local in new_concepts_list:
            for _c in ent_list_local:
                total_count += 1
            new_offsets.append(total_count)
        new_offsets_t = torch.tensor(new_offsets, dtype=torch.long, device=self.device)
        new_sizes_t = new_offsets_t[1:] - new_offsets_t[:-1]

        new_tl = TorchLevenshtein.__new__(TorchLevenshtein)
        new_tl.concepts_per_entity = new_concepts_list
        new_tl.device = self.device
        new_tl.num_entities = new_num_entities
        new_tl.num_concepts_total = new_num_concepts
        new_tl.all_concepts = [self.all_concepts[cid] for cid in used_concepts.tolist()]
        new_tl.codes_all_2D = sub_codes_all_2D
        new_tl.lengths_all = sub_lengths_all
        new_tl.E = E_sub
        new_tl.entity_offsets = new_offsets_t
        new_tl.entity_sizes = new_sizes_t
        new_tl.distance_matrix = sub_distance
        new_tl.synergy_matrix = None
        new_tl.dp_buffer = None
        new_tl.build_synergy_matrix()
        return new_tl

    def subset_from_concepts_instance(self, concept_strings: List[str]) -> "TorchLevenshtein":
        """
        Description:
            Creates a new TorchLevenshtein instance restricted to specified concept strings.
            1) All entities are retained.
            2) Only the requested concepts are kept.
            3) Synergy matrix is rebuilt based on the partial membership.

        Parameters:
            concept_strings (List[str]): Concept strings to keep.

        Returns:
            new_tl (TorchLevenshtein): A new instance containing only those concept strings.
        """
        concept_indices_collected = []
        for c_str in concept_strings:
            idx_found = self._find_concept_index(c_str)
            if idx_found is not None:
                concept_indices_collected.append(idx_found)
        if len(concept_indices_collected) == 0:
            empty_concepts_per_entity = [[] for _ in range(self.num_entities)]
            return TorchLevenshtein(empty_concepts_per_entity, device=self.device)
        return self.subset_from_concepts_indices_instance(concept_indices_collected)

    def subset_from_concepts_indices_instance(self, concept_indices: List[int]) -> "TorchLevenshtein":
        """
        Description:
            Creates a new TorchLevenshtein instance restricted to specified concept indices.
            1) All entities are retained.
            2) The membership is pruned to only those concept indices.
            3) Synergy matrix is rebuilt.

        Parameters:
            concept_indices (List[int]): Indices of concepts to keep.

        Returns:
            new_tl (TorchLevenshtein): A new instance with only those concept indices.
        """
        T_all = self.num_concepts_total
        c_t_raw = torch.tensor(concept_indices, device=self.device, dtype=torch.long)
        valid_mask = (c_t_raw >= 0) & (c_t_raw < T_all)
        c_t_valid = c_t_raw[valid_mask]
        c_t_uniq = torch.unique(c_t_valid, sorted=True)

        if c_t_uniq.numel() == 0:
            empty_concepts_per_entity = [[] for _ in range(self.num_entities)]
            return TorchLevenshtein(empty_concepts_per_entity, device=self.device)

        sub_codes_all_2D = self.codes_all_2D[c_t_uniq, :]
        sub_lengths_all = self.lengths_all[c_t_uniq]
        sub_distance = self.distance_matrix[c_t_uniq][:, c_t_uniq]

        n_all = self.num_entities
        E_mask = torch.nonzero(self.E, as_tuple=True)
        E_rows, E_cols = E_mask[0], E_mask[1]
        col_in_subset = torch.isin(E_cols, c_t_uniq)
        E_rows_sub = E_rows[col_in_subset]
        E_cols_sub = E_cols[col_in_subset]

        # Map old column indices to new
        new_concept_index_map = {}
        for i, c_idx in enumerate(c_t_uniq.tolist()):
            new_concept_index_map[c_idx] = i

        E_sub = torch.zeros((n_all, c_t_uniq.numel()), dtype=torch.float32, device=self.device)
        old_rows_list = E_rows_sub.tolist()
        old_cols_list = E_cols_sub.tolist()
        for k in range(len(old_rows_list)):
            old_r = old_rows_list[k]
            old_c = old_cols_list[k]
            new_c = new_concept_index_map[old_c]
            E_sub[old_r, new_c] = 1.0

        new_concepts_list: List[List[str]] = []
        for row_idx in range(n_all):
            idx_used = torch.nonzero(E_sub[row_idx], as_tuple=True)[0]
            cset_local = [self.all_concepts[c_t_uniq[c].item()] for c in idx_used]
            new_concepts_list.append(cset_local)

        new_offsets = [0]
        total_count = 0
        for ent_list_local in new_concepts_list:
            for _c in ent_list_local:
                total_count += 1
            new_offsets.append(total_count)
        new_offsets_t = torch.tensor(new_offsets, dtype=torch.long, device=self.device)
        new_sizes_t = new_offsets_t[1:] - new_offsets_t[:-1]

        new_tl = TorchLevenshtein.__new__(TorchLevenshtein)
        new_tl.concepts_per_entity = new_concepts_list
        new_tl.device = self.device
        new_tl.num_entities = n_all
        new_tl.num_concepts_total = c_t_uniq.numel()
        new_tl.all_concepts = [self.all_concepts[cid] for cid in c_t_uniq.tolist()]
        new_tl.codes_all_2D = sub_codes_all_2D
        new_tl.lengths_all = sub_lengths_all
        new_tl.E = E_sub
        new_tl.entity_offsets = new_offsets_t
        new_tl.entity_sizes = new_sizes_t
        new_tl.distance_matrix = sub_distance
        new_tl.synergy_matrix = None
        new_tl.dp_buffer = None
        new_tl.build_synergy_matrix()
        return new_tl
