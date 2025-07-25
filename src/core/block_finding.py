# Re-running the function and data test to validate the actual output
import pandas as pd
from itertools import combinations

import numpy as np
from itertools import combinations
import time

def find_non_overlapping_blocks(df, min_support=2, max_cols=4):
    used_rows = set()
    block_defs = []
    block_id = 0

    while True:
        candidates = []

        for r in range(max_cols, 1, -1):  # From larger to smaller combinations
            for cols in combinations(df.columns, r):
                cols = list(cols)
                temp = df[cols]
                grouped = temp.groupby(cols).size().reset_index(name="count")

                for _, row in grouped.iterrows():
                    values = row.iloc[:-1].tolist()
                    count = row["count"]

                    # Create a mask for matching rows
                    match_mask = (df[cols] == values).all(axis=1)
                    matched_indices = set(df[match_mask].index)

                    # Only accept if enough new (unused) rows match
                    new_indices = matched_indices - used_rows
                    if len(new_indices) >= min_support:
                        candidates.append({
                            "columns": cols,
                            "values": values,
                            "support": len(new_indices),
                            "matched_rows": new_indices
                        })

        if not candidates:
            break

        # Sort: prefer largest blocks, then highest support
        candidates.sort(key=lambda x: (-len(x["columns"]), -x["support"]))
        best = candidates[0]

        block_defs.append({
            "block_id": block_id,
            "columns": best["columns"],
            "values": best["values"]
        })

        used_rows.update(best["matched_rows"])
        block_id += 1

    return block_defs

def fast_blocks_numpy(df, min_support=5, max_cols=3, max_blocks=None):

    used_rows = set()
    block_defs = []
    block_id = 0
    value_maps = {}

    print(f"Starting block extraction: {df.shape[0]} rows, {df.shape[1]} columns")

    # Pre-factorize all columns
    encoded_df = {}
    for col in df.columns:
        codes, uniques = pd.factorize(df[col])
        encoded_df[col] = codes.astype(np.int32)
        value_maps[col] = uniques

    while True:
        start_iter = time.time()
        candidates = []
        total_combos = 0

        for r in range(max_cols, 1, -1):
            for cols in combinations(df.columns, r):
                col_list = list(cols)
                encoded_array = np.stack([encoded_df[col] for col in col_list], axis=1)

                uniq, counts = np.unique(encoded_array, axis=0, return_counts=True)

                for val, count in zip(uniq, counts):
                    if count < min_support:
                        continue

                    match_mask = (encoded_array == val).all(axis=1)
                    matched_indices = set(df.index[match_mask])
                    new_rows = matched_indices - used_rows

                    if len(new_rows) >= min_support:
                        decoded_vals = [
                            value_maps[col][v] for col, v in zip(col_list, val)
                        ]

                        candidates.append({
                            "columns": col_list,
                            "values": decoded_vals,
                            "support": len(new_rows),
                            "matched_rows": new_rows
                        })

                total_combos += 1

        print(f"Iteration {block_id}: {total_combos} column groups, "
              f"{len(candidates)} valid, time: {time.time() - start_iter:.2f}s")

        if not candidates:
            break

        candidates.sort(key=lambda x: (-len(x["columns"]), -x["support"]))
        best = candidates[0]

        block_defs.append({
            "block_id": block_id,
            "columns": best["columns"],
            "values": best["values"]
        })

        used_rows.update(best["matched_rows"])
        block_id += 1

        if max_blocks and block_id >= max_blocks:
            break

    print(f"Finished with {len(block_defs)} blocks.")
    return block_defs



if __name__ == "__main__":
        
    # Data from the user
    df = pd.DataFrame([
        ["A", "B", "C", "X"],
        ["A", "B", "C", "W"],
        ["A", "B", "C", "Z"],
        ["C", "D", "D", "X"],
        ["E", "D", "D", "X"],
    ], columns=["c1", "c2", "c3", "c4"])

    # Apply the function
    results = find_non_overlapping_blocks(df, min_support=2, max_cols=4)

    print(results)