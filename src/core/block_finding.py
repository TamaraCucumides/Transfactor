# Re-running the function and data test to validate the actual output
import pandas as pd
from itertools import combinations

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