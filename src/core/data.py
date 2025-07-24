import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Any

class BlockTabularData:
    def __init__(self, df: pd.DataFrame, block_definitions: List[Dict]):
        """
        Parameters:
        - df: Tabular data in a dataframe
        - block_definitions: List of dicts defining blocks. Each block must contain:
            {
                "block_id": int; identifies each block,
                "columns": List[str], columns that compone the block,
                "values": List[Any], corresponding values of the block
            }
        """
        self.df = df.reset_index(drop=True)
        self.block_definitions = block_definitions
        self.column_names = list(df.columns)

        self.row_blocks = []  # List[List[Tuple[List[str], block_id]]]
        self.block_to_rows = defaultdict(set)  # Dict[int, Set[int]], maps block to rows

        self._compute_row_blocks()

    def _compute_row_blocks(self):
        for idx, row in self.df.iterrows():
            matched_blocks = []
            used_columns = set()

            for block in self.block_definitions:
                cols = block["columns"]
                vals = block["values"]

                # Skip block if any column is missing
                if any(col not in self.df.columns for col in cols):
                    continue

                valid = True
                for col, val in zip(cols, vals):
                    try:
                        if row[col] != val:
                            valid = False
                            break
                    except Exception:
                        valid = False
                        break

                if valid and not any(col in used_columns for col in cols):  # check overlap
                    matched_blocks.append((cols, block["block_id"]))
                    used_columns.update(cols)
                    self.block_to_rows[block["block_id"]].add(idx)

            self.row_blocks.append(matched_blocks)


    def get_token_sequence(self, row_idx: int) -> List[Union[Any, List[Any]]]:
        """
        Returns the token sequence for a given row.
        Instead of collapsing block columns into one token, we replicate
        the same block token at each column position it covers.
        """
        row = self.df.iloc[row_idx]
        token_seq = [None] * len(self.column_names)  # placeholder list

        # Identify which blocks apply to this row
        applicable_blocks = self.row_blocks[row_idx]  # List of (cols, block_id)

        used_columns = set()

        for cols, block_id in applicable_blocks:
            block_values = [row[col] for col in cols]
            for col in cols:
                idx = self.column_names.index(col)
                token_seq[idx] = ("BLOCK", tuple(block_values), tuple(cols), block_id)
                used_columns.add(col)

        # Fill in singleton (non-block) columns
        for i, col in enumerate(self.column_names):
            if token_seq[i] is None:
                token_seq[i] = row[col]

        return token_seq

    def get_row_block_ids(self, row_idx: int) -> List[int]:
        """Returns the block IDs that apply to a given row."""
        return [block_id for _, block_id in self.row_blocks[row_idx]]

    def get_block_rows(self, block_id: int) -> List[int]:
        """Returns the list of row indices where this block applies."""
        return list(self.block_to_rows.get(block_id, []))

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":

    df = pd.DataFrame([
        ["A", "B", "X", "Y"],  
        ["A", "B", "B", "Y"],  
        ["A", "B", "B", "Z"],  
    ], columns=["c1", "c2", "c3", "c4"])

    # Define one block: c1 and c2 must be A and B respectively
    block_definitions = [
        {"block_id": 0, "columns": ["c1", "c2"], "values": ["A", "B"]}
    ]

    data = BlockTabularData(df, block_definitions)

    for i in range(len(data)):
        tokens = data.get_token_sequence(i)
        print(f"Row {i} tokens:", tokens)

    print("Block 0 used in rows:", data.get_block_rows(0))