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
        """
        For each row, determine which blocks apply based on column-value patterns.
        Avoid overlapping blocks (no column reused in the same row). Not needed if blocks
        are not overlapping in the first place
        """
        for idx, row in self.df.iterrows():
            matched_blocks = []
            used_columns = set()

            for block in self.block_definitions:
                cols = block["columns"]
                vals = block["values"]

                if all(row[col] == val for col, val in zip(cols, vals)):
                    if not any(col in used_columns for col in cols): #check overlap
                        matched_blocks.append((cols, block["block_id"]))
                        used_columns.update(cols)
                        self.block_to_rows[block["block_id"]].add(idx)

            self.row_blocks.append(matched_blocks)

    def get_token_sequence(self, row_idx: int) -> List[Union[Any, List[Any]]]:
        """
        Return a list of tokens for the given row:
        TODO: check if this is what we want
        - A token can be a single value (e.g., 5, "X")
        - Or a list of values for a matched block (e.g., [1, 2, 3]) 
        """
        row = self.df.iloc[row_idx]
        blocks = self.row_blocks[row_idx]
        block_column_map = {}  # col -> block_id

        for cols, block_id in blocks:
            for col in cols:
                block_column_map[col] = block_id

        used_columns = set()
        token_sequence = []

        for col in self.column_names:
            if col in used_columns:
                continue
            if col in block_column_map:
                block_id = block_column_map[col]
                block_cols = [c for c in self.column_names if block_column_map.get(c) == block_id]
                block_values = [row[c] for c in block_cols]
                token_sequence.append(block_values)
                used_columns.update(block_cols)
            else:
                token_sequence.append(row[col])
                used_columns.add(col)

        return token_sequence

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

    # Define one block: c1 and c2 must be A and respectively
    block_definitions = [
        {"block_id": 0, "columns": ["c1", "c2"], "values": ["A", "B"]}
    ]

    data = BlockTabularData(df, block_definitions)

    for i in range(len(data)):
        tokens = data.get_token_sequence(i)
        print(f"Row {i} tokens:", tokens)

    print("Block 0 used in rows:", data.get_block_rows(0))