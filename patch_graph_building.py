import csv
import pandas as pd
from ucimlrepo import fetch_ucirepo
from typing import List, Tuple, Optional, Set
from itertools import combinations
from collections import defaultdict

import networkx as nx


def read_csv_as_columns(file_path: str, has_header: bool = True) -> List[List[str]]:
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        if has_header:
            next(reader)  # Skip the header row
        rows = list(reader)
    
    if not rows:
        return []
    return [list(col) for col in zip(*rows)]


def dataframe_to_columns(df: pd.DataFrame) -> List[List[str]]:
    # convert df to table, each row in the table is one column in the df
    return [df[col].astype(str).tolist() for col in df.columns]


def check_patch_validity(column_set: Tuple[List[str], ...], row_indices: Tuple[int, ...]) -> Tuple[bool, List[str]]:
    first_row = [column[row_indices[0]] for column in column_set]
    for other_row_index in row_indices[1:]:
        other_row = [column[other_row_index] for column in column_set]
        if first_row != other_row:
            return False, []
    return True, first_row


def process_table(table: List[List[str]], min_support: int = 2) -> List[Tuple[Tuple[str, ...], List[int]]]:
    possible_patch_count = 0
    illegal_patch_count = 0
    max_patch_size = 0
    max_patch: Optional[List[str]] = None

    # Dictionary: patch → set of row indices that support it
    patch_counter: defaultdict[Tuple[str, ...], Set[int]] = defaultdict(set)

    for nr_columns in range(2, len(table) + 1):
        for nr_rows in range(2, len(table[0]) + 1):
            for column_set in combinations(table, nr_columns):
                possible_rows = list(range(len(table[0])))
                for row_indices in combinations(possible_rows, nr_rows):
                    possible_patch_count += 1
                    valid, patch = check_patch_validity(column_set, row_indices)
                    if valid:
                        patch_key = tuple(patch)
                        patch_counter[patch_key].update(row_indices)

                        patch_size = (nr_columns - 1) * (nr_rows - 1) - 1
                        if patch_size > max_patch_size:
                            max_patch_size = patch_size
                            max_patch = patch
                    else:
                        illegal_patch_count += 1

    print("Possible patches checked:", possible_patch_count)
    print("Invalid patches found:", illegal_patch_count)
    print("Max patch size:", max_patch_size)
    print("Max patch:", max_patch if max_patch else "No patch found")

    # Filter and sort patches by frequency (support size)
    filtered_patches = [(p, list(s)) for p, s in patch_counter.items() if len(s) >= min_support]
    sorted_patches = sorted(filtered_patches, key=lambda x: len(x[1]), reverse=True)

    print(f"Top {min(10, len(sorted_patches))} patches by support:")
    for i, (patch, support) in enumerate(sorted_patches[:10]):
        print(f"Patch {i}: {patch}, support size: {len(support)}")

    return sorted_patches

def create_patch_graph(
    num_instances: int,
    patches_with_support: List[Tuple[Tuple[str, ...], List[int]]]
) -> nx.Graph:
    G = nx.Graph()

    # Add instance nodes (one per instance/row)
    for i in range(num_instances):
        G.add_node(f"inst_{i}", type="instance", index=i)

    # Add patch nodes and connect to supporting instances (rows-nodes)
    for patch_id, (patch_pattern, supporting_rows) in enumerate(patches_with_support):
        patch_node = f"patch_{patch_id}"
        G.add_node(patch_node, type="patch", pattern=patch_pattern)

        for row_id in supporting_rows:
            instance_node = f"inst_{row_id}"
            ## can make different edge types per patch?
            G.add_edge(patch_node, instance_node, relation="supports")

    return G


if __name__ == "__main__":
    #mushroom = fetch_ucirepo(id=73) 
    #df_mushroom = mushroom.data.features 
    #task = mushroom.data.targets 
    #table = dataframe_to_columns(df_mushroom)

    table = [
    ["a1", "a1", "a1", "a1", "a1", "a1"],
    ["b1", "b1", "b1", "b2", "b2", "b3"],
    ["c1", "c1", "c2", "c3", "c3", "c2"],
    ["d1", "d2", "d2", "d2", "d2", "d2"],
    ["e1", "e2", "e3", "e4", "e5", "e6"]]


    patches = process_table(table, min_support=2)
    num_instances = len(table[0])  # one per column

    G = create_patch_graph(num_instances, patches)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print("Some edges:")
    for u, v, d in list(G.edges(data=True))[:10]:
        print(f"{u} --[{d['relation']}]--> {v}")
