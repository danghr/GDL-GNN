# -*- coding: utf-8 -*-

import time

import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
from dgl.distributed import partition_graph


DATASET_DIR = "dataset"
DATASET_NAME = "ogbn-products"
# DATASET_NAME = "ogbn-products"
PARTITION_METHOD = "metis"
NUM_HOPS = 2

print("Loading")
dataset = AsNodePredDataset(DglNodePropPredDataset(DATASET_NAME, root="dataset"))
g = dataset[0]
print(
    f"Dataset type: {type(dataset)} / Graph type: {type(g)} / Partitionable: {isinstance(g, dgl.DGLGraph)}"
)

for NUM_PARTS in [3]:
    print(f"Partition parts: {NUM_PARTS}")

    # STORE_DIR = f"dataset/{DATASET_NAME}-{PARTITION_METHOD}-{NUM_PARTS}-allnodes"
    STORE_DIR = (
        f"dataset/{DATASET_NAME}-{PARTITION_METHOD}-{NUM_PARTS}-allnodes-outermost"
    )

    print("Partitioning...")
    start_time = time.time()
    partition_graph(
        g,
        DATASET_NAME,
        dataset.num_classes,
        NUM_PARTS,
        STORE_DIR,
        NUM_HOPS,
        PARTITION_METHOD,
    )
    print(f"Partition time: {time.time() - start_time}")
