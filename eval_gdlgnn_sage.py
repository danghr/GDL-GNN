# -*- coding: utf-8 -*-

import argparse
import os
import time
import gc
import json

import dgl
import dgl.nn as dglnn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)
from dgl.distributed import load_partition

from queue import Empty


class SAGE(nn.Module):
    def __init__(self, proc_id: int, in_size: int, hid_size: int, out_size: int):
        super().__init__()
        self.proc_id = proc_id
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(
        self,
        g: dgl.DGLGraph,
        nid: torch.Tensor,
        non_outermost_nid: torch.Tensor,
        node_feat: torch.Tensor,
        device: torch.device,
        batch_size: int,
        use_uva: bool = False,
        num_workers: int = 0,
    ):
        """Conduct layer-wise inference to get all the node embeddings."""
        non_outermost_idx = torch.concat(
            torch.nonzero(non_outermost_nid, as_tuple=True)
        )
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader1 = DataLoader(
            g,
            non_outermost_idx.to(device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            use_uva=use_uva,
        )
        dataloader2 = DataLoader(
            g,
            torch.arange(torch.sum(nid)).to(device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            use_uva=False,
        )

        feat = node_feat
        del node_feat

        for l, layer in enumerate(self.layers):
            dataloader = dataloader1 if l != len(self.layers) - 1 else dataloader2
            print(
                f"[Process {self.proc_id} model] inferring layer {l + 1} / {len(self.layers)} on {len(dataloader)} mini-batches",
                flush=True,
            )
            # Create a new tensor for the output features according to needed length
            y = torch.empty(
                torch.sum(non_outermost_nid.to(torch.long))
                if l != len(self.layers) - 1
                else torch.sum(nid.to(torch.long)),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=device,
                pin_memory=False,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                del x, input_nodes, blocks
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes] = h
                del h, output_nodes

            feat = y

            print(
                f"[Process {self.proc_id} model] inferring of layer {l + 1} / {len(self.layers)} completed",
                flush=True,
            )

        return feat


def layerwise_infer(
    proc_id: int,
    device: int,
    g: dgl.DGLGraph,
    nid: torch.Tensor,
    non_outermost_nid: torch.Tensor,
    node_feat: torch.Tensor,
    model: nn.Module,
    batch_size: int = 4096,
):
    device = torch.device(f"cuda:{device}")
    # Send node_feat first since it is the largest. Hopefully it can work.
    node_feat = node_feat.to(device, non_blocking=True)
    g = g.to(device, non_blocking=True)
    nid = nid.to(device, non_blocking=True)
    non_outermost_nid = non_outermost_nid.to(device, non_blocking=True)

    model.eval()
    with torch.no_grad():
        pred = model.inference(g, nid, non_outermost_nid, node_feat, device, batch_size)

    return pred.to("cpu")


def load_subgraph(
    proc_id: int,
    partition_queue: mp.SimpleQueue,
    next_graph_queue: mp.Queue,
    next_graph_finish_event: mp.Event,
):
    while True:
        try:
            partition = partition_queue.get()
            if partition == None:
                print(
                    f"[Process {proc_id} loading subprocess] no more partitions to load",
                    flush=True,
                )
                break
            config_json, part_id = partition
        except Exception as e:
            print(
                f"[Process {proc_id} loading subprocess] error when loading partition from the queue. Exception: {e}",
                flush=True,
            )
            exit(255)

        print(
            f"[Process {proc_id} loading subprocess] loading partition {part_id + 1}",
            flush=True,
        )

        g, node_feat, _, _, _, _, _, _ = load_partition(config_json, part_id)
        # g = dgl.add_self_loop(g)  # No need for GraphSAGE
        next_graph_queue.put((part_id, g, node_feat))

        print(
            f"[Process {proc_id} loading subprocess] partition {part_id + 1} loaded",
            flush=True,
        )

    # After all loading finished
    next_graph_queue.put(None)
    print(
        f"[Process {proc_id} loading subprocess] loading subprocess finish", flush=True
    )

    # Wait for user (model runner) to gather the result
    next_graph_finish_event.wait()
    next_graph_finish_event.clear()
    print(f"[Process {proc_id} loading subprocess] loading subprocess exit", flush=True)


def run_model(
    proc_id: int,
    device: torch.device,
    model: nn.Module,
    pred_local: list,
    labels_local: list,
    next_graph_queue: mp.Queue,
    next_graph_finish_event: mp.Event,
):
    next_graph = next_graph_queue.get()
    while next_graph != None:
        gc.collect()
        torch.cuda.empty_cache()
        (part_id, g, node_feat) = next_graph
        g.create_formats_()
        nid = node_feat["_N/local_node_mask"].to(torch.bool)
        non_outermost_nid = node_feat["_N/non_outermost_node_mask"].to(torch.bool)

        print(
            f"[Process {proc_id}] starting inference on partition {part_id + 1} with {g.num_nodes()} nodes",
            flush=True,
        )

        pred = layerwise_infer(
            proc_id,
            device,
            g,
            nid,
            non_outermost_nid,
            node_feat["_N/feat"],
            model,
            batch_size=4096,
        )

        pred_local.append(pred)
        labels_local.append(node_feat["_N/label"][nid])
        print(
            f"[Process {proc_id}] finished inference on partition {part_id + 1}",
            flush=True,
        )

        next_graph = next_graph_queue.get()

    next_graph_finish_event.set()
    print(f"[Process {proc_id}] inference finished", flush=True)


def inference_work(
    proc_id: int,
    partition_queue: mp.SimpleQueue,
    device: int,
    model: nn.Module,
    pred_local: list,
    labels_local: list,
):
    # Allow only one element in the queue for better load balance.
    # Two element holding at maximum
    # (One in the queue, the other waiting to be put into the queue)
    next_graph_queue = mp.Queue(maxsize=1)
    next_graph_finish_event = mp.Event()

    print(f"[Process {proc_id}] setting loading subprocess", flush=True)

    load_process = mp.Process(
        target=load_subgraph,
        args=(proc_id, partition_queue, next_graph_queue, next_graph_finish_event),
    )
    load_process.start()

    run_model(
        proc_id,
        device,
        model,
        pred_local,
        labels_local,
        next_graph_queue,
        next_graph_finish_event,
    )
    load_process.join()


def run(
    proc_id: int,
    nprocs: int,
    devices: tuple,
    partition_queue: mp.Queue,
    result_queue: mp.Queue,
    dataset_name: str,
    model_args: list[int, int, int],
    start_time: float,
):
    # Find corresponding device for current process
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU usage
        init_method="tcp://127.0.0.1:12347",
        world_size=nprocs,
        rank=proc_id,
    )

    print(f"[Process {proc_id}] running on GPU {device}", flush=True)
    model = SAGE(proc_id, *model_args).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(".", "model_parameter", f"dgl-SAGE-{2}-{dataset_name}.pt")
        )
    )

    pred_local, labels_local = [], []

    dist.barrier()
    if proc_id == 0:
        start_time = time.time()

    inference_work(
        proc_id,
        partition_queue,
        device,
        model,
        pred_local,
        labels_local,
    )

    pred_local = torch.cat(pred_local) if pred_local != [] else None
    labels_local = torch.cat(labels_local) if labels_local != [] else None

    if proc_id != 0:
        # Put result to the queue
        result_queue.put((pred_local, labels_local, proc_id))

        print(
            f"[Process {proc_id}] result put into the queue",
            flush=True,
        )

    if proc_id == 0:
        pred_local = [pred_local] if pred_local != None else []
        labels_local = [labels_local] if labels_local != None else []
        # Gather results from the queue and calculate accuracy

        print(
            f"[Process {proc_id}] gathering result. If you got stuck, check whether there are enough results put into the queue.",
            flush=True,
        )
        for _ in range(1, nprocs):
            try:
                pred_get, labels_get, proc_id_get = result_queue.get()
            except Empty:
                print(f"[Process {proc_id}] no enough result", flush=True)
                exit(1)
            except Exception as e:
                print(
                    f"[Process {proc_id}] error when loading results from the queue. Exception: {e}",
                    flush=True,
                )
                exit(255)

            if pred_get != None:
                pred_local.append(pred_get)
            if labels_get != None:
                labels_local.append(labels_get)
            print(
                f"[Process {proc_id}] gathered result from process {proc_id_get}",
                flush=True,
            )

        pred_local = torch.cat(pred_local)
        labels_local = torch.cat(labels_local)

        print(
            f"[Process {proc_id}] INFERENCE TIME: {time.time() - start_time}",
            flush=True,
        )

    dist.barrier()
    print(
        f"[Process {proc_id}] GPU {device} MAXIMUM MEMORY ALLOCATED: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} MB"
    )
    dist.barrier()

    if proc_id == 0:
        acc = MF.accuracy(
            pred_local,
            labels_local.to(torch.int64),
            task="multiclass",
            num_classes=model_args[-1],
            ignore_index=172 if dataset_name == "ogbn-papers100M" else None,
        ).item()

        print(f"[Process {proc_id}] TOTAL ACCURACY: {acc}", flush=True)

    print(
        f"[Process {proc_id}] waiting for other processes to finish",
        flush=True,
    )

    # Make sure that one process does not exit before others to avoid releasing
    # the element in the queue before consumer use it.
    dist.barrier()
    print(f"[Process {proc_id}] goodbye!", flush=True)


if __name__ == "__main__":
    # Maybe this helps to avoid running out of GPU memory?
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        default="0,1,2",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ogbn-products",
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Root directory of dataset.",
    )
    parser.add_argument(
        "--part_method", type=str, default="metis", help="Method of partitions."
    )
    parser.add_argument(
        "--num_hops", type=int, default=2, help="Number of hops in partitions."
    )
    parser.add_argument(
        "--num_parts", type=int, default=8, help="Number of partitions."
    )
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))

    # Create queues for distributing partitions and gathering results
    mpcontext = mp.get_context("spawn")
    partition_queue = mpcontext.SimpleQueue()
    result_queue = mpcontext.SimpleQueue()

    nprocs = len(devices)
    os.system("nvidia-smi")
    print(f"Testing GDL-GNN using {nprocs} GPU(s)")
    print(f"Model: SAGE / Dataset: {args.dataset_name}")

    start_time = time.time()

    # Load dataset metadata
    print(
        f"[Main] loading metadata for {args.dataset_name} with {args.num_parts} partitions"
    )
    dataset_dir = os.path.join(
        args.dataset_dir,
        f"{args.dataset_name}-{args.part_method}-{args.num_hops}-{args.num_parts}",
    )
    dataset_json = os.path.join(dataset_dir, f"{args.dataset_name}.json")
    with open(dataset_json) as f:
        load_dict = json.load(f)
    graph_name = load_dict["graph_name"]
    num_classes = load_dict["num_classes"]
    part_method = load_dict["part_method"]
    num_parts = load_dict["num_parts"]
    halo_hops = load_dict["halo_hops"]

    assert graph_name == args.dataset_name
    assert num_parts == args.num_parts
    assert part_method == args.part_method
    assert halo_hops == args.num_hops

    # Set parameter for models
    if graph_name == "ogbn-products":
        in_size = 100
    elif graph_name == "ogbn-papers100M":
        in_size = 128
    else:
        raise NotImplementedError(f"No trained model for dataset {graph_name}")
    model_layer = halo_hops
    hid_size, out_size = 256, num_classes
    model_args = (in_size, hid_size, out_size)

    # Put partition into queues
    for part_id in range(num_parts):
        partition_queue.put((dataset_json, part_id))
        print(
            f"[Main] putting partition {part_id + 1} / {num_parts} into the load queue",
            flush=True,
        )

    for _ in range(nprocs):
        partition_queue.put(None)

    # Start execution
    print("[Main] spawning processes")
    mp.spawn(
        run,
        args=(
            nprocs,
            devices,
            partition_queue,
            result_queue,
            args.dataset_name,
            model_args,
            start_time,
        ),
        nprocs=nprocs,
        join=True,
    )

    print("[Main] goodbye!", flush=True)
