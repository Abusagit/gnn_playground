from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import dgl
import numpy as np
import torch
import ujson
from sklearn.preprocessing import StandardScaler

OUTPUT_MASK_NAME = "output_mask"
FEATURES_DATA_NAME = "features"
MASK_DATA_NAME = "mask"
LABELS_DATA_NAME = "labels"
USERID_DATA_NAME = "userid"

# MODE_FIELD = "mode"


def _scale_features(train_val_test_features_container: list[np.ndarray], scaler_state_file: Path = None):
    if scaler_state_file.exists():
        import joblib

        scaler = joblib.load(filename=scaler_state_file)
    else:
        scaler = StandardScaler()
        scaler.fit(train_val_test_features_container[0])  # first is train

    transformed_container = [scaler.transform(features) for features in train_val_test_features_container]

    return transformed_container, scaler


def _construct_dgl_graph(adjacency_matrix, features, targets, mask, user_ids):
    row_coordinates, col_coordinates = adjacency_matrix.nonzero()
    row_coordinates, col_coordinates = torch.tensor(row_coordinates), torch.tensor(col_coordinates)

    graph = dgl.graph(data=(row_coordinates, col_coordinates))
    graph.ndata[FEATURES_DATA_NAME] = torch.tensor(features, dtype=torch.float)
    graph.ndata[LABELS_DATA_NAME] = torch.tensor(targets, dtype=torch.float)  # .reshape(-1, 1)
    graph.ndata[MASK_DATA_NAME] = torch.tensor(mask, dtype=torch.bool)  # .reshape(-1, 1)

    graph.ndata[USERID_DATA_NAME] = torch.tensor(user_ids, dtype=torch.int64)

    return graph


def get_features_and_labels_from_a_graph(graph: dgl.DGLGraph):
    mask = graph.ndata[MASK_DATA_NAME].bool()

    features = graph.ndata[FEATURES_DATA_NAME][mask].numpy()
    labels = graph.ndata[LABELS_DATA_NAME][mask].numpy()

    return features, labels


def standard_graph_collate(graph_container):
    graph = graph_container[0]

    features = graph.ndata["features"]
    labels = graph.ndata["labels"]
    mask = graph.ndata["mask"]

    return graph, features, labels, mask


def init_dataloader(
    graph: dgl.DGLGraph,
    sampler: dgl.dataloading.Sampler,
    device: str = "cpu",
    shuffle: bool = True,
    batch_size: int = 10_000,
    num_workers: int = 12,
):
    dataloader = dgl.dataloading.DataLoader(
        graph=graph,
        indices=torch.arange(graph.num_nodes(), dtype=torch.int32),
        device=device,
        graph_sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        use_prefetch_thread=False,
        pin_prefetcher=False,
    )

    return dataloader


def construct_subgraph_from_blocks(
    blocks: list[Any],
    batch_size: int,
    node_attributes_to_copy: list[str],
    device: str,
) -> dgl.DGLGraph:
    """
    Constructs a copy of a Message flow graphs (MFG), defined as a list of MFGs.

    NOTE: this function is an example of constructing graph for node classification tasks, graph obly contains node features


    params:

    `blocks`: list of consecutive message flow graphs, len(blocks) == number of layers in graph convolution
    `batch_size`: number of destination nodes
    `node_attributes_to_copy`: list of names of node attributes to copy to a new graph
    """

    merged_block = deepcopy(dgl.merge([dgl.block_to_graph(b) for b in blocks]))

    row_coords, col_coords = merged_block.edges()

    new_graph = dgl.graph(data=(row_coords, col_coords))

    for node_data_name in node_attributes_to_copy:
        new_graph.ndata[node_data_name] = merged_block.srcdata[node_data_name]

    # create mask marking only destination nodes, which are needed for
    num_of_nodes = new_graph.num_nodes()
    output_mask = torch.zeros(num_of_nodes).bool()
    output_mask[:batch_size] = True
    new_graph.ndata[OUTPUT_MASK_NAME] = output_mask.to(device)

    del merged_block

    return new_graph.to(device)


def prepare_json_input(data_dir: Path):
    scaler_state_filename: Path = data_dir / "scaler.bin"
    json_input_filename: Path = data_dir / "input_json.json"

    with open(json_input_filename) as handler:
        input_dict = ujson.load(handler)
        test_mode = input_dict["mode"]

    masks_dict: dict[str, np.ndarray] = input_dict["masks"]

    test_data_dict: dict[str, np.ndarray] = input_dict["test_data"]
    train_data_dict: Optional[dict[str, np.ndarray]] = input_dict.get(
        "train_data", "test_data"
    )  # TODO make more flexible

    train_mask = masks_dict.get("train_mask", "test_mask")
    val_mask = masks_dict.get("val_mask", "test_mask")
    test_mask = masks_dict["test_mask"]

    train_features = train_data_dict[FEATURES_DATA_NAME][train_mask]
    val_features = train_data_dict[FEATURES_DATA_NAME][val_mask]
    test_features = test_data_dict[FEATURES_DATA_NAME][test_mask]

    [train_features, val_features, test_features], scaler = _scale_features(
        train_val_test_features_container=[train_features, val_features, test_features],
        scaler_state_file=scaler_state_filename,
    )

    train_targets = val_targets = train_data_dict["targets"]
    test_targets = test_data_dict["targets"]

    train_adjacency = val_adjacency = train_data_dict["adjacency_matrix"]
    test_adjacency = test_data_dict["adjacency_matrix"]

    _zip_container = [
        [train_adjacency, train_features, train_targets, train_mask],
        [val_adjacency, val_features, val_targets, test_mask],
        [test_adjacency, test_features, test_targets, test_mask],
    ]

    graph_train, graph_valid, graph_test = (
        _construct_dgl_graph(*adj_feats_targets_mask) for adj_feats_targets_mask in zip(_zip_container)
    )

    return [graph_train, graph_valid, graph_test], scaler
