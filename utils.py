from copy import deepcopy
from typing import Any

import dgl
import torch


OUTPUT_MASK_NAME = "output_mask"
FEATURES_DATA_NAME = "features"
MASK_DATA_NAME = "mask"
LABELS_DATA_NAME = "labels"



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

def init_dataloader(graph: dgl.DGLGraph, 
                    sampler: dgl.dataloading.Sampler, 
                    device:str="cpu", shuffle:bool=True, 
                    batch_size:int=10_000, 
                    num_workers:int=12):

    dataloader = dgl.dataloading.DataLoader(graph=graph, indices=torch.arange(graph.num_nodes(), dtype=torch.int32), device=device,
                                            graph_sampler=sampler, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,
                                            drop_last=False, use_prefetch_thread=False, pin_prefetcher=False)
    
    return dataloader

def construct_subgraph_from_blocks(blocks: list[Any], 
                                   batch_size:int,
                                   node_attributes_to_copy: list[str],
                                   device:str,
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