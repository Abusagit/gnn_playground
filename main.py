import sys

from typing import Any
import argparse
from pathlib import Path
import torch
import numpy as np

from torch.hub import tqdm

import dgl
from dgl import load_graphs, remove_self_loop
from utils import FEATURES_DATA_NAME, MASK_DATA_NAME, LABELS_DATA_NAME, OUTPUT_MASK_NAME
from utils import init_dataloader, construct_subgraph_from_blocks


##################### Nirvana ##########################################
from nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot ###
##################### Nirvana ##########################################



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append("./")


class TrainEval:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 batch_size, num_epochs, predictions_test_file, test_dataloader=None, val_every_steps:int=5, early_stopping_steps:int=40):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = num_epochs
        self.device = device
        self.batch_size = batch_size
        self.predictions_test_file = predictions_test_file
        self.val_every_steps = val_every_steps
        self.early_stopping_steps = early_stopping_steps
        
        
        self.node_data_names = [FEATURES_DATA_NAME, MASK_DATA_NAME, LABELS_DATA_NAME]
    
    
    def get_logits_and_labels_for_output_nodes(self, subgraph: dgl.DGLGraph, apply_train_val_mask:bool=True):
        output_nodes_mask = subgraph.ndata[OUTPUT_MASK_NAME]
        input_features = subgraph.ndata[FEATURES_DATA_NAME]
        all_output_mask = subgraph.ndata[MASK_DATA_NAME]

        all_logits = self.model(subgraph, input_features)
        output_nodes_logits = all_logits[output_nodes_mask]
        output_nodes_train_mask = all_output_mask[output_nodes_mask]
        output_nodes_labels = subgraph.ndata[LABELS_DATA_NAME][output_nodes_mask]
        


        if apply_train_val_mask:
            logits = output_nodes_logits[output_nodes_train_mask]
            labels = output_nodes_labels[output_nodes_train_mask]
            
        else:
            logits = output_nodes_logits
            labels = output_nodes_labels
            
        return dict(
            output_nodes_train_val_mask=output_nodes_train_mask,
            logits=logits,
            labels=labels,
        )
        
        
    
    def get_subgraph_from_data(self, data) -> dgl.DGLGraph:
        _, _, layers_subgraphs = data
        
        subgraph: dgl.DGLGraph = construct_subgraph_from_blocks(blocks=layers_subgraphs, 
                                                                node_attributes_to_copy=self.node_data_names,
                                                                batch_size=self.batch_size,
                                                                device=self.device,
                                                                )
        
        return subgraph

    
        
    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = 0.0
        tk = tqdm(self.train_dataloader, 
                  desc="EPOCH" + "[TRAIN]" + str(current_epoch + 1) + "/" + str(self.epoch))

        for t, data in enumerate(tk):
            
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)
            
            
            self.optimizer.zero_grad()
            
            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph)
            loss = self.criterion(return_dict["logits"], return_dict["labels"])
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss # asynchronously
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            

        return total_loss.item() / len(self.train_dataloader)
    
    
    def eval_fn(self, current_epoch):
        self.model.eval()
        total_loss = 0.0
        tk = tqdm(self.val_dataloader, 
                  desc="EPOCH" + "[VALID]" + str(current_epoch + 1) + "/" + str(self.epoch))

        for t, data in enumerate(tk):
            
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)

            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph)
            loss = self.criterion(return_dict["logits"], return_dict["labels"])

            total_loss += loss # asynchronously
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})


        return total_loss.item() / len(self.val_dataloader)
    
    def test(self) -> tuple[dict[int, float], dict[int, float]]:
        
        def list_of_tensors_to_numpy_flat(array, apply_func:None):
            tensor = torch.cat(array, dim=0)
            if apply_func is not None:
                tensor = apply_func(tensor)
            
            return tensor.cpu().numpy().reshape(-1)
            
        self.model.eval()
        
        predictions: list[torch.Tensor] = [] # type: ignore
        labels: list[torch.Tensor] = [] # type: ignore
        output_nodes_indices: list[torch.Tensor] = [] # type: ignore
        
        tk = tqdm(self.test_dataloader, desc="TEST")
        
        total_loss = 0.0
        
        for t, data in enumerate(tk):
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)
            
            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph, apply_train_val_mask=False)
            
            logits = return_dict["logits"]
            labels = return_dict["labels"]
            
            output_nodes = data[1]
            
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() # synchronously
            
            labels.append(labels)
            output_nodes_indices.append(output_nodes)
            predictions.append(logits)
            
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            
        predictions: np.ndarray = list_of_tensors_to_numpy_flat(predictions, apply_func=torch.sigmoid)
        labels: np.ndarray = list_of_tensors_to_numpy_flat(labels)
        output_nodes_indices: np.ndarray = list_of_tensors_to_numpy_flat(output_nodes_indices)
        
        id2prediction: dict[int, float] = dict(zip(output_nodes_indices, predictions))
        id2target: dict[int, float] = dict(zip(output_nodes_indices, labels))
        
        return id2prediction, id2target

    
    def train(self):
        best_valid_loss = np.inf
        best_train_loss = np.inf
        
        for i in range(self.epoch):
            train_loss = self.train_fn(i)
            
            if i + 1 % self.val_every_steps == 0:
                val_loss = self.eval_fn(i)

            if val_loss < best_valid_loss:
                torch.save(self.model.state_dict(), "checkpoints/best-weights.pt")
                print("Saved Best Weights")
                best_valid_loss = val_loss
                best_train_loss = train_loss

                #############
                # IMPORTANT #
                #############

                copy_out_to_snapshot("checkpoints")
                
                
        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")
        
        if self.test_dataloader is not None:
            print("Performing test on test dataloader")
            
            id2prediction, id2target = self.test()
            
            return id2prediction, id2target
        
        return {}, {}
        

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph Neural Network for fraud prediction")
    parser.add_argument("--datadir", type=Path, help="File with graphs", required=True)
    parser.add_argument("--mode", type=str, choices=["GBDT", "GNN"], 
                        default="GBDT_ready_data",
                        help="The direction of a program workflow")
    
    parser.add_argument("--data_dtype", type=str, 
                        choices=["dglgraph", "json"],
                        help="Indicator whether or not the data is already preprocessed and stored as a graph", 
                        default="json")
    
    parser.add_argument("--remove_self_loops", type=bool, default=True,
                        help="Whether or note to remove self loops from a graph",
                        )
    
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    return parser

def main():
    args = get_parser().parse_args()
    
    #############
    # IMPORTANT #
    #############
    copy_snapshot_to_out("checkpoints")
    
    datadir: Path = args.datadir if args.debug else Path("./data")
    mode: str = args.mode
    data_dtype: str = args.data_dtype
    
    print("Device is ", DEVICE)
    
    if data_dtype == "json":
        raise NotImplementedError
    else:
        graphs_filename = datadir / "graphs_train_val_test.bin" # this is predefined name, used for testing
        graphs, _ = load_graphs(graphs_filename) 
    
    
    if args.remove_self_loops:
        [graph_train, graph_valid, graph_test] = [remove_self_loop(g) for g in graphs]
    else:
        graph_train, graph_valid, graph_test = graphs
    
    
    if mode == "GBDT":
        from xgboost import XGBClassifier
        
        from utils import get_features_and_labels_from_a_graph
        
        X_train, y_train = get_features_and_labels_from_a_graph(graph_train)
        X_val, y_val = get_features_and_labels_from_a_graph(graph_valid)
        X_test, y_test = get_features_and_labels_from_a_graph(graph_test)
        
        model = XGBClassifier()
        model.fit(X_train, y_train)

        logits = model.predict_proba(X_test)[:, 1]
        
        
        
    elif mode == "GNN":
        ...
        
    
    
if __name__ == '__main__':
    main()