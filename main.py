import sys

import pickle

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

OUTPUT_FILE_NAME = "id2logits_py_dict.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append("./")


class TrainEval:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                 batch_size, num_epochs, test_dataloader=None, val_every_steps:int=5, early_stopping_steps:int=40):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = num_epochs
        self.device = device
        self.batch_size = batch_size
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
                  desc="EPOCH" + "[TRAIN]" + str(current_epoch) + "/" + str(self.epoch))

        for t, data in enumerate(tk, 1):
            
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)
            
            
            self.optimizer.zero_grad()
            
            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph)
            loss = self.criterion(return_dict["logits"], return_dict["labels"])
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / t)})
            

        return total_loss / len(self.train_dataloader)
    
    @torch.no_grad()
    def eval_fn(self, current_epoch):
        self.model.eval()
        total_loss = 0.0
        tk = tqdm(self.val_dataloader, 
                  desc="EPOCH" + "[VALID]" + str(current_epoch) + "/" + str(self.epoch))

        for t, data in enumerate(tk, 1):
            
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)

            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph)
            loss = self.criterion(return_dict["logits"], return_dict["labels"])

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / t)})


        return total_loss / len(self.val_dataloader)
    
    @torch.no_grad()
    def test(self) -> tuple[dict[int, float], dict[int, float]]:
        self.model.eval()

        def list_of_tensors_to_numpy_flat(array):
            return torch.cat(array, dim=0).cpu().numpy().reshape(-1)
            

        
        predictions: list[torch.Tensor] = [] # type: ignore
        labels: list[torch.Tensor] = [] # type: ignore
        output_nodes_indices: list[torch.Tensor] = [] # type: ignore
        
        tk = tqdm(self.test_dataloader, desc="TEST")
        
        total_loss = 0.0
        
        for t, data in enumerate(tk, 1):

            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)
            
            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph, apply_train_val_mask=False)
            
            logits = return_dict["logits"]
            true_labels = return_dict["labels"]
            
            output_nodes = data[1]
            
            loss = self.criterion(logits, true_labels)
            
            total_loss += loss.item()
            
            labels.append(true_labels.cpu())
            output_nodes_indices.append(output_nodes.cpu())
            predictions.append(logits.cpu())                        
            
            tk.set_postfix({"Loss": "%6f" % float(total_loss / t)})
            
        final_logits: np.ndarray = list_of_tensors_to_numpy_flat(predictions)
        labels: np.ndarray = list_of_tensors_to_numpy_flat(labels)
        output_nodes_indices: np.ndarray = list_of_tensors_to_numpy_flat(output_nodes_indices)
        
        id2logits: dict[int, float] = dict(zip(output_nodes_indices, final_logits))
        # id2target: dict[int, float] = dict(zip(output_nodes_indices, labels))
        
        return id2logits

    
    def train(self):
        best_valid_loss = np.inf
        best_train_loss = np.inf
        
        for i in range(1, self.epoch+1):
            train_loss = self.train_fn(i)
            
            if i % self.val_every_steps == 0:
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
            
            torch.cuda.empty_cache()
                
        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")
        
        if self.test_dataloader is not None:
            print("Performing test on test dataloader")
            
            id2logits = self.test()
            
            return id2logits
        
        return {}
        

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph Neural Network for fraud prediction")
    parser.add_argument("--datadir", type=Path, help="Directory with data", default="./data")
    parser.add_argument("--model_type", type=str, choices=["GBDT", "GNN"], 
                        default="GBDT_ready_data",
                        help="The direction of a program workflow")
    
    parser.add_argument("--mode", choices=["training", "inference"], default="training")
    
    parser.add_argument("--data_type", type=str, 
                        choices=["dglgraph", "json"],
                        help="Indicator whether or not the data is already preprocessed and stored as a graph", 
                        default="json")
    
    parser.add_argument("--remove_self_loops", action="store_true",
                        help="Whether or note to remove self loops from a graph",
                        )
    
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    return parser

TRAINING_PARAMETERS = {
        "batch_size": 30000,
        "num_epochs": 1,
        "max_num_neighbors": 50, # -1 for all neighbors to be sampled

        "learning_rate": 0.0003,
        "weight_decay": 0.00001,

        "val_every_steps": 5,
        "early_stopping_steps": 40,
        "num_workers": 12,
}


MODEL_PARAMS = {
    
        "num_hidden_features": 128,
        "normalisation_name": "batch",
        "convolution_name": "sage",
        "convolution_params":
            {
                "aggregator_type": 'mean',
                }, 
        "activation_name": 'gelu',
        "apply_skip_connection": True,
        "num_preprocessing_layers": 1,
        "num_encoder_layers": 2,
        "num_predictor_layers": 1,
        
        # "n_frequencies": 48,
        # "frequency_scale": 0.02,
        # "d_embedding": 16,

}

def main():
    args = get_parser().parse_args()
    
    #############
    # IMPORTANT #
    #############
    copy_snapshot_to_out("checkpoints")
    
    datadir: Path = args.datadir
    model_type: str = args.model_type
    data_dtype: str = args.data_type
    mode: str = args.mode
    
    print("Device is ", DEVICE)
    
    if data_dtype == "json":
        raise NotImplementedError
    else:
        graphs_filename = str(datadir / "graphs_train_val_test.bin") # this is predefined name, used for testing
        graphs, _ = load_graphs(graphs_filename) 
    
    
    if args.remove_self_loops:
        [graph_train, graph_valid, graph_test] = [remove_self_loop(g) for g in graphs]
    else:
        graph_train, graph_valid, graph_test = graphs
    
    graph_train.ndata[MASK_DATA_NAME] = graph_train.ndata[MASK_DATA_NAME].bool()
    graph_valid.ndata[MASK_DATA_NAME] = graph_valid.ndata[MASK_DATA_NAME].bool()
    graph_test.ndata[MASK_DATA_NAME] = graph_test.ndata[MASK_DATA_NAME].bool()
    
    num_input_features = graph_train.ndata[FEATURES_DATA_NAME].shape[1]
    
    print("Successfully created graphs")
    
    if model_type == "GBDT":
        print("The mode is GBDT")
        from xgboost import XGBClassifier
        from utils import get_features_and_labels_from_a_graph
        
        X_train, y_train = get_features_and_labels_from_a_graph(graph_train)
        X_val, y_val = get_features_and_labels_from_a_graph(graph_valid)
        X_test, y_test = get_features_and_labels_from_a_graph(graph_test)
        
        model = XGBClassifier()
        model.fit(X_train, y_train)
        logits = model.predict_proba(X_test)[:, 1].reshape(-1)
        id2logits = dict(zip(range(X_test.shape[0]), logits))
        
        model.save("checkpoints/xgboost_model.bin")
        
    else:
        print("The mode is GNN")
        from models.gnn_initial_and_plre import create_graph_model
        
        MODEL_PARAMS.update(dict(num_input_features=num_input_features))
        model = create_graph_model(model_name=model_type, model_params=MODEL_PARAMS).to(DEVICE)
        
        
        
        loss_func = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMETERS["learning_rate"], weight_decay=TRAINING_PARAMETERS["weight_decay"])
        
        sampler = dgl.dataloading.NeighborSampler(fanouts=[TRAINING_PARAMETERS["max_num_neighbors"]] * MODEL_PARAMS["num_encoder_layers"])

        
        batch_size = TRAINING_PARAMETERS["batch_size"]
        num_workers = TRAINING_PARAMETERS["num_workers"]
        
        train_loader = init_dataloader(graph_train, sampler, DEVICE, batch_size=batch_size, num_workers=num_workers)
        val_loader = init_dataloader(graph_valid, sampler, DEVICE, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        test_loader = init_dataloader(graph_test, sampler, DEVICE, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        
        trainer = TrainEval(model=model, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader,
                            optimizer=optimizer, criterion=loss_func, device=DEVICE, batch_size=batch_size, 
                            num_epochs=TRAINING_PARAMETERS["num_epochs"], val_every_steps=TRAINING_PARAMETERS["val_every_steps"])
        print("Initialized trainer")
        id2logits = trainer.train()
        
        
    with open(OUTPUT_FILE_NAME, "wb") as f_write:
        pickle.dump(id2logits, f_write)
        
                
if __name__ == '__main__':
    main()
