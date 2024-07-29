import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional, Callable

import dgl
import joblib
import numpy as np
import torch
from dgl import load_graphs, remove_self_loop
from torch.hub import tqdm
import ujson
import pandas as pd

##################### Nirvana ##########################################
from nirvana_utils import copy_out_to_snapshot, copy_snapshot_to_out  ###
from utils import (
    FEATURES_DATA_NAME,
    LABELS_DATA_NAME,
    MASK_DATA_NAME,
    OUTPUT_MASK_NAME,
    USERID_DATA_NAME,
    construct_subgraph_from_blocks,
    init_dataloader,
    write_output_to_YT,
)

##################### Nirvana ##########################################

OUTPUT_FILE_NAME = "index2logit"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append("./")


class TrainEval:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        device,
        batch_size,
        num_epochs,
        mode,
        test_dataloader=None,
        val_every_steps: int = 5,
        early_stopping_steps: int = 40,
        state_dict_file: Optional[str] = None,
    ):
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

        self.node_data_names = [FEATURES_DATA_NAME, MASK_DATA_NAME, LABELS_DATA_NAME, USERID_DATA_NAME]

        self.mode = mode

        if state_dict_file is not None:
            state_dict: Mapping[str, torch.FloatTensor] = torch.load(state_dict_file, map_location=device)
            self.model.load_state_dict(state_dict)
            print("State dict is loaded")

    def get_logits_and_labels_for_output_nodes(self, subgraph: dgl.DGLGraph, apply_train_val_mask: bool = True):
        output_nodes_mask = subgraph.ndata[OUTPUT_MASK_NAME]
        input_features = subgraph.ndata[FEATURES_DATA_NAME]
        all_output_mask = subgraph.ndata[MASK_DATA_NAME]

        all_logits = self.model(subgraph, input_features)
        
        output_nodes_train_mask = all_output_mask[output_nodes_mask]
        
        output_nodes_logits = all_logits[output_nodes_mask]
        output_nodes_labels = subgraph.ndata[LABELS_DATA_NAME][output_nodes_mask]
        output_nodes_ids = subgraph.ndata[USERID_DATA_NAME][output_nodes_mask]

        # if apply_train_val_mask:
        #     logits = output_nodes_logits[output_nodes_train_mask]
        #     labels = output_nodes_labels[output_nodes_train_mask]
        #     ids = output_nodes_ids[output_nodes_train_mask]

        # else:
        #     logits = output_nodes_logits
        #     labels = output_nodes_labels
        #     ids = output_nodes_ids
        
        # NOTE need to fix this to be able to compare results with previous
        logits = output_nodes_logits[output_nodes_train_mask]
        labels = output_nodes_labels[output_nodes_train_mask]
        ids = output_nodes_ids[output_nodes_train_mask]

        return dict(
            output_nodes_train_val_mask=output_nodes_train_mask,
            logits=logits,
            labels=labels,
            ids=ids,
        )

    def get_subgraph_from_data(self, data) -> dgl.DGLGraph:
        _, _, layers_subgraphs = data

        subgraph: dgl.DGLGraph = construct_subgraph_from_blocks(
            blocks=layers_subgraphs,
            node_attributes_to_copy=self.node_data_names,
            batch_size=self.batch_size,
            device=self.device,
        )

        return subgraph

    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = 0.0
        tk = tqdm(self.train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch) + "/" + str(self.epoch))

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
        tk = tqdm(self.val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch) + "/" + str(self.epoch))

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

        def list_of_tensors_to_numpy_flat(array, apply_func: Optional[Callable[[torch.Tensor], torch.Tensor]]=None):
            plain_array = torch.cat(array, dim=0).cpu().reshape(-1)
            
            if apply_func is not None:
                plain_array = apply_func(plain_array)
            return plain_array.numpy()

        predictions: list[torch.Tensor] = []  # type: ignore
        labels: list[torch.Tensor] = []  # type: ignore
        output_nodes_ids: list[torch.Tensor] = []  # type: ignore

        tk = tqdm(self.test_dataloader, desc="TEST")

        total_loss = 0.0

        for t, data in enumerate(tk, 1):
            subgraph: dgl.DGLGraph = self.get_subgraph_from_data(data)

            return_dict = self.get_logits_and_labels_for_output_nodes(subgraph, apply_train_val_mask=False)

            logits = return_dict["logits"]
            true_labels = return_dict["labels"]
            output_batch_ids = return_dict["ids"]


            loss = self.criterion(logits, true_labels)

            total_loss += loss.item()

            labels.append(true_labels.cpu())
            
            output_nodes_ids.append(output_batch_ids.cpu())

            
            predictions.append(logits.cpu())

            tk.set_postfix({"Loss": "%6f" % float(total_loss / t)})

        predictions: np.ndarray = list_of_tensors_to_numpy_flat(predictions, apply_func=torch.sigmoid)
        labels: np.ndarray = list_of_tensors_to_numpy_flat(labels)
        output_nodes_ids: np.ndarray = list_of_tensors_to_numpy_flat(output_nodes_ids)

        id2logits_df = pd.DataFrame(
            data={USERID_DATA_NAME: output_nodes_ids, "score": predictions}, columns=[USERID_DATA_NAME, "score"]
        )

        return id2logits_df

    def train_and_test(self):
        if self.mode == "training":
            best_valid_loss = np.inf
            best_train_loss = np.inf

            for i in range(1, self.epoch + 1):
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
        else:
            f"The mode is {self.mode}, going straight to testing"
            
            
        torch.save(self.model.state_dict(), "checkpoints/last-weights.pt")
        print("Saved Last Weights")
        copy_out_to_snapshot("checkpoints")

        if self.test_dataloader is not None:
            print("Performing test on test dataloader")

            id2logits = self.test()

            return id2logits


        return {}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph Neural Network for fraud prediction")
    
    parser.add_argument("--datadir", type=Path, help="Directory with data", default="./data")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["GBDT", "GNN"],
        default="GNN",
        help="The direction of a program workflow",
    )

    parser.add_argument("--mode", choices=["training", "inference"], default="training")

    parser.add_argument(
        "--data_type",
        type=str,
        choices=["dglgraph", "json"],
        help="Indicator whether or not the data is already preprocessed and stored as a graph",
        default="json",
    )

    parser.add_argument(
        "--remove_self_loops",
        action="store_true",
        help="Whether or note to remove self loops from a graph",
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="Number of training epochs",
        default=1,
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size",
        default=2000000,
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device index for GPU"
    )
    
    parser.add_argument(
        "--out_table_path",
        type=str,
        default="//home/yr/fvelikon/tmp",
        help="Root directory for writing YT table with results"
    )
    
    
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    return parser


TRAINING_PARAMETERS = {
    "batch_size": 2000000,
    "num_epochs": 150,
    "max_num_neighbors": -1,  # -1 for all neighbors to be sampled
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "val_every_steps": 25,
    "early_stopping_steps": 1000,
    "num_workers": 12,
}


MODEL_PARAMS = {
    "num_hidden_features": 128,
    "normalisation_name": "batch",
    "convolution_name": "sage",
    "convolution_params": {
        "aggregator_type": "mean",
    },
    "activation_name": "gelu",
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
    
    if args.device is not None:
        DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    TRAINING_PARAMETERS["batch_size"] = args.batch
    #############
    # IMPORTANT #
    #############
    copy_snapshot_to_out("checkpoints")
    datadir: Path = args.datadir
    model_type: str = args.model_type
    data_dtype: str = args.data_type
    mode: str = args.mode
    table_output_root_path: str = args.out_table_path

    print("Device is ", DEVICE)

    # NOTE: this is not the final implementation of training and evaluation!
    if mode == "training":
        weights_file = None
        train_metadata_file = None
        print(f"The mode is {mode}, launching initial training...")
    else:
        weights_file = "checkpoints/last-weights.pt"
        train_metadata_file = "checkpoints/train_metadata"

        print(f"The mode is {mode}, picking preemtped weights...")

    if data_dtype == "json":
        from utils import prepare_json_input

        graphs, scaler, train_metadata = prepare_json_input(data_dir=datadir, train_metadata_file=train_metadata_file)
        joblib.dump(scaler, "checkpoints/scaler.bin")
        
        # breakpoint()
        with open("checkpoints/train_metadata", "wb") as write_handler:
            joblib.dump(train_metadata, write_handler)

    else:
        graphs_filename = str(datadir / "graphs_train_val_test.bin")  # this is predefined name, used for testing
        graphs, _ = load_graphs(graphs_filename)

        graphs[0].ndata[MASK_DATA_NAME] = graphs[0].ndata[MASK_DATA_NAME].bool()
        graphs[1].ndata[MASK_DATA_NAME] = graphs[1].ndata[MASK_DATA_NAME].bool()
        graphs[2].ndata[MASK_DATA_NAME] = graphs[2].ndata[MASK_DATA_NAME].bool()
        
        

    if args.remove_self_loops:
        [graph_train, graph_valid, graph_test] = [remove_self_loop(g) for g in graphs]
    else:
        graph_train, graph_valid, graph_test = graphs

    num_input_features = graph_train.ndata[FEATURES_DATA_NAME].shape[1]
    
    if USERID_DATA_NAME not in graphs[0].ndata:
        for i in range(3):
            graphs[i].ndata[USERID_DATA_NAME] = torch.arange(len(graphs[i].ndata[FEATURES_DATA_NAME]))

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
        
        index2logits_df: pd.DataFrame = pd.DataFrame(
            data={USERID_DATA_NAME: graph_test.ndata[USERID_DATA_NAME].numpy().reshape(-1), "score": logits},
            columns=[USERID_DATA_NAME, "score"]
        )
        
        

        joblib.dump(model, "checkpoints/xgboost_model.bin")

    else:
        
        TRAINING_PARAMETERS["num_epochs"] = args.n_epochs
        
        print("The mode is GNN")
        from models.gnn_initial_and_plre import create_graph_model

        MODEL_PARAMS.update(dict(num_input_features=num_input_features))
        model = create_graph_model(model_name=model_type, model_params=MODEL_PARAMS).to(DEVICE)

        loss_func = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAINING_PARAMETERS["learning_rate"],
            weight_decay=TRAINING_PARAMETERS["weight_decay"],
        )

        sampler = dgl.dataloading.NeighborSampler(
            fanouts=[TRAINING_PARAMETERS["max_num_neighbors"]] * MODEL_PARAMS["num_encoder_layers"]
        )

        batch_size = TRAINING_PARAMETERS["batch_size"]
        num_workers = TRAINING_PARAMETERS["num_workers"]

        train_loader = init_dataloader(graph_train, sampler, DEVICE, batch_size=batch_size, num_workers=num_workers)
        val_loader = init_dataloader(
            graph_valid, sampler, DEVICE, shuffle=False, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = init_dataloader(
            graph_test, sampler, DEVICE, shuffle=False, batch_size=batch_size, num_workers=num_workers
        )

        trainer = TrainEval(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            criterion=loss_func,
            device=DEVICE,
            batch_size=batch_size,
            mode=mode,
            num_epochs=TRAINING_PARAMETERS["num_epochs"],
            val_every_steps=TRAINING_PARAMETERS["val_every_steps"],
            state_dict_file=weights_file,
        )
        print("Initialized trainer")
        index2logits_df: pd.DataFrame = trainer.train_and_test()

    
    # properly format the output
    
    index2logits_df[USERID_DATA_NAME] = index2logits_df[USERID_DATA_NAME].apply(lambda x: f"/user/{x}")
    print(index2logits_df)
    
        
    index2logits_df.to_csv("index2logits_df.csv")
    index2logits_list_of_dicts = index2logits_df.to_dict('records')
    
    
    
    mr_table_output: dict[str, str] = write_output_to_YT(output=index2logits_list_of_dicts, 
                                                         table_path_root=table_output_root_path)
    
    with open(OUTPUT_FILE_NAME, "w") as out_handler:
        for line in map(ujson.dumps, index2logits_list_of_dicts):
            print(line, file=out_handler)
            
    with open("MR_TABLE", "w") as out_handler:
        ujson.dump(mr_table_output, out_handler)
        
    
    
    copy_out_to_snapshot("checkpoints", dump=True)

if __name__ == "__main__":
    main()
