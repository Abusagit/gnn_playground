from typing import Any
import argparse
from xgboost import XGBClassifier


OUTPUT_MASK_NAME = "output_mask"
FEATURES_DATA_NAME = "features"
MASK_DATA_NAME = "mask"
LABELS_DATA_NAME = "labels"
DEVICE = "cpu"

#### Nirvana ####
from nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot



model_name_to_class = {
    "GBDT": XGBClassifier,
}


def create_graph_model(model_name, model_params):
    model_class = model_name_to_class[model_name]
    model = model_class(**model_params)

    return model



CONFIG = {
    "model": "GBDT",
    "name": "initial_GBDT_toy_test",
    
    "training_params": {
        "batch_size": 1000000,
        "num_epochs": 100,
        "max_num_neighbors": -1, # -1 for all neighbors to be sampled
        "learning_rate": 0.0003,
        "weight_decay": 0.00001,
        "val_every_steps": 5,
        "early_stopping_steps": 40,
        "num_workers": 12,
    },
    
    "model_params": {
        "num_input_features": 224,
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

        "n_frequencies": 48,
        "frequency_scale": 0.01,
        "d_embedding": 16,
    }
    
}


def main():
    
    parser = argparse.ArgumentParser(description="Graph Neural Network for antifraud prediction")
    
    parser.add_argument("--graphs", type=str, help="File with graphs", default="./graphs.bin")
    parser.add_argument("--features_mask", type=str, help="File masking some of features", default="./feature_mask.bin")
    parser.add_argument("--predictions_test_file", type=str, help="File with predictions testing information", default="prediction_test.npz")
    
    args = parser.parse_args()
    
    graphs, _ = load_graphs(args.graphs)
    [graph_train, graph_valid, graph_test] = [dgl.remove_self_loop(g) for g in graphs]
    

    NAME = CONFIG["name"]
    MODEL_NAME = CONFIG["model"]
    TRAINING_PARAMS = CONFIG["training_params"]
    MODEL_PARAMS = CONFIG["model_params"]
    
    if args.features_mask is not None:
        
        features_mask = np.load(args.features_mask)
        graph_train.ndata[FEATURES_DATA_NAME] = graph_train.ndata[FEATURES_DATA_NAME][:, features_mask]
        graph_valid.ndata[FEATURES_DATA_NAME] = graph_valid.ndata[FEATURES_DATA_NAME][:, features_mask]
        graph_test.ndata[FEATURES_DATA_NAME] = graph_test.ndata[FEATURES_DATA_NAME][:, features_mask]
    
    
    num_input_features = graph_train.ndata[FEATURES_DATA_NAME].shape[1]
    
    model_parameters = {

        "n_frequencies": 48, 
        "frequency_scale": 0.2, 
        "d_embedding": 16,
    }
    
    copy_snapshot_to_out("checkpoints")
    
    TRAINING_PARAMS.update(dict(learning_rate=1e-5))
    
    
    MODEL_PARAMS.update(model_parameters)
    MODEL_PARAMS["num_input_features"] = num_input_features
    
    
    model = create_graph_model(MODEL_NAME, MODEL_PARAMS).to(DEVICE)
    
    
    model, report = train_gnn(model=model, graphs=[graph_train, graph_valid, graph_test], device=DEVICE,
                            predictions_test_file=PREDICTIONS_TEST,
                            num_encoder_layers=MODEL_PARAMS["num_encoder_layers"],
                            **TRAINING_PARAMS,
                        )
    
if __name__ == '__main__':
    main()