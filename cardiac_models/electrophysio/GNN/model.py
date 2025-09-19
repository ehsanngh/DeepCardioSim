from .gnn import GraphUNet

def initialize_GNN_model(size_hidden_layers=64):
    model = GraphUNet(
        in_channels=8,    
        mlp_hidden_channels=256,
        out_channels=1,
        size_hidden_layers = size_hidden_layers,
        dim_enc = size_hidden_layers,
        mlp_depth = 4,
        bn_bool = True,
    )
    return model