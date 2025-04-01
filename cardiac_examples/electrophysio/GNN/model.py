from .gnn import GraphUNet
model = GraphUNet(
    in_channels=8,    
    mlp_hidden_channels=256,
    out_channels=1,
    size_hidden_layers = 64,
    dim_enc = 64,
    mlp_depth = 4,
    bn_bool = True,
)