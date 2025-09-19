from .gino import GINO

def initialize_GINO_model(n_fno_modes):
    model = GINO(
        in_channels=5,  # [ploc_bool, D_iso, ef_vector]
        out_channels=1,
        gno_coord_dim=3,
        gno_coord_embed_dim=16,
        gno_radius=0.1,
        gno_transform_type='linear',
        fno_n_modes=[n_fno_modes, n_fno_modes, n_fno_modes, 1],  # x_1, x_2, x_3, t
        fno_hidden_channels=64,
        fno_use_mlp=True,
        fno_norm='instance_norm',
        fno_ada_in_features=32,
        fno_factorization='tucker',
        fno_rank=0.4,
        fno_domain_padding=0.125,
        fno_mlp_expansion=1.0,
        fno_output_scaling_factor=1,
    )
    return model
