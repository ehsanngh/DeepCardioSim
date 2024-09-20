from deepcardio.neuralop_core.transforms import DataProcessor
from torch_geometric.data import Data

class CustomDataProcessorGINO(DataProcessor):
    def __init__(
        self,
        x_normalizer=None,
        input_geom_normalizer=None,
        query_normalizer=None,
        out_normalizer=None
    ):
        super().__init__()
        self.x_normalizer = x_normalizer
        self.input_geom_normalizer = input_geom_normalizer
        self.query_normalizer = query_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.x_normalizer is not None:
            self.x_normalizer = self.x_normalizer.to(device)
        if self.input_geom_normalizer is not None:
            self.input_geom_normalizer = self.input_geom_normalizer.to(device)
        if self.query_normalizer is not None:
            self.query_normalizer = self.query_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict_batch):
        data_dict_batch = data_dict_batch.to(self.device)
        x = data_dict_batch['x']
        input_geom = data_dict_batch['input_geom']
        latent_queries = data_dict_batch['latent_queries']
        y = data_dict_batch['y']

        if self.x_normalizer is not None:
            x[:, -1:] = self.x_normalizer.transform(x[:, -1:])
        if self.input_geom_normalizer is not None:
            input_geom = self.input_geom_normalizer.transform(input_geom)
        if self.query_normalizer is not None:
            latent_queries = self.query_normalizer.transform(latent_queries)
        if self.out_normalizer is not None:
            y = self.out_normalizer.transform(y)
        
        
        data_dict_batch_updated = Data(
            x=x, input_geom=input_geom,
            latent_queries=latent_queries, y=y)
        
        return data_dict_batch.update(data_dict_batch_updated)

    def postprocess(self, output, data_dict_batch):
        data_dict_batch = data_dict_batch.to(self.device)
        x = data_dict_batch['x']
        input_geom = data_dict_batch['input_geom']
        latent_queries = data_dict_batch['latent_queries']
        y = data_dict_batch['y']

        if self.out_normalizer is not None and not self.training:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)

        output = output * (1 - x[:, 0:1]) + y * x[:, 0:1]

        if self.x_normalizer is not None:
            x[:, -1:] = self.x_normalizer.inverse_transform(x[:, -1:])
        if self.input_geom_normalizer is not None:
            input_geom = self.input_geom_normalizer.inverse_transform(input_geom)
        if self.query_normalizer is not None:
            latent_queries = self.query_normalizer.inverse_transform(latent_queries)
        
        data_dict_batch_updated = Data(
            x=x, input_geom=input_geom,
            latent_queries=latent_queries, y=y)
        
        return output, data_dict_batch.update(data_dict_batch_updated)

    def forward(self, **data_dict_batch):
        data_dict_batch = self.preprocess(data_dict_batch)
        output = self.model(**data_dict_batch)
        output = self.postprocess(output, data_dict_batch)
        return output, data_dict_batch