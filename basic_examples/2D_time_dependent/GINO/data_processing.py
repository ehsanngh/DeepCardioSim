from deepcardio.neuralop_core.transforms import DataProcessor
from torch_geometric.data import Data

class CustomDataProcessorGINO(DataProcessor):
    def __init__(
        self,
        a_normalizer=None,
        input_geom_normalizer=None,
        query_normalizer=None,
        out_normalizer=None
    ):
        super().__init__()
        self.a_normalizer = a_normalizer
        self.input_geom_normalizer = input_geom_normalizer
        self.query_normalizer = query_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.a_normalizer is not None:
            self.a_normalizer = self.a_normalizer.to(device)
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
        a = data_dict_batch['a']
        input_geom = data_dict_batch['input_geom']
        latent_queries = data_dict_batch['latent_queries']
        y = data_dict_batch['y']

        if self.a_normalizer is not None:
            a[:, :, -1:] = self.a_normalizer.transform(a[:, :, -1:])
        if self.input_geom_normalizer is not None:
            input_geom = self.input_geom_normalizer.transform(input_geom)
        if self.query_normalizer is not None:
            latent_queries = self.query_normalizer.transform(latent_queries)
        if self.out_normalizer is not None:
            y = self.out_normalizer.transform(y)
        
        
        data_dict_batch_updated = Data(
            a=a, input_geom=input_geom,
            latent_queries=latent_queries, y=y)
        
        return data_dict_batch.update(data_dict_batch_updated)

    def postprocess(self, output, data_dict_batch):
        data_dict_batch = data_dict_batch.to(self.device)
        a = data_dict_batch['a']
        input_geom = data_dict_batch['input_geom']
        latent_queries = data_dict_batch['latent_queries']
        y = data_dict_batch['y']

        if self.out_normalizer is not None and not self.training:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)

        output = output * (1 - a[:, :, 0:1]) + y * a[:, :, 0:1]

        if self.a_normalizer is not None:
            a[:, :, -1:] = self.a_normalizer.inverse_transform(a[:, :, -1:])
        if self.input_geom_normalizer is not None:
            input_geom = self.input_geom_normalizer.inverse_transform(input_geom)
        if self.query_normalizer is not None:
            latent_queries = self.query_normalizer.inverse_transform(latent_queries)
        
        data_dict_batch_updated = Data(
            a=a, input_geom=input_geom,
            latent_queries=latent_queries, y=y)
        
        return output, data_dict_batch.update(data_dict_batch_updated)

    def forward(self, **data_dict_batch):
        data_dict_batch = self.preprocess(data_dict_batch)
        output = self.model(**data_dict_batch)
        output = self.postprocess(output, data_dict_batch)
        return output, data_dict_batch