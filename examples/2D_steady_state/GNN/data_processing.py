from deepcardio.neuralop_core.transforms import DataProcessor
from torch_geometric.data import Data

class CustomDataProcessorGraph(DataProcessor):
    def __init__(
        self,
        in_normalizer=None,
        edge_attr_normalizer=None,
        pos_normalizer=None,
        out_normalizer=None
    ):
        super().__init__()
        self.in_normalizer = in_normalizer
        self.edge_attr_normalizer = edge_attr_normalizer 
        self.out_normalizer = out_normalizer
        self.pos_normalizer = pos_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.edge_attr_normalizer is not None:
            self.edge_attr_normalizer = self.edge_attr_normalizer.to(device)
        if self.pos_normalizer is not None:
            self.pos_normalizer = self.pos_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, graph_data_dict_batch):
        graph_data_dict_batch = graph_data_dict_batch.to(self.device)
        x = graph_data_dict_batch['x']
        edge_attr = graph_data_dict_batch['edge_attr']
        pos = graph_data_dict_batch['pos']
        y = graph_data_dict_batch['y']

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None:
            y = self.out_normalizer.transform(y)
        if self.edge_attr_normalizer is not None:
            edge_attr = self.edge_attr_normalizer.transform(edge_attr)
        if self.pos_normalizer is not None:
            pos = self.pos_normalizer.transform(pos)

        updated = Data(
            x=x,
            edge_attr=edge_attr,
            pos=pos,
            y=y)
        return graph_data_dict_batch.update(updated)

    def postprocess(self, output, graph_data_dict_batch):
        graph_data_dict_batch = graph_data_dict_batch.to(self.device)
        x = graph_data_dict_batch['x']
        edge_attr = graph_data_dict_batch['edge_attr']
        pos = graph_data_dict_batch['pos']
        y = graph_data_dict_batch['y']
        
        if self.in_normalizer is not None:
            x = self.in_normalizer.inverse_transform(x)
        if self.out_normalizer is not None and not self.training:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)

        output = output * (1 - x[:, -2:-1]) + y * x[:, -2:-1]

        if self.edge_attr_normalizer is not None:
            edge_attr = self.edge_attr_normalizer.inverse_transform(edge_attr)

        if self.pos_normalizer is not None:
            pos = self.pos_normalizer.inverse_transform(pos)
        
        updated = Data(
            x=x,
            edge_attr=edge_attr,
            pos=pos,
            y=y)

        return output, graph_data_dict_batch.update(updated)

    def forward(self, **graph_data_dict_batch):
        graph_data_dict_batch = self.preprocess(graph_data_dict_batch)
        output = self.model(graph_data_dict_batch)
        output = self.postprocess(output)
        return output, graph_data_dict_batch