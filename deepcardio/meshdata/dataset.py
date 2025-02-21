from torch_geometric.data import Data, Dataset
import torch

class MeshDataset(Dataset):
    def __init__(self, data_list):
        super(MeshDataset, self).__init__()
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        self.num_nodes = self.a.size(0)
        if key == 'edge_index':
            return torch.tensor([
                [torch.tensor(self.latent_queries.size()[1:-1]).prod()],
                [self.a.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)