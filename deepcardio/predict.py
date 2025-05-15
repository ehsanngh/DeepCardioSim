import torch
import meshio
from scipy.spatial import cKDTree

class ModelInference:
    def __init__(
            self,
            model: torch.nn.modules,
            model_checkpoint_path: str,
            dataprocessor_path: str,
            single_case_handling: callable = None,
            device=None):
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        checkpoint = torch.load(
            model_checkpoint_path, map_location='cpu', weights_only=False)
        self.current_epoch = checkpoint["CURRENT_EPOCH"]
        self.best_loss = checkpoint.get("BEST_LOSS", None)
        
        model.load_state_dict(checkpoint['MODEL_STATE'])
        self.model = model.to(self.device)
        self.model.eval()
        self.data_processor = torch.load(dataprocessor_path, weights_only=False).to(self.device)
        self.data_processor.eval()
        self.single_case_handling = single_case_handling
        
        self.num_timesteps = 0 
        self.sample = None
        self.output = None
        self.local_error = None
        self.case_ID = None
        self.xdmf_file = None

    def file_to_inp_data(self, file):
        if self.single_case_handling is None:
            raise ValueError('"single_case_handling" must be passed to predict from a raw file.')
        else:
            sample = self.single_case_handling(file=file)
        return sample

    def set_Diso(self, sample, Diso):
        sample = sample.clone()
        sample['a'][..., 1] = Diso * torch.ones_like(sample['a'][..., 1])
        return sample
    
    def set_pacingsite(self, sample, plocs, r):
        sample = sample.clone()
        locs = torch.as_tensor(
            plocs,
            dtype=sample['input_geom'].dtype,
            device=sample['input_geom'].device)
        
        r = torch.as_tensor(
            r,
            dtype=sample['input_geom'].dtype,
            device=sample['input_geom'].device)
        
        dists = torch.cdist(sample['input_geom'], locs)
        cond = (dists <= r).any(dim=1)
            
        if sample['a'].ndim == 3:
            cond = cond.unsqueeze(-1)
        sample['a'][..., 0] = torch.where(cond, 1., 0.)
        return sample
        
    def predict(self, inp, Diso=None, plocs=None, r=0.5):
        if isinstance(inp, str) and self.single_case_handling is not None:
            sample = self.file_to_inp_data(file=inp)
        else:
            sample = inp
        
        if Diso is not None:
            sample = self.set_Diso(sample, Diso)
            if 'y' in sample:
                del sample['y']
        if plocs is not None:
            sample = self.set_pacingsite(sample, plocs, r)
            if 'y' in sample:
                del sample['y']

        sample = self.data_processor.preprocess(sample)
        output = self.model(**sample)
        output, sample = self.data_processor.postprocess(output, sample)
        
        if 'y' in sample:  # Only compute y and error if 'y' is in the sample
            flattened_output = torch.flatten(output, start_dim=2)
            flattened_y = torch.flatten(sample['y'], start_dim=2)
            norm_diff = torch.linalg.vector_norm(
                flattened_output - flattened_y,
                ord=2, dim=-1, keepdim=True
            )
            self.local_error = norm_diff

        self.sample = sample
        self.num_timesteps = output.shape[1]
        self.case_ID = sample.get('label', None)
        self.output = output
        return output

    def write_xdmf(
            self,
            inp=None,
            mesh_directory='../data/mesh/case',
            xdmf_directory='./results/xdmf/case'):
        if self.output is None:
            self.predict(inp)
        meshfile = mesh_directory + self.case_ID + '.vtk'
        self.xdmf_file = xdmf_directory + self.case_ID + '.xdmf'
        mesh = meshio.read(meshfile)
        meshio_points = mesh.points

        cells = mesh.cells_dict.get("tetra")
        if cells is None:
            raise ValueError("No tetrahedral cells found in the mesh")
        
        output = self.output.detach().cpu()
        data_points = self.sample['input_geom'].cpu()
        tree = cKDTree(data_points)
        _, indices = tree.query(meshio_points)

        # Only reorder y and error if they have been computed
        if 'y' in self.sample and self.local_error is not None:
            y = self.sample['y'].cpu()
            local_error = self.local_error.detach().cpu()
            reordered_y = y[indices]
            reordered_error = local_error[indices]
        else:
            reordered_y = None
            reordered_error = None
        
        a = self.sample['a'].clone()
        if a.ndim == 2:
            a = a.unsqueeze(1)
        
        with meshio.xdmf.TimeSeriesWriter(self.xdmf_file) as writer:
            writer.write_points_cells(mesh.points, mesh.cells)
            for i in range(self.num_timesteps):
                data1 = reordered_y[:, i, 0].numpy() if reordered_y is not None else None
                data2 = output[indices][:, i, 0].numpy()
                data3 = reordered_error[:, i, 0].numpy() if reordered_error is not None else None
                data4 = a[indices, i, -5].cpu().numpy()
                data5 = a[indices, i, -4].cpu().numpy()
                data6 = a[indices, i, -3:].cpu().numpy()
                point_data = {
                    "y_est": data2,
                    "ploc_bool": data4,
                    "D_iso": data5,
                    "ef": data6}
                
                if data1 is not None:
                    point_data["y_true"] = data1
                if data3 is not None:
                    point_data["error"] = data3

                writer.write_data(i, point_data=point_data)


