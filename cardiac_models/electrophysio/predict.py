import torch
import meshio
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np

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
    
    def set_pacingsite(self, sample, plocs):
        sample = sample.clone()
        locs = torch.as_tensor(
            plocs,
            dtype=sample['input_geom'].dtype,
            device=sample['input_geom'].device)
        
        num_plocs = locs.shape[0]
        
        dists = torch.cdist(sample['input_geom'], locs)
        closest_indices = []
        for i in range(num_plocs):
            closest_idx = torch.argmin(dists[:, i])
            closest_indices.append(closest_idx)
        
        sample['a'][..., 0] = 0.
        for idx in closest_indices:
            sample['a'][idx, ..., 0] = 1.
        return sample
        
    def predict(self, inp, Diso=None, plocs=None, crt_invproblem=False):
        if isinstance(inp, str) and self.single_case_handling is not None:
            sample = self.file_to_inp_data(file=inp)
        else:
            sample = inp
        
        if Diso is not None:
            sample = self.set_Diso(sample, Diso)
            if 'y' in sample and not crt_invproblem:
                del sample['y']
        if plocs is not None:
            sample = self.set_pacingsite(sample, plocs)
            if 'y' in sample and not crt_invproblem:
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
            local_error = norm_diff
            self.local_error = local_error

        self.sample = sample
        self.case_ID = sample.get('label', None)
        self.output = output
        return output

    def write_xdmf(
            self,
            inp_data=None,
            inp_meshdir=None,
            xdmf_dir=None,
            sample=None,
            case_ID=None,
            output=None,
            local_error=None):
        if self.output is None and inp_data is not None:
            self.predict(inp_data)
        if inp_meshdir is None:
            inp_meshdir = '/mnt/home/naghavis/Documents/Research/DeepCardioSim/deepcardio/LVmean/LV_mean.vtk'
            npyfile = '/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data/npy/case'+ self.case_ID + '_nplocs1.npy'
            if not Path(npyfile).exists():
                npyfile = '/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data/npy/case'+ self.case_ID + '_nplocs2.npy'
                if not Path(npyfile).exists():
                    raise ValueError('No default case-specific npy file found for setting up the mesh please pass the input mesh directory and try again.')
            npydata = np.load(npyfile)
            mesh_points = npydata[:, :3]
            mesh = meshio.read(inp_meshdir)
            mesh.points = mesh_points
            mesh.point_data.clear()
        else:
            mesh = meshio.read(inp_meshdir)
            mesh_points = mesh.points
            cells = mesh.cells_dict.get("tetra")
            if cells is None:
                raise ValueError("No tetrahedral cells found in the mesh")
        
        if xdmf_dir is None:
            xdmf_file = './results/xdmf/case' + self.case_ID + '.xdmf'
            self.xdmf_file = xdmf_file
            Path(xdmf_file).parent.mkdir(parents=True, exist_ok=True)
        else:
            xdmf_file = xdmf_dir
            self.xdmf_file = xdmf_dir

        if output is None:
            output = self.output.detach().cpu()
        
        num_timesteps = output.shape[1]
        if sample is None:
            sample = self.sample
        if case_ID is None:
            case_ID = self.case_ID
        if local_error is None and self.local_error is not None:
            local_error = self.local_error
        data_points = sample['input_geom'].cpu()
        tree = cKDTree(data_points)
        _, indices = tree.query(mesh_points)
        if 'y' in sample and local_error is not None:
            y = sample['y'].cpu()
            local_error = local_error.detach().cpu()
            reordered_y = y[indices]
            reordered_error = local_error[indices]
        else:
            reordered_y = None
            reordered_error = None
        
        a = sample['a'].clone()
        if a.ndim == 2:
            a = a.unsqueeze(1)
        
        with meshio.xdmf.TimeSeriesWriter(xdmf_file) as writer:
            writer.write_points_cells(mesh.points, mesh.cells)
            for i in range(num_timesteps):
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

                if "computed_labels" in sample.keys():
                    point_data["computed_labels"] = sample["computed_labels"][indices].cpu().numpy()
                writer.write_data(i, point_data=point_data)


