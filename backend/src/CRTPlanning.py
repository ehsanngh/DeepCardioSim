from deepcardio.losses import LpLoss
import numpy as np
from scipy.optimize import differential_evolution
import torch
import time
import joblib
from pathlib import Path
from torch_geometric.data import Batch
from torch_cluster import radius
from torch_scatter import scatter_max

l2loss = LpLoss(d=2, p=2, reductions='mean')
l2loss.reduce_dims = None

class CRTWorkflow:
    def __init__(self, sample):
        self.sample = sample
        self.d_iso = None
        self.ploc1_indice = None
        self.ploc2_indice = None
        self.epi_pts_indices = np.where(self.sample["computed_labels"] == 1)[0]
        self.ploc1_xyz = None
        self.ploc2_xyz = None
        self.act_max_base = 200
        self.act_max_crt = 199

    def _prepare_vectorized_optimization(self, model_inference, batch_size, invprob_flag):
        self.sample['a'][:, ..., 0] = 0.
        row, col = radius(
                self.sample['input_geom'][self.epi_pts_indices],
                self.sample['input_geom'],
                r=.75,
                max_num_neighbors=512)
        
        neighbors_dict = {}
        for i, epi_idx in enumerate(self.epi_pts_indices):
            neighbors_dict[int(epi_idx)] = row[col == i]
        
        max_num_neighbors = max(len(neighbors) for neighbors in neighbors_dict.values())

        neighbors_padded = torch.full(
            (len(self.epi_pts_indices), max_num_neighbors), -1, dtype=torch.long, device=model_inference.device)
        neighbors_mask = torch.zeros(
            (len(self.epi_pts_indices), max_num_neighbors), dtype=torch.bool, device=model_inference.device)
        for i, epi_idx in enumerate(self.epi_pts_indices):
            neighbors = neighbors_dict[int(epi_idx)]
            k = neighbors.numel()
            if k > 0:
                neighbors_padded[i, :k] = neighbors
                neighbors_mask[i, :k] = True
        
        if not invprob_flag:  # Optimization problem
            ploc1_mask = torch.where(self.epi_pts_indices == self.ploc1_indice)[0]
            ploc1_indices = neighbors_padded[ploc1_mask][neighbors_mask[ploc1_mask]]
            self.sample['a'][ploc1_indices, ..., 0] = 1.
            self.sample['a'][:, ..., 1] = self.d_iso

        sample_list = [self.sample.clone() for _ in range(batch_size)]
        batched_sample = Batch.from_data_list(sample_list)
        return batched_sample, neighbors_padded, neighbors_mask

    def _inverse_problem(self, model_inference):
        self.sample = self.sample.to(model_inference.device)
        y_true = self.sample['y']
        epi_vals = y_true[self.epi_pts_indices]
        min_pos = int(epi_vals.argmin().item())
        t0 = min_pos / float(len(self.epi_pts_indices) - 1)
        x0 = np.array([0.2, t0], dtype=float)
        self.act_max_base = y_true.max().item()
        pop_size = 15
        batch_size = pop_size * 2
        batched_sample, neighbors_padded, neighbors_mask = self._prepare_vectorized_optimization(model_inference, batch_size, invprob_flag=True)
        
        def _inverse_problem_wrapper_func(x_population): 
            batch_size = x_population.shape[1]
            if batch_size == 1:
                wrapper_sample = self.sample
                offsets = torch.tensor([0], device=model_inference.device)
            else:
                wrapper_sample = batched_sample
                offsets = batched_sample.ptr[:-1].unsqueeze(1)
            wrapper_sample['a'][:, 0] = 0.
            normalized_positions = x_population[1, :] * len(self.epi_pts_indices)
            ploc_indices_in_epi_pts = np.clip(
                np.round(normalized_positions).astype(int),
                0,
                len(self.epi_pts_indices) - 1
                )
            neighbors_global = neighbors_padded[ploc_indices_in_epi_pts] + offsets
            neighbors_global = neighbors_global[neighbors_mask[ploc_indices_in_epi_pts]]
            wrapper_sample['a'][neighbors_global, 0] = 1.
            wrapper_sample['a'][:, 1] = torch.repeat_interleave(
                torch.tensor(x_population[0, :],
                device=model_inference.device,
                dtype=self.sample['input_geom'].dtype),
                self.sample['input_geom'].shape[0])
            with torch.no_grad():
                output_batch = model_inference.model(
                    **model_inference.data_processor.preprocess(wrapper_sample))
                output_batch, _ = model_inference.data_processor.postprocess(
                    output_batch, wrapper_sample)
                loss = l2loss(output_batch, **wrapper_sample)
            return loss.detach().cpu().numpy()
        
        start_time = time.time()
        result = differential_evolution(
            _inverse_problem_wrapper_func,
            bounds=[(0.1, 2.), (0, 1)],
            popsize=pop_size,
            seed=45,
            strategy='best1bin',
            updating='deferred',
            # maxiter=100,
            x0=x0,
            vectorized=True,
            polish=True
        )
        end_time = time.time()
        print(f"Time taken for the inverse problem: {end_time - start_time} (s)")
        self.d_iso = result.x[0]
        self.ploc1_indice = self.epi_pts_indices[
            min(int(round(result.x[1] * len(self.epi_pts_indices))),
            len(self.epi_pts_indices) - 1)]
        print(f"result 1st problem: {result.fun, result.x}")
        self.ploc1_xyz = self.sample['input_geom'][self.ploc1_indice].unsqueeze(0).detach().cpu()
        return None

    def _optimization_problem(self, model_inference):
        pop_size = 15
        batch_size = pop_size * 1
        self.sample = self.sample.to(model_inference.device)
        batched_sample, neighbors_padded, neighbors_mask = self._prepare_vectorized_optimization(model_inference, batch_size, invprob_flag=False)
        def _optimization_problem_ploc2_wrapper(x_population):
            batch_size = x_population.shape[1]
            if batch_size == 1:
                wrapper_sample = self.sample.clone()
                offsets = torch.tensor([0], device=model_inference.device)
            else:
                wrapper_sample = batched_sample.clone()
                offsets = batched_sample.ptr[:-1].unsqueeze(1)
            normalized_positions = x_population[0, :] * len(self.epi_pts_indices)
            ploc_indices_in_epi_pts = np.clip(
                np.round(normalized_positions).astype(int),
                0,
                len(self.epi_pts_indices) - 1
                )
            neighbors_global = neighbors_padded[ploc_indices_in_epi_pts] + offsets
            neighbors_global = neighbors_global[neighbors_mask[ploc_indices_in_epi_pts]]
            wrapper_sample['a'][neighbors_global, 0] = 1.
            with torch.no_grad():
                output_batch = model_inference.model(
                    **model_inference.data_processor.preprocess(wrapper_sample))
                output_batch, _ = model_inference.data_processor.postprocess(
                    output_batch, wrapper_sample)
                if batch_size == 1:
                    output = output_batch.max()
                else:
                    output, _ = scatter_max(output_batch.flatten(), wrapper_sample['batch'])
            return output.detach().cpu().numpy()
        
        start_time = time.time()
        result = differential_evolution(
            _optimization_problem_ploc2_wrapper,
            bounds=[(0, 1)],
            popsize=pop_size,
            seed=45,
            strategy='best1bin',
            vectorized=True,
            polish=True,
            updating='deferred',
            # maxiter=100
            )
        end_time = time.time()
        print(f"Time taken for the optimization problem: {end_time - start_time} (s)")
        self.ploc2_indice = self.epi_pts_indices[
            min(int(round(result.x[0] * len(self.epi_pts_indices))),
                len(self.epi_pts_indices) - 1)]
        self.ploc2_xyz = self.sample['input_geom'][self.ploc2_indice].unsqueeze(0).detach().cpu()
        print(f"result 2nd problem: {result.fun, result.x}")
        self.act_max_crt = result.fun
        return None
    
    def run_invproblem(self, file_token, model_inference, inp_meshdir, xdmf_dir):
        self._inverse_problem(model_inference)
        print(f"intrinsic ploc: {self.ploc1_xyz}")
        output = model_inference.predict(
            self.sample, Diso=self.d_iso, plocs=self.ploc1_xyz, crt_invproblem=True).detach().cpu()
        sample = model_inference.sample.cpu()
        model_inference.case_ID = file_token
        model_inference.write_xdmf(
            inp_meshdir=inp_meshdir + '/' + model_inference.case_ID + '.vtk',
            xdmf_dir=xdmf_dir + model_inference.case_ID + '.xdmf',
            sample=sample,
            case_ID=file_token,
            output=output,
            local_error=model_inference.local_error
        )
        return None
    
    def run_optim(self, file_token, model_inference, inp_meshdir, xdmf_dir):
        self._optimization_problem(model_inference)
        plocs_xyz = torch.concat((
                self.ploc1_xyz,
                self.ploc2_xyz), dim=0)

        print(f"plocs_xyz: {plocs_xyz}")
        output = model_inference.predict(
            self.sample, Diso=self.d_iso, plocs=plocs_xyz).detach().cpu()
        sample = model_inference.sample.cpu()
        
        model_inference.case_ID = file_token
        model_inference.write_xdmf(
            inp_meshdir=inp_meshdir + '/' + model_inference.case_ID + '.vtk',
            xdmf_dir=xdmf_dir + model_inference.case_ID + '.xdmf',
            sample=sample,
            case_ID=file_token,
            output=output,
            local_error=None
        )

        print(self.d_iso, self.ploc1_xyz, self.ploc2_xyz)
        return None
    
    def save_state(self, token: str, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "d_iso": None if self.d_iso is None else float(self.d_iso),
            "ploc1_xyz": None if self.ploc1_xyz is None else np.asarray(self.ploc1_xyz.detach().cpu()).reshape(-1).tolist(),
            "ploc2_xyz": None if self.ploc2_xyz is None else np.asarray(self.ploc2_xyz.detach().cpu()).reshape(-1).tolist(),
            "act_max_base": float(self.act_max_base),
            "act_max_crt": float(self.act_max_crt),
            "v": 1,
        }
        joblib.dump(state, state_dir / f"{token}.pkl")

    @classmethod
    def load_from_state(cls, token: str, state_dir: Path, *, file_path: str, model_inference) -> "CRTWorkflow":
        """Recompute sample from file; rebuild a fresh workflow."""
        state = joblib.load(state_dir / f"{token}.pkl")
        sample = model_inference.file_to_inp_data(file=file_path)
        wf = cls(sample)

        wf.d_iso = state.get("d_iso")
        wf.ploc1_xyz = state.get("ploc1_xyz")
        wf.ploc2_xyz = state.get("ploc2_xyz")
        wf.act_max_base = state.get("act_max_base", 200.0)
        wf.act_max_crt = state.get("act_max_crt", 199.0)

        if wf.ploc1_xyz is not None:
            wf.ploc1_xyz = torch.tensor(
                wf.ploc1_xyz,
                dtype=wf.sample["input_geom"].dtype,
                device=wf.sample["input_geom"].device).unsqueeze(0)
            wf.ploc1_indice = (
                torch.abs(wf.sample["input_geom"] - wf.ploc1_xyz).sum(dim=1)).argmin()
        
        if wf.ploc2_xyz is not None:
            wf.ploc2_xyz = torch.tensor(
                wf.ploc2_xyz,
                dtype=wf.sample["input_geom"].dtype,
                device=wf.sample["input_geom"].device).unsqueeze(0)
            wf.ploc2_indice = (
                torch.abs(wf.sample["input_geom"] - wf.ploc2_xyz).sum(dim=1)).argmin()
        return wf


if __name__ == "__main__":
    from pathlib import Path
    from deepcardio.electrophysio import initialize_GINO_model, single_case_handling
    from deepcardio.electrophysio import ModelInference
    import os
    from dotenv import load_dotenv
    from deepcardio.electrophysio import data_processing
    import sys
    sys.modules['data_processing'] = data_processing
    BACKEND_PATH = Path(__file__).parent.parent
    sys.path.insert(0, str(BACKEND_PATH))
    from src.InputProcessing import handle_input_file
    load_dotenv()
    import subprocess
    import time
    model = initialize_GINO_model(16)
    FENICS_CONTAINER = os.getenv('FENICS_CONTAINER_PATH')
    MODEL_CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT_PATH')
    DATAPROCESSOR_PATH = os.getenv('DATAPROCESSOR_PATH')

    model_inference = ModelInference(
        model=model,
        model_checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataprocessor_path=DATAPROCESSOR_PATH,
        single_case_handling=single_case_handling)

    UPLOAD_DIR = Path('/mnt/home/naghavis/Documents/Research/DeepCardioSim/backend/uploads')

    CNVRS_DIR = Path('/mnt/home/naghavis/Documents/Research/DeepCardioSim/backend/conversions')

    PRED_DIR = Path('/mnt/home/naghavis/Documents/Research/DeepCardioSim/backend/predicted')

    CRT_RUNTIME_DIR = Path('/mnt/home/naghavis/Documents/Research/DeepCardioSim/backend/crt_runtime')

    token = 'sample_crt'
    input_path = UPLOAD_DIR / f"{token}.vtk"
    output_path = CNVRS_DIR / f"{token}.vtp"

    handle_input_file(str(input_path), str(output_path))

    def run_generate_EF(token: str) -> bool:

        cmd = [
            "singularity", "exec",
            FENICS_CONTAINER,
            "python3",
            "/mnt/home/naghavis/Documents/Research/DeepCardioSim/backend/src/generate_EF.py",
            "--token", token
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
            
        return True

    run_generate_EF(token)

    file_path = str(UPLOAD_DIR / 'sample_crt.vtk')
    sample = model_inference.file_to_inp_data(file=file_path)
    crt_workflow = CRTWorkflow(sample)
    start_time = time.time()
    crt_workflow.run_invproblem(file_token=token,
                model_inference=model_inference,
                inp_meshdir = str(UPLOAD_DIR) + '/',
                xdmf_dir = str(PRED_DIR) + '/')
    crt_workflow.save_state(token, CRT_RUNTIME_DIR)

    crt_workflow = CRTWorkflow.load_from_state(
        token, CRT_RUNTIME_DIR, file_path=file_path, model_inference=model_inference)

    crt_workflow.run_optim(file_token=token,
                model_inference=model_inference,
                inp_meshdir = str(UPLOAD_DIR) + '/',
                xdmf_dir = str(PRED_DIR) + '/')
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

