from deepcardio.losses import LpLoss
import numpy as np
from scipy.optimize import differential_evolution
import torch
import time

l2loss = LpLoss(d=2, p=2, reductions='mean')

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
    
    def _inverse_problem(self, model_inference):
        y_true = self.sample['y'].to(model_inference.device)
        epi_vals = y_true[self.epi_pts_indices]

        min_pos = int(epi_vals.argmin().item())
        t0 = min_pos / float(len(self.epi_pts_indices) - 1)
        x0 = np.array([0.2, t0], dtype=float)
        self.act_max_base = y_true.max().item()
        def _inverse_problem_diso_wrapper(x):
            d_iso = x[0]
            idx = min(
                int(round(x[1] * len(self.epi_pts_indices))),
                len(self.epi_pts_indices) - 1)
            ploc1_indice = self.epi_pts_indices[idx]
            ploc_xyz = self.sample['input_geom'][ploc1_indice].unsqueeze(0)
            output = model_inference.predict(
                self.sample, Diso=d_iso, plocs=ploc_xyz, r=0.55)
            loss = l2loss(output, y_true)
            return loss.item()
        
        start_time = time.time()
        result = differential_evolution(
            _inverse_problem_diso_wrapper,
            bounds=[(0.1, 2.), (0, 1)],
            # popsize=50,
            seed=45,
            strategy='best1bin',
            # maxiter=100,
            x0=x0,
            )
        end_time = time.time()
        print(f"Time taken for the inverse problem: {end_time - start_time} (s)")
        self.d_iso = result.x[0]
        self.ploc1_indice = self.epi_pts_indices[
            min(int(round(result.x[1] * len(self.epi_pts_indices))),
                len(self.epi_pts_indices) - 1)]
        print(f"result 1st problem: {result.fun, result.x}")
        self.ploc1_xyz = self.sample['input_geom'][self.ploc1_indice].unsqueeze(0)
        return None

    def _optimization_problem(self, model_inference):
        def _optimization_problem_ploc2_wrapper(x):
            idx = min(
                int(round(x[0] * len(self.epi_pts_indices))),
                len(self.epi_pts_indices) - 1)
            ploc2_indice = self.epi_pts_indices[idx]
            ploc2_xyz = self.sample['input_geom'][ploc2_indice].unsqueeze(0)
            plocs_xyz = torch.concat((
                self.ploc1_xyz,
                ploc2_xyz), dim=0)

            output = model_inference.predict(
                self.sample, Diso=self.d_iso, plocs=plocs_xyz, r=0.55)

            return output.max().item()
        
        start_time = time.time()
        result = differential_evolution(
            _optimization_problem_ploc2_wrapper,
            bounds=[(0, 1)],
            # popsize=50,
            seed=45,
            strategy='best1bin',
            # maxiter=100
            )
        end_time = time.time()
        print(f"Time taken for the optimization problem: {end_time - start_time} (s)")
        self.ploc2_indice = self.epi_pts_indices[
            min(int(round(result.x[0] * len(self.epi_pts_indices))),
                len(self.epi_pts_indices) - 1)]
        self.ploc2_xyz = self.sample['input_geom'][self.ploc2_indice].unsqueeze(0)
        print(f"result 2nd problem: {result.fun, result.x}")
        self.act_max_crt = result.fun
        return None
    
    def run_invproblem(self, file_token, model_inference, mesh_directory, xdmf_directory):
        self._inverse_problem(model_inference)
        print(f"intrinsic ploc: {self.ploc1_xyz}")
        output = model_inference.predict(
            self.sample, Diso=self.d_iso, plocs=self.ploc1_xyz, r=0.55, crt_invproblem=True)
        
        model_inference.case_ID = file_token
        model_inference.write_xdmf(
            mesh_directory=mesh_directory,
            xdmf_directory=xdmf_directory
        )
        return None
    
    def run_optim(self, file_token, model_inference, mesh_directory, xdmf_directory):
        self._optimization_problem(model_inference)
        plocs_xyz = torch.concat((
                self.ploc1_xyz,
                self.ploc2_xyz), dim=0)

        print(f"plocs_xyz: {plocs_xyz}")
        output = model_inference.predict(
            self.sample, Diso=self.d_iso, plocs=plocs_xyz, r=0.55)
        
        model_inference.case_ID = file_token
        model_inference.write_xdmf(
            mesh_directory=mesh_directory,
            xdmf_directory=xdmf_directory
        )

        print(self.d_iso, self.ploc1_xyz, self.ploc2_xyz)
        return None


if __name__ == "__main__":
    from pathlib import Path
    from deepcardio.electrophysio import model, single_case_handling
    from deepcardio.electrophysio import ModelInference
    import os
    from dotenv import load_dotenv
    from InputProcessing import handle_input_file
    from deepcardio.electrophysio import data_processing
    import sys
    sys.modules['data_processing'] = data_processing
    load_dotenv()
    import subprocess
    import time

    FENICS_CONTAINER = os.getenv(
        'FENICS_CONTAINER_PATH', 
        '/mnt/home/naghavis/Documents/Research/FEniCSx/fenics_legacy3.sif')
    MODEL_CHECKPOINT_PATH = os.getenv(
        'MODEL_CHECKPOINT_PATH',
        '/mnt/home/naghavis/Documents/Research/DeepCardioSim/cardiac_models/electrophysio/GINO/ckpt/ckpt_16/best_model_snapshot_dict.pt')
    DATAPROCESSOR_PATH = os.getenv(
        'DATAPROCESSOR_PATH',
        '/mnt/home/naghavis/Documents/Research/DeepCardioSim/cardiac_models/electrophysio/GINO/data_processor.pt')

    model_inference = ModelInference(
        model=model,
        model_checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataprocessor_path=DATAPROCESSOR_PATH,
        single_case_handling=single_case_handling)

    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    CNVRS_DIR = Path("conversions")
    CNVRS_DIR.mkdir(parents=True, exist_ok=True)

    PRED_DIR = Path("predicted")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    token = 'sample_crt2'
    input_path = UPLOAD_DIR / f"{token}.vtk"
    output_path = CNVRS_DIR / f"{token}.vtp"

    handle_input_file(str(input_path), str(output_path))

    def run_generate_EF(token: str) -> bool:

        cmd = [
            "singularity", "exec",
            FENICS_CONTAINER,
            "python3",
            "generate_EF.py",
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

    file_path = str('./uploads/sample_crt2.vtk')
    sample = model_inference.file_to_inp_data(file=file_path)
    crt_workflow = CRTWorkflow(sample)
    start_time = time.time()
    crt_workflow.run_invproblem(file_token=token,
                model_inference=model_inference,
                mesh_directory = str(UPLOAD_DIR) + '/',
                xdmf_directory = str(PRED_DIR) + '/')

    crt_workflow.run_optim(file_token=token,
                model_inference=model_inference,
                mesh_directory = str(UPLOAD_DIR) + '/',
                xdmf_directory = str(PRED_DIR) + '/')
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

