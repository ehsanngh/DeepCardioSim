from deepcardio.electrophysio import ModelInference, initialize_GINO_model, single_case_handling
from deepcardio.electrophysio import data_processing
from pathlib import Path
import sys
sys.modules['data_processing'] = data_processing
BACKEND_PATH = Path(__file__).parent
sys.path.insert(0, str(BACKEND_PATH))
from dotenv import load_dotenv
import torch
import os
from state_management import update_token_field, load_token_state
from src.CRTPlanning import CRTWorkflow
load_dotenv()
import logging
import time
from state_management import rdb

MODEL_CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT_PATH')
DATAPROCESSOR_PATH = os.getenv('DATAPROCESSOR_PATH')
PREEMPT_GRACE_S = int(os.getenv("GPU_PREEMPT_GRACE_S"))

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = Path("predicted")
PRED_DIR.mkdir(parents=True, exist_ok=True)

CRT_RUNTIME_DIR = Path("crt_runtime")
CRT_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

model_inference = ModelInference(
    model=initialize_GINO_model(int(os.getenv('NUM_GINO_FNO_MODES', 16))),
    model_checkpoint_path=MODEL_CHECKPOINT_PATH,
    dataprocessor_path=DATAPROCESSOR_PATH,
    single_case_handling=single_case_handling,
    device='cuda:0')


job_gpu_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))

def run_crt_invproblem(token: str) -> bool:
    """Run CRT inverse problem in background"""
    rdb().set(f"gpu_preempt:{job_gpu_id}", "1", ex=60*15)
    time.sleep(PREEMPT_GRACE_S + 1)
    token_data = load_token_state(token)
    try:
        if not token_data:
            logging.error(f"Token {token} not found to start CRT inverse problem")
            return False
        file_path = Path(token_data['input_path'])
        update_token_field(token, 'crt_invproblem_status', 'running')
        
        # Initialize CRT workflow
        sample = model_inference.file_to_inp_data(file=str(file_path))
        crt_workflow = CRTWorkflow(sample)
        logging.info(f"CRT workflow initialized for token {token}")
        update_token_field(token, 'crt_invproblem_status', 'inverse_problem_running')
        
        # Run inverse problem
        crt_workflow.run_invproblem(
            file_token=token,
            model_inference=model_inference,
            inp_meshdir = str(UPLOAD_DIR) + '/',
            xdmf_dir = str(PRED_DIR) + '/')
        
        crt_workflow.save_state(token, CRT_RUNTIME_DIR)
        
        # Store the workflow and update status
        update_token_field(token, 'crt_invproblem_status', 'completed')
        logging.info(f"CRT inverse problem completed for token {token}")
        return True
        
    except Exception as e:
        logging.error(f"Error in CRT inverse problem for token {token}: {e}")
        if load_token_state(token):
            update_token_field(token, 'crt_invproblem_status', 'failed')
        return False
    finally:
        rdb().delete(f"gpu_preempt:{job_gpu_id}")
        rdb().delete(f"gpu_owner:{job_gpu_id}") 


def run_crt_optimization(token: str) -> bool:
    """Run CRT optimization in background"""
    rdb().set(f"gpu_preempt:{job_gpu_id}", "1", ex=60*15)
    time.sleep(PREEMPT_GRACE_S + 1) 
    token_data = load_token_state(token)
    try:
        if not token_data:
            logging.error(f"Token {token} not found in file_mappings")
            return False
        
        file_path = str(Path(token_data['input_path']))
        update_token_field(token, 'crt_optim_status', 'running')
        crt_workflow = CRTWorkflow.load_from_state(
            token, CRT_RUNTIME_DIR, file_path=file_path, model_inference=model_inference)
        update_token_field(token, 'crt_optim_status', 'optimization_running')
        
        # Run optimization
        crt_workflow.run_optim(
            file_token=token,
            model_inference=model_inference,
            inp_meshdir = str(UPLOAD_DIR) + '/',
            xdmf_dir = str(PRED_DIR) + '/')
        
        # Update status to completed
        crt_workflow.save_state(token, CRT_RUNTIME_DIR)
        update_token_field(token, 'crt_optim_status', 'completed')
        logging.info(f"CRT optimization completed for token {token}")
        return True
        
    except Exception as e:
        logging.error(f"Error in CRT optimization for token {token}: {e}")
        if load_token_state(token):
            update_token_field(token, 'crt_optim_status', 'failed')
        return False

    finally:
        rdb().delete(f"gpu_preempt:{job_gpu_id}")
        rdb().delete(f"gpu_owner:{job_gpu_id}")
        torch.cuda.empty_cache()