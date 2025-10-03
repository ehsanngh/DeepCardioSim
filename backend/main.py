import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from deepcardio.electrophysio import ModelInference
from deepcardio.electrophysio import initialize_GINO_model, single_case_handling
from deepcardio.electrophysio import data_processing
import sys
sys.modules['data_processing'] = data_processing
BACKEND_PATH = Path(__file__).parent
sys.path.insert(0, str(BACKEND_PATH))
from src.format_conversions import convert_xdmf_to_vtp
from src.InputProcessing import handle_input_file
from src.CRTPlanning import CRTWorkflow
import uuid
from fastapi.responses import FileResponse, JSONResponse
from state_management import now_iso, save_token_state, load_token_state, update_token_field
import numpy as np
import json
from gpu_pool import WarmPool
import os
from dotenv import load_dotenv
load_dotenv()


import redis
from rq import Queue
from rq.registry import StartedJobRegistry

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.from_url(REDIS_URL)

EF_QUEUE  = Queue("ef",  connection=redis_conn, default_timeout=60*10)
CRT_QUEUE = Queue("crt", connection=redis_conn, default_timeout=2*60*60)

MODEL_CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT_PATH')
DATAPROCESSOR_PATH = os.getenv('DATAPROCESSOR_PATH')

NUM_WORKERS = int(os.getenv('NUM_WORKERS', 1))
NUM_PROCS = min(int(os.getenv('NUM_PROCS_PER_WORKER', 1)), int(torch.get_num_threads() / NUM_WORKERS))
NUM_GPUS = min(int(os.getenv('NUM_GPUS', 0)), torch.cuda.device_count())
API_VISIBLE_GPUS = [int(x) for x in os.getenv("API_VISIBLE_GPUS", "0").split(",")]

if len(API_VISIBLE_GPUS) != NUM_GPUS:
    NUM_GPUS = len(API_VISIBLE_GPUS)
    raise Warning("NUM_GPUS does not match API_VISIBLE_GPUS length. Adjusting NUM_GPUS accordingly.")

torch.set_num_threads(NUM_PROCS)
torch.set_num_interop_threads(NUM_PROCS)

def is_proc_available(max_unfinished: int = max(NUM_PROCS * NUM_WORKERS - NUM_GPUS, 1)) -> bool:
    running = len(StartedJobRegistry('ef', connection=redis_conn))
    queued = EF_QUEUE.count
    return (running + queued) < max_unfinished

def is_GPU_available(max_unfinished: int = NUM_GPUS) -> bool:
    running = len(StartedJobRegistry('crt', connection=redis_conn))
    queued  = CRT_QUEUE.count
    return (running + queued) < max_unfinished

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CNVRS_DIR = Path("conversions")
CNVRS_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = Path("predicted")
PRED_DIR.mkdir(parents=True, exist_ok=True)

CRT_RUNTIME_DIR = Path("crt_runtime")
CRT_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

model_inference = None
gpu_warm_pool = None

def initialize_model_inference(device):
    return ModelInference(
        model=initialize_GINO_model(16),
        model_checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataprocessor_path=DATAPROCESSOR_PATH,
        single_case_handling=single_case_handling,
        device=device    
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_inference, gpu_warm_pool
    model_inference = initialize_model_inference("cpu")
    
    if NUM_GPUS > 0:
        gpu_warm_pool = WarmPool(API_VISIBLE_GPUS, initialize_model_inference)
        gpu_warm_pool.start()
    yield
    model_inference = None
    EF_QUEUE.enqueue("tasks.sweep_files", job_timeout=300)
    if NUM_GPUS > 0:
        gpu_warm_pool.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173/",
        "http://localhost:4173/",
        "https://dcsim.egr.msu.edu/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World!"}


@app.post("/api/convert_input_to_vtp")
async def convert_input_to_vtp(
    file: UploadFile,
):
    if not is_proc_available():
        raise HTTPException(429, "Busy: try again later")
    token = uuid.uuid4().hex
    if file.filename.lower().endswith(".vtk"):    
        input_path = UPLOAD_DIR / f"{token}.vtk"
    elif file.filename.lower().endswith(".txt"):
        input_path = UPLOAD_DIR / f"{token}.txt"
    else:
        raise HTTPException(400, "Unsupported file format")
    output_path = CNVRS_DIR / f"{token}.vtp"

    size = 0
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 10_000_000:
                f.close()
                input_path.unlink(missing_ok=True)
                raise HTTPException(400, "File size exceeds 10 MB limit")
            f.write(chunk)

    try:
        handle_input_file(str(input_path), str(output_path))
        
    except Exception as e:
        raise HTTPException(500, f"Conversion failed: {e}")
        
    token_data = {
        'input_path': str(input_path),
        'vtp_path': str(output_path),
        'original_filename': file.filename,
        'timestamp': now_iso(),
        'generate_EF_status': 'pending'
    }
    save_token_state(token, token_data)
    EF_QUEUE.enqueue("tasks.run_generate_EF", token, job_timeout=660)
    EF_QUEUE.enqueue("tasks.sweep_files", job_timeout=300)

    return FileResponse(
        path=output_path,
        filename=file.filename,
        media_type="application/octet-stream",
        headers={"X-File-Token": token, "Access-Control-Expose-Headers": "X-File-Token"}
    )

@app.post("/api/forward_run")
async def forward_run(
    file_token: str = Form(...),
    input_params: str = Form(...), 
    selected_points: str = Form(...)
    ):
    
    input_params_dict = json.loads(input_params)
    selected_points_list = json.loads(selected_points)
    d_iso = input_params_dict['d_iso']
    plocs_xyz = []
    for i in range(input_params_dict['n_plocs']):
        plocs_xyz.append(
            [selected_points_list[i]['x'],
             selected_points_list[i]['y'],
             selected_points_list[i]['z']])
    plocs_xyz = np.array(plocs_xyz)
    try:
        token_data = load_token_state(file_token)
        if not token_data:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        if token_data.get('generate_EF_status') != 'completed':
            raise HTTPException(
                status_code=400, 
                detail="generate_EF process has not completed successfully. Please wait or try again later."
            )
                
        file_path = Path(token_data['input_path'])
        if not file_path.exists():
            raise HTTPException(status_code=400, detail="Original input file not found")
        
        if gpu_warm_pool is not None:
            gpu_warm_pool.predict_and_write(
                inp_file = str(file_path),
                Diso=d_iso,
                plocs=plocs_xyz,
                file_token=file_token,
                inp_meshdir='./uploads' + '/' + file_token + '.vtk',
                xdmf_dir='./predicted' + '/' + file_token + '.xdmf'
                )
        else:
            output = model_inference.predict(str(file_path), Diso=d_iso, plocs=plocs_xyz).detach()
            sample = model_inference.sample
            model_inference.sample = None
            model_inference.output = None
            model_inference.write_xdmf(
                inp_meshdir=str(UPLOAD_DIR) + '/' + file_token + '.vtk',
                xdmf_dir=str(PRED_DIR) + '/' + file_token + '.xdmf',
                sample=sample,
                case_ID=file_token,
                output=output,
                local_error=None
            )
        
        output_xdmf = PRED_DIR / f"{file_token}.xdmf"
        output_h5 = PRED_DIR / f"{file_token}.h5"
        
        output_path = CNVRS_DIR / f"{file_token}.vtp"
        convert_xdmf_to_vtp(output_xdmf, output_path)

        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename="predicted.vtp"  
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data in input_params or selected_points")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crt_invproblem")
async def crt_invproblem(
    file_token: str = Form(...)
):
    if not is_GPU_available():
        raise HTTPException(429, "Busy: try again later")
    token_data = load_token_state(file_token)
    try:
        if not token_data:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        if token_data['generate_EF_status'] != 'completed':
            raise HTTPException(
                status_code=400, 
                detail="generate_EF process has not completed successfully. Please wait or try again later."
            )
        
        file_path = Path(token_data['input_path'])
        if not file_path.exists():
            raise HTTPException(status_code=400, detail="Original input file not found")
        
        # Initialize status as pending
        update_token_field(file_token, 'crt_invproblem_status', 'pending')
        CRT_QUEUE.enqueue("crt_tasks.run_crt_invproblem", file_token, job_timeout=1800)
        
        
        return JSONResponse(content={
            "message": "CRT inverse problem started",
            "status": "pending",
            "file_token": file_token
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crt_optim")
async def crt_optim(
    file_token: str = Form(...)
):
    token_data = load_token_state(file_token)
    try:
        if not token_data:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        if token_data['crt_invproblem_status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail="CRT inverse problem has not completed successfully. Please wait or try again later."
            )
        
        # Initialize status as pending
        update_token_field(file_token, 'crt_optim_status', 'pending')
        CRT_QUEUE.enqueue("crt_tasks.run_crt_optimization", file_token, job_timeout=1200)        
        
        return JSONResponse(content={
            "message": "CRT optimization started",
            "status": "pending", 
            "file_token": file_token
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crt_params/{file_token}")
async def get_crt_params(file_token: str):
    """Get CRT parameters (d_iso and pacing locations) from the stored CRT workflow"""
    token_data = load_token_state(file_token)
    try:
        if not token_data:
            raise HTTPException(status_code=404, detail="Invalid or expired token")
        
        crt_workflow = CRTWorkflow.load_from_state(
            file_token, CRT_RUNTIME_DIR, 
            file_path=token_data['input_path'], 
            model_inference=model_inference)
                
        if crt_workflow.d_iso is None:
            raise HTTPException(
                status_code=400,
                detail="CRT inverse problem has not been completed. d_iso is not available."
            )
        
        pacing_params = {
            "d_iso": float(crt_workflow.d_iso),
            "n_plocs": 1 if crt_workflow.ploc2_xyz is None else 2,
            "act_max_base": float(crt_workflow.act_max_base),
            "act_max_crt": float(crt_workflow.act_max_crt),
        }
        
        selected_points = []
        
        # Add first pacing location
        if crt_workflow.ploc1_xyz is not None:
            ploc1_coords = crt_workflow.ploc1_xyz.cpu().numpy().flatten()
            selected_points.append({
                "x": float(ploc1_coords[0]),
                "y": float(ploc1_coords[1]),
                "z": float(ploc1_coords[2])
            })
        
        # Add second pacing location if available
        if crt_workflow.ploc2_xyz is not None:
            ploc2_coords = crt_workflow.ploc2_xyz.cpu().numpy().flatten()
            selected_points.append({
                "x": float(ploc2_coords[0]),
                "y": float(ploc2_coords[1]),
                "z": float(ploc2_coords[2])
            })
            pacing_params["n_plocs"] = 2
        
        return JSONResponse(content={
            "pacing_params": pacing_params,
            "selected_points": selected_points
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cleanup_files")
async def cleanup_files(request: Request):    
    data = await request.json()
    file_token = data.get("file_token")
    
    token_state = load_token_state(file_token)
    if not token_state:
        return JSONResponse(status_code=404, content={"message": "Invalid or expired token"})
    
    try:
        file_path = Path(token_state['input_path'])
        output_path = CNVRS_DIR / f"{file_token}.vtp"
        output_xdmf = PRED_DIR / f"{file_token}.xdmf"
        output_h5 = PRED_DIR / f"{file_token}.h5"
        output_pkl = CRT_RUNTIME_DIR / f"{file_token}.pkl"
        for path in [file_path, output_xdmf, output_h5, output_path, output_pkl]:
            if path and path.exists():
                os.remove(path)
        print(f"Cleaned up files for {file_token}")
        return JSONResponse(content={"message": "Files cleaned up successfully"})
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error cleaning up files: {str(e)}"}
        )


@app.get("/api/ef_status/{file_token}")
async def get_status(file_token: str):
    token_state = load_token_state(file_token)
    return {"efStatus": (token_state or {}).get("generate_EF_status", "unknown")}


@app.get("/api/crt_invproblem_status/{file_token}")
async def get_crt_invproblem_status(file_token: str):
    """Get CRT inverse problem status"""
    token_data = load_token_state(file_token)
    if not token_data:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    return {"crtInvproblemStatus": (token_data or {}).get("crt_invproblem_status", "unknown")}


@app.get("/api/crt_optim_status/{file_token}")
async def get_crt_optim_status(file_token: str):
    """Get CRT optimization status"""
    token_data = load_token_state(file_token)
    if not token_data:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    return {"crtOptimStatus": (token_data or {}).get("crt_optim_status", "unknown")}


@app.get("/api/crt_invproblem_result/{file_token}")
async def get_crt_invproblem_result(file_token: str):
    """Get CRT inverse problem result file"""
    token_data = load_token_state(file_token)
    try:
        if not token_data:
            raise HTTPException(status_code=404, detail="Invalid or expired token")
        
        if token_data['crt_invproblem_status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail="CRT inverse problem has not completed successfully."
            )
        
        output_xdmf = PRED_DIR / f"{file_token}.xdmf"
        output_path = CNVRS_DIR / f"{file_token}.vtp"
        convert_xdmf_to_vtp(output_xdmf, output_path)
        
        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename="reconstructed_input.vtp"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crt_optim_result/{file_token}")
async def get_crt_optim_result(file_token: str):
    """Get CRT optimization result file"""
    token_data = load_token_state(file_token)
    try:
        if not token_data:
            raise HTTPException(status_code=404, detail="Invalid or expired token")
        
        if token_data['crt_optim_status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail="CRT optimization has not completed successfully."
            )
        
        output_xdmf = PRED_DIR / f"{file_token}.xdmf"
        output_path = CNVRS_DIR / f"{file_token}.vtp"
        convert_xdmf_to_vtp(output_xdmf, output_path)
        
        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename="predicted.vtp"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

