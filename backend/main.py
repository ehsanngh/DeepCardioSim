from fastapi import FastAPI, UploadFile, HTTPException, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from deepcardio.electrophysio import ModelInference
from deepcardio.electrophysio import model, single_case_handling
from typing import Optional, Dict
import logging
from deepcardio.electrophysio import data_processing
import sys
sys.modules['data_processing'] = data_processing
from format_conversions import convert_xdmf_to_vtp
from InputProcessing import handle_input_file
import uuid
from fastapi.responses import FileResponse, JSONResponse
import json
from datetime import datetime, timedelta
import numpy as np
import subprocess
import os
from dotenv import load_dotenv
from CRTPlanning import CRTWorkflow
load_dotenv()

logging.basicConfig(level=logging.DEBUG)

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

file_mappings: Dict[str, dict] = {}


def cleanup_old_files():
    """Remove files older than 1 hour"""
    current_time = datetime.now()
    to_remove = []
    
    for token, mapping in file_mappings.items():
        if current_time - mapping['timestamp'] > timedelta(hours=1):
            try:
                Path(mapping['input_path']).unlink(missing_ok=True)
                Path(mapping['vtp_path']).unlink(missing_ok=True)
                
                xdmf_path = PRED_DIR / f"{token}.xdmf"
                h5_path = PRED_DIR / f"{token}.h5"
                xdmf_path.unlink(missing_ok=True)
                h5_path.unlink(missing_ok=True)
                
                to_remove.append(token)
            except Exception as e:
                logging.error(f"Error cleaning up files for token {token}: {e}")
    
    for token in to_remove:
        del file_mappings[token]


def run_generate_EF(token: str) -> bool:
    """Run generate_EF in the Singularity container"""
    try:
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
        
        logging.info(f"generate_EF output: {result.stdout}")
        if result.stderr:
            logging.warning(f"generate_EF warnings: {result.stderr}")
        
        if token in file_mappings:
            file_mappings[token]['generate_EF_status'] = 'completed'
            
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running generate_EF: {e}")
        logging.error(f"Command output: {e.output}")
        logging.error(f"Command stderr: {e.stderr}")
        if token in file_mappings:
            file_mappings[token]['generate_EF_status'] = 'failed'
        return False
    except Exception as e:
        logging.error(f"Unexpected error running generate_EF: {e}")
        if token in file_mappings:
            file_mappings[token]['generate_EF_status'] = 'failed'
        return False
    

def run_crt_invproblem(token: str) -> bool:
    """Run CRT inverse problem in background"""
    try:
        if token not in file_mappings:
            logging.error(f"Token {token} not found in file_mappings")
            return False
            
        file_mapping = file_mappings[token]
        file_path = Path(file_mapping['input_path'])
        
        # Update status to running
        file_mappings[token]['crt_invproblem_status'] = 'running'
        
        # Initialize CRT workflow
        sample = model_inference.file_to_inp_data(file=str(file_path))
        crt_workflow = CRTWorkflow(sample)
        logging.info(f"CRT workflow initialized for token {token}")
        
        # Update status to inverse problem running
        file_mappings[token]['crt_invproblem_status'] = 'inverse_problem_running'
        
        # Run inverse problem
        crt_workflow.run_invproblem(
            file_token=token,
            model_inference=model_inference,
            mesh_directory=str(UPLOAD_DIR) + '/',
            xdmf_directory=str(PRED_DIR) + '/')
        
        # Store the workflow and update status
        file_mappings[token]['crt_workflow'] = crt_workflow
        file_mappings[token]['crt_invproblem_status'] = 'completed'
        
        logging.info(f"CRT inverse problem completed for token {token}")
        return True
        
    except Exception as e:
        logging.error(f"Error in CRT inverse problem for token {token}: {e}")
        if token in file_mappings:
            file_mappings[token]['crt_invproblem_status'] = 'failed'
        return False


def run_crt_optimization(token: str) -> bool:
    """Run CRT optimization in background"""
    try:
        if token not in file_mappings:
            logging.error(f"Token {token} not found in file_mappings")
            return False
            
        file_mapping = file_mappings[token]
        
        if 'crt_workflow' not in file_mapping:
            logging.error(f"CRT workflow not found for token {token}")
            file_mappings[token]['crt_optim_status'] = 'failed'
            return False
        
        # Update status to running
        file_mappings[token]['crt_optim_status'] = 'running'
        
        crt_workflow = file_mapping['crt_workflow']
        
        # Update status to optimization running
        file_mappings[token]['crt_optim_status'] = 'optimization_running'
        
        # Run optimization
        crt_workflow.run_optim(
            file_token=token,
            model_inference=model_inference,
            mesh_directory=str(UPLOAD_DIR) + '/',
            xdmf_directory=str(PRED_DIR) + '/')
        
        # Update status to completed
        file_mappings[token]['crt_optim_status'] = 'completed'
        
        logging.info(f"CRT optimization completed for token {token}")
        return True
        
    except Exception as e:
        logging.error(f"Error in CRT optimization for token {token}: {e}")
        if token in file_mappings:
            file_mappings[token]['crt_optim_status'] = 'failed'
        return False


app = FastAPI()
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
    background: BackgroundTasks
):
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
        
    file_mappings[token] = {
        'input_path': str(input_path),
        'vtp_path': str(output_path),
        'original_filename': file.filename,
        'timestamp': datetime.now(),
        'generate_EF_status': 'pending'  # Initialize status as pending
    }
    
    background.add_task(run_generate_EF, token)
    
    cleanup_old_files()

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
    print(f"input_params_dict: {input_params_dict}")
    print(f"selected_points_list: {selected_points_list}")
    d_iso = input_params_dict['d_iso']
    plocs_xyz = []
    for i in range(input_params_dict['n_plocs']):
        plocs_xyz.append(
            [selected_points_list[i]['x'],
             selected_points_list[i]['y'],
             selected_points_list[i]['z']])
    plocs_xyz = np.array(plocs_xyz)
    try:
        if file_token not in file_mappings:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        file_mapping = file_mappings[file_token]
        
        if file_mapping.get('generate_EF_status') != 'completed':
            raise HTTPException(
                status_code=400, 
                detail="generate_EF process has not completed successfully. Please wait or try again later."
            )
        
        file_path = Path(file_mapping['input_path'])
        if not file_path.exists():
            raise HTTPException(status_code=400, detail="Original input file not found")
        
        model_inference.predict(str(file_path), Diso=d_iso, plocs=plocs_xyz, r=0.55)
        model_inference.case_ID = file_token
        model_inference.write_xdmf(
            mesh_directory=str(UPLOAD_DIR) + '/',
            xdmf_directory=str(PRED_DIR) + '/'
        )

        output_xdmf = Path(model_inference.xdmf_file)
        output_h5 = Path(model_inference.xdmf_file[:-4] + 'h5')
        
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
    background: BackgroundTasks,
    file_token: str = Form(...)
):
    try:
        if file_token not in file_mappings:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        file_mapping = file_mappings[file_token]
        
        if file_mapping.get('generate_EF_status') != 'completed':
            raise HTTPException(
                status_code=400, 
                detail="generate_EF process has not completed successfully. Please wait or try again later."
            )
        
        file_path = Path(file_mapping['input_path'])
        if not file_path.exists():
            raise HTTPException(status_code=400, detail="Original input file not found")
        
        # Initialize status as pending
        file_mappings[file_token]['crt_invproblem_status'] = 'pending'
        
        # Start background task
        background.add_task(run_crt_invproblem, file_token)
        
        return JSONResponse(content={
            "message": "CRT inverse problem started",
            "status": "pending",
            "file_token": file_token
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crt_optim")
async def crt_optim(
    background: BackgroundTasks,
    file_token: str = Form(...)
):
    try:
        if file_token not in file_mappings:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        file_mapping = file_mappings[file_token]
        
        if 'crt_workflow' not in file_mapping:
            raise HTTPException(
                status_code=400, 
                detail="CRT workflow has not been initialized. Please run crt_invproblem first."
            )
        
        if file_mapping.get('crt_invproblem_status') != 'completed':
            raise HTTPException(
                status_code=400,
                detail="CRT inverse problem has not completed successfully. Please wait or try again later."
            )
        
        # Initialize status as pending
        file_mappings[file_token]['crt_optim_status'] = 'pending'
        
        # Start background task
        background.add_task(run_crt_optimization, file_token)
        
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
    try:
        if file_token not in file_mappings:
            raise HTTPException(status_code=404, detail="Invalid or expired token")
        
        file_mapping = file_mappings[file_token]
        
        if 'crt_workflow' not in file_mapping:
            raise HTTPException(
                status_code=400, 
                detail="CRT workflow has not been initialized. Please run crt_invproblem first."
            )
        
        crt_workflow = file_mapping['crt_workflow']
        
        # Check if the required parameters are available
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
    
    if not file_token or file_token not in file_mappings:
        return JSONResponse(
            status_code=404,
            content={"message": f"Token {file_token} not found"}
        )
    
    try:
        file_mapping = file_mappings[file_token]
        file_path = Path(file_mapping['input_path'])
        output_path = CNVRS_DIR / f"{file_token}.vtp"
        output_xdmf = PRED_DIR / f"{file_token}.xdmf"
        output_h5 = PRED_DIR / f"{file_token}.h5"
        for path in [file_path, output_xdmf, output_h5, output_path]:
            if path and path.exists():
                os.remove(path)
        
        del file_mappings[file_token]
        
        return JSONResponse(content={"message": "Files cleaned up successfully"})
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error cleaning up files: {str(e)}"}
        )


@app.get("/api/ef_status/{file_token}")
async def get_status(file_token: str):
    if file_token not in file_mappings:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    
    status = file_mappings[file_token].get('generate_EF_status', 'unknown')
    return {"efStatus": status}


@app.get("/api/crt_invproblem_status/{file_token}")
async def get_crt_invproblem_status(file_token: str):
    """Get CRT inverse problem status"""
    if file_token not in file_mappings:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    
    status = file_mappings[file_token].get('crt_invproblem_status', 'unknown')
    return {"crtInvproblemStatus": status}


@app.get("/api/crt_optim_status/{file_token}")
async def get_crt_optim_status(file_token: str):
    """Get CRT optimization status"""
    if file_token not in file_mappings:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    
    status = file_mappings[file_token].get('crt_optim_status', 'unknown')
    return {"crtOptimStatus": status}


@app.get("/api/crt_invproblem_result/{file_token}")
async def get_crt_invproblem_result(file_token: str):
    """Get CRT inverse problem result file"""
    try:
        if file_token not in file_mappings:
            raise HTTPException(status_code=404, detail="Invalid or expired token")
        
        file_mapping = file_mappings[file_token]
        
        if file_mapping.get('crt_invproblem_status') != 'completed':
            raise HTTPException(
                status_code=400,
                detail="CRT inverse problem has not completed successfully."
            )
        
        # Convert and return the result
        output_xdmf = Path(model_inference.xdmf_file)
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
    try:
        if file_token not in file_mappings:
            raise HTTPException(status_code=404, detail="Invalid or expired token")
        
        file_mapping = file_mappings[file_token]
        
        if file_mapping.get('crt_optim_status') != 'completed':
            raise HTTPException(
                status_code=400,
                detail="CRT optimization has not completed successfully."
            )
        
        # Convert and return the result
        output_xdmf = Path(model_inference.xdmf_file)
        output_path = CNVRS_DIR / f"{file_token}.vtp"
        convert_xdmf_to_vtp(output_xdmf, output_path)
        
        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename="predicted.vtp"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


