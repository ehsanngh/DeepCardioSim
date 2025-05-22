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
from format_conversions import convert_xdmf_to_vtp, convert_vtk_to_vtp
import uuid
from fastapi.responses import FileResponse, JSONResponse
import json
from datetime import datetime, timedelta
import numpy as np
import time
import subprocess
import os
logging.basicConfig(level=logging.DEBUG)

FENICS_CONTAINER = r"/mnt/home/naghavis/Documents/Research/FEniCSx/fenics_legacy3.sif"

model_inference = ModelInference(
    model=model,
    model_checkpoint_path=
    '/mnt/home/naghavis/Documents/Research/DeepCardioSim/cardiac_examples/electrophysio/GINO/ckpt/ckpt_16/' + 'best_model_snapshot_dict.pt',
    dataprocessor_path=
    '/mnt/home/naghavis/Documents/Research/DeepCardioSim/cardiac_examples/electrophysio/GINO/data_processor.pt',
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
                Path(mapping['vtk_path']).unlink(missing_ok=True)
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
    

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello World!"}


@app.post("/api/convert_vtk_to_vtp")
async def convert_vtk_to_vtp_endpoint(
    file: UploadFile,
    background: BackgroundTasks
):
    if not file.filename.lower().endswith(".vtk"):
        raise HTTPException(400, "Only .vtk files are allowed")
    
    token = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{token}.vtk"
    output_path = CNVRS_DIR / f"{token}.vtp"

    size = 0
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 10_000_000:
                # Clean up partial file
                f.close()
                input_path.unlink(missing_ok=True)
                raise HTTPException(400, "File size exceeds 10 MB limit")
            f.write(chunk)

    try:
        convert_vtk_to_vtp(str(input_path), str(output_path))
        
        file_mappings[token] = {
            'vtk_path': str(input_path),
            'vtp_path': str(output_path),
            'original_filename': file.filename,
            'timestamp': datetime.now(),
            'generate_EF_status': 'pending'  # Initialize status as pending
        }
        
        background.add_task(run_generate_EF, token)
        
        cleanup_old_files()
        
    except Exception as e:
        raise HTTPException(500, f"Conversion failed: {e}")

    print(f"File token: {token}")
    return FileResponse(
        path=output_path,
        filename=file.filename,
        media_type="application/octet-stream",
        headers={"X-File-Token": token, "Access-Control-Expose-Headers": "X-File-Token"}
    )


@app.post("/api/predict_and_send")
async def predict_and_send(
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
        if file_token not in file_mappings:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        file_mapping = file_mappings[file_token]
        
        if file_mapping.get('generate_EF_status') != 'completed':
            raise HTTPException(
                status_code=400, 
                detail="generate_EF process has not completed successfully. Please wait or try again later."
            )
        
        file_path = Path(file_mapping['vtk_path'])
        print(f"File path: {file_path}")
        if not file_path.exists():
            raise HTTPException(status_code=400, detail="Original VTK file not found")
        
        start_time = time.time()
        model_inference.predict(str(file_path), Diso=d_iso, plocs=plocs_xyz, r=0.55)
        print(f"Time taken to predict: {time.time() - start_time} seconds")
        model_inference.case_ID = file_token
        model_inference.write_xdmf(
            mesh_directory=str(UPLOAD_DIR) + '/',
            xdmf_directory=str(PRED_DIR) + '/'
        )

        output_xdmf = Path(model_inference.xdmf_file)
        output_h5 = Path(model_inference.xdmf_file[:-4] + 'h5')
        
        output_path = CNVRS_DIR / f"{file_token}.vtp"
        convert_xdmf_to_vtp(output_xdmf, output_path)

        print(f"output path: {output_path}")

        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename="predicted.vtp"  
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data in input_params or selected_points")
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
        file_path = Path(file_mapping['vtk_path'])
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

