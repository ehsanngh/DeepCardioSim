import subprocess, logging
import os
from pathlib import Path
import time
from state_management import now_iso, update_token_field, load_token_state
from dotenv import load_dotenv
load_dotenv()

FENICS_CONTAINER = os.getenv('FENICS_CONTAINER_PATH')

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CNVRS_DIR = Path("conversions")
CNVRS_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = Path("predicted")
PRED_DIR.mkdir(parents=True, exist_ok=True)

CRT_RUNTIME_DIR = Path("crt_runtime")
CRT_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def run_generate_EF(token: str, num_procs=1) -> bool:
    print(f"Running generate_EF for token {token} with {num_procs} processes")
    """Run generate_EF in the Singularity container"""
    token_data = load_token_state(token)
    if not token_data:
        logging.error(f"Token {token} not found in state management")
        return False
    update_token_field(token, "generate_EF_status", "running")
    update_token_field(token, "generate_EF_started_at", now_iso())
    
    cmd = [
        "singularity", "exec",
        FENICS_CONTAINER,
        # "mpirun.mpich", "-np", str(num_procs),  # TODO: Hangs with mpirun
        "python3", "src/generate_EF.py",
        "--token", token,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        stdout_tail = (proc.stdout or "")[-10000:]
        stderr_tail = (proc.stderr or "")[-10000:]
        
        update_token_field(token, "generate_EF_returncode", proc.returncode)
        if stdout_tail: update_token_field(token, "generate_EF_stdout_tail", stdout_tail)
        if stderr_tail: update_token_field(token, "generate_EF_stderr_tail", stderr_tail)
        update_token_field(token, "generate_EF_ended_at", now_iso())

        ok = proc.returncode == 0
        update_token_field(token, "generate_EF_status", "completed" if ok else "failed")
        logging.info("generate_EF[%s] rc=%s", token, proc.returncode)
        print(f"generate_EF[{token}] rc={proc.returncode}")
        return ok
    
    except Exception as e:
        logging.exception("generate_EF[%s] crashed", token)
        update_token_field(token, "generate_EF_status", "failed")
        update_token_field(token, "generate_EF_error", repr(e))
        update_token_field(token, "generate_EF_ended_at", now_iso())
        print(f"generate_EF[{token}] crashed: {e}")
        return False


def sweep_files(threshold_secs: int = 3600 * 2) -> int:
    """
    Delete files older than threshold if their token is not active.
    Returns count of tokens cleaned.
    """
    cleaned = 0

    for f in UPLOAD_DIR.iterdir():
        if not f.is_file(): 
            continue
        token = f.stem
        vtp = CNVRS_DIR / f"{token}.vtp"
        xdmf = PRED_DIR / f"{token}.xdmf"
        h5   = PRED_DIR / f"{token}.h5"
        pkl = CRT_RUNTIME_DIR / f"{token}.pkl"

        # use the newest mtime among the set as the token age
        mtimes = [p.stat().st_mtime for p in [f, vtp, xdmf, h5, pkl] if p.exists()]
        if not mtimes:
            continue
        age_ok = (time.time() - max(mtimes)) > threshold_secs
        token_data = load_token_state(token)

        if age_ok and not token_data:
            for p in [f, vtp, xdmf, h5, pkl]:
                try:
                    p.unlink(missing_ok=True)
                except Exception as e:
                    logging.warning(f"GC: failed to delete {p}: {e}")
            cleaned += 1

    for dir_ in (CNVRS_DIR, PRED_DIR, CRT_RUNTIME_DIR):
        for f in dir_.iterdir():
            if not f.is_file(): 
                continue
            token = f.stem

            age_ok = (time.time() - f.stat().st_mtime) > threshold_secs
            token_data = load_token_state(token)

            if age_ok and not token_data:
                try: f.unlink(missing_ok=True); cleaned += 1
                except Exception as e: logging.warning(f"GC: {f}: {e}")

    logging.info(f"GC: cleaned {cleaned} token(s)")
    return cleaned
