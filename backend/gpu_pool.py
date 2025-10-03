import os, time, threading
from typing import List, Optional
import torch
from dotenv import load_dotenv
from state_management import rdb
load_dotenv()

VERBOSE = False

OWNER_TTL = int(os.getenv("GPU_OWNER_TTL_S"))
HEARTBEAT_S = int(os.getenv("GPU_HEARTBEAT_S"))
PREEMPT_GRACE_S = int(os.getenv("GPU_PREEMPT_GRACE_S"))

def k_owner(gid): return f"gpu_owner:{gid}"
def k_preempt(gid): return f"gpu_preempt:{gid}"
def k_inflight(gid): return f"gpu_api_inflight:{gid}"

class WarmGPU:
    def __init__(self, gid: int, initialize_model_inference: callable, owner_id: str):
        self.gid = gid
        self.owner_id = owner_id
        self.initialize_model_inference = initialize_model_inference
        self.cpu_model_inference = self.initialize_model_inference("cpu")
        self.gpu_model_inference = None
        self.device = torch.device("cpu")
        self.lock = threading.RLock()

    def claim_and_warm(self) -> bool:
        if not torch.cuda.is_available():
            if VERBOSE: print(f"[GPU {self.gid}] ❌ CUDA not available")
            return False
        
        existing = rdb().get(k_owner(self.gid))
        if VERBOSE: print(f"[GPU {self.gid}] Claiming... (current owner: {existing})")
        
        if not rdb().set(k_owner(self.gid), self.owner_id, nx=True, ex=OWNER_TTL):
            if VERBOSE: print(f"[GPU {self.gid}] ❌ Failed to claim (owned by {existing})")
            return False
        
        if VERBOSE: print(f"[GPU {self.gid}] ✅ Claimed ownership, loading model...")
        with self.lock:
            self.device = torch.device(f"cuda:{self.gid}")
            self.gpu_model_inference = self.initialize_model_inference(self.device)
            self.gpu_model_inference.device = self.device
            self.gpu_model_inference.model.to(torch.device(f"cuda:{self.gid}"), non_blocking=True)
            self.gpu_model_inference.data_processor.to(torch.device(f"cuda:{self.gid}"))
        if VERBOSE: print(f"[GPU {self.gid}] ✅✅ Model loaded and WARM")
        return True

    def still_ours(self) -> bool:
        cur = rdb().get(k_owner(self.gid))
        return (cur or b"") == self.owner_id.encode()

    def preempted(self) -> bool:
        return rdb().exists(k_preempt(self.gid))

    def refresh(self):
        if self.still_ours():
            rdb().expire(k_owner(self.gid), OWNER_TTL)

    def demote(self):
        if VERBOSE: print(f"[GPU {self.gid}] >>> DEMOTION STARTED")
        with self.lock:
            if self.gpu_model_inference is not None:
                if PREEMPT_GRACE_S > 0:
                    if VERBOSE: print(f"[GPU {self.gid}]   Waiting {PREEMPT_GRACE_S}s grace period...")
                    time.sleep(PREEMPT_GRACE_S)
                if VERBOSE: print(f"[GPU {self.gid}]   Transferring state to CPU...")
                if self.gpu_model_inference.sample is not None:
                    self.cpu_model_inference.sample = self.gpu_model_inference.sample.to("cpu")
                if self.gpu_model_inference.output is not None:
                    self.cpu_model_inference.output = self.gpu_model_inference.output.to("cpu")
                if self.gpu_model_inference.local_error is not None:
                    self.cpu_model_inference.local_error = self.gpu_model_inference.local_error.to("cpu")
                if self.gpu_model_inference.case_ID is not None:
                    self.cpu_model_inference.case_ID = self.gpu_model_inference.case_ID
                if VERBOSE: print(f"[GPU {self.gid}]   Unloading GPU model...")
                del self.gpu_model_inference
                self.gpu_model_inference = None
                self.device = torch.device("cpu")
                torch.cuda.empty_cache()
        
        if self.still_ours():
            if VERBOSE: print(f"[GPU {self.gid}]   Releasing ownership key...")
            rdb().delete(k_owner(self.gid))
        else:
            if VERBOSE: print(f"[GPU {self.gid}]   Not ours anymore, skipping ownership release")
        
        if VERBOSE: print(f"[GPU {self.gid}] ✅ DEMOTION COMPLETE")

    def predict_and_write_gpu(self, inp_file, Diso, plocs, inp_meshdir, xdmf_dir, file_token):
        if VERBOSE: print(f"[GPU {self.gid}] >>> PREDICT START for {file_token}")
        with self.lock:
            if self.gpu_model_inference is not None:
                if VERBOSE: print(f"[GPU {self.gid}]   Running on GPU...")
                output = self.gpu_model_inference.predict(inp_file, Diso, plocs).detach().cpu()
                sample = self.gpu_model_inference.sample.to("cpu")
                self.gpu_model_inference.case_ID = None
                self.gpu_model_inference.sample = None
                self.gpu_model_inference.output = None
                if VERBOSE: print(f"[GPU {self.gid}]   Writing results...")
                self.gpu_model_inference.write_xdmf(
                    inp_data=None,
                    inp_meshdir=inp_meshdir,
                    xdmf_dir=xdmf_dir,
                    sample=sample,
                    case_ID=file_token,
                    output=output,
                    local_error=None
                    )
                print(f"[GPU {self.gid}] ✅✅ COMPLETE (GPU) for {file_token}")
            else:
                if VERBOSE: print(f"[GPU {self.gid}]   GPU not warm! Running on CPU...")
                output = self.cpu_model_inference.predict(inp_file, Diso, plocs).detach()
                sample = self.cpu_model_inference.sample
                self.cpu_model_inference.sample = None
                self.cpu_model_inference.output = None
                self.cpu_model_inference.write_xdmf(
                    inp_data=None,
                    inp_meshdir=inp_meshdir,
                    xdmf_dir=xdmf_dir,
                    sample=sample,
                    case_ID=file_token,
                    output=output,
                    local_error=None
                    )
                print(f"[GPU {self.gid}] ✅ COMPLETE (CPU fallback) for {file_token}")
        return None
    
    def predict_and_write_cpu(self, inp_file, Diso, plocs, inp_meshdir, xdmf_dir, file_token):
        """Force CPU execution, used for fallback"""
        if VERBOSE: print(f"[CPU FALLBACK] >>> PREDICT START for {file_token}")
        output = self.cpu_model_inference.predict(inp_file, Diso, plocs).detach()
        sample = self.cpu_model_inference.sample
        self.cpu_model_inference.sample = None
        self.cpu_model_inference.output = None
        if VERBOSE: print(f"[CPU FALLBACK]   Writing results...")
        self.cpu_model_inference.write_xdmf(
            inp_data=None,
            inp_meshdir=inp_meshdir,
            xdmf_dir=xdmf_dir,
            sample=sample,
            case_ID=file_token,
            output=output,
            local_error=None
            )
        print(f"[CPU FALLBACK] ✅✅ COMPLETE for {file_token}")
        return None

class WarmPool:
    """Keeps up to N GPUs warm; 1 in-flight request per GPU; overflow -> CPU."""
    def __init__(self, gpu_ids: List[int], initialize_model_inference: callable):
        self.owner_id = f"api:{os.getpid()}"
        if VERBOSE: print(f"\n{'='*60}")
        if VERBOSE: print(f"[WarmPool] INITIALIZING")
        if VERBOSE: print(f"[WarmPool] Owner ID: {self.owner_id}")
        if VERBOSE: print(f"[WarmPool] Target GPUs: {gpu_ids}")
        if VERBOSE: print(f"{'='*60}")
        
        r = rdb()
        if VERBOSE: print(f"[WarmPool] Checking for stale keys...")
        old_keys = []
        for gid in gpu_ids:
            for key in [k_owner(gid), k_preempt(gid), k_inflight(gid)]:
                if r.exists(key):
                    val = r.get(key)
                    if VERBOSE: print(f"[WarmPool]   STALE: {key} = {val}")
                    old_keys.append(key)
        
        if old_keys:
            r.delete(*old_keys)
            if VERBOSE: print(f"[WarmPool] ✅ Cleaned {len(old_keys)} stale keys")
        else:
            if VERBOSE: print(f"[WarmPool] ✅ No stale keys found")
        
        if VERBOSE: print(f"[WarmPool] Creating {len(gpu_ids)} WarmGPU instances...")
        self.gpus = [WarmGPU(g, initialize_model_inference, self.owner_id) for g in gpu_ids]
        if VERBOSE: print(f"[WarmPool] ✅ Initialization complete\n")
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1)

    def _loop(self):
        r = rdb()
        loop_count = 0
        while self.running:
            loop_count += 1
            if loop_count % 10 == 1:
                if VERBOSE: print(f"\n[Heartbeat #{loop_count}] Status check...")
            
            for w in self.gpus:
                if w.gpu_model_inference is not None:
                    is_preempted = w.preempted()
                    is_ours = w.still_ours()
                    
                    if loop_count % 10 == 1:
                        if VERBOSE: print(f"  GPU {w.gid}: WARM (ours={is_ours}, preempted={is_preempted})")
                    
                    if is_preempted or not is_ours:
                        if VERBOSE: print(f"[GPU {w.gid}] ⚠️  DEMOTING (preempted={is_preempted}, ours={is_ours})")
                        w.demote()
                    else:
                        w.refresh()
                else:
                    if loop_count % 10 == 1:
                        preempt = r.exists(k_preempt(w.gid))
                        if VERBOSE: print(f"  GPU {w.gid}: NOT WARM (preempt_flag={preempt})")
                    
                    if torch.cuda.is_available() and not r.exists(k_preempt(w.gid)):
                        w.claim_and_warm()
            
            time.sleep(HEARTBEAT_S)

    def _try_reserve_gpu(self) -> Optional[WarmGPU]:
        """Atomically reserve one GPU for this request (cap=1 in-flight)."""
        r = rdb()
        if VERBOSE: print(f"\n[RESERVE] Searching for available GPU...")
        
        for w in self.gpus:
            is_warm = w.gpu_model_inference is not None
            is_ours = w.still_ours()
            is_preempted = w.preempted()
            
            if VERBOSE: print(f"[RESERVE]   GPU {w.gid}: warm={is_warm}, ours={is_ours}, preempt={is_preempted}")
            
            # must be warm, owned by us, and not preempted
            if not is_warm or not is_ours or is_preempted:
                continue
            
            # INCR and check <=1 (cap 1). If >1, immediately DECR and try next.
            val = r.incr(k_inflight(w.gid))
            if VERBOSE: print(f"[RESERVE]   GPU {w.gid}: inflight counter -> {val}")
            
            if val <= 1:
                if VERBOSE: print(f"[RESERVE] ✅✅ RESERVED GPU {w.gid}")
                return w
            else:
                r.decr(k_inflight(w.gid))
                if VERBOSE: print(f"[RESERVE]   GPU {w.gid}: BUSY, trying next...")
        
        if VERBOSE: print(f"[RESERVE] ❌ NO GPU AVAILABLE -> CPU FALLBACK")
        return None

    def _release_gpu(self, w: WarmGPU):
        new_val = rdb().decr(k_inflight(w.gid))
        if VERBOSE: print(f"[Reserve] Released GPU {w.gid}, inflight now: {new_val}")

    def predict_and_write(self, **kwargs):
        """First try to reserve a GPU; else run on CPU."""
        w = self._try_reserve_gpu()
        if w is None:
            # CPU fallback
            # Use any CPU copy (they are identical)
            return self.gpus[0].predict_and_write_cpu(**kwargs)
        try:
            return w.predict_and_write_gpu(**kwargs)
        finally:
            self._release_gpu(w)


if __name__ == "__main__":
    import numpy as np
    from deepcardio.electrophysio import ModelInference
    from deepcardio.electrophysio import initialize_GINO_model, single_case_handling
    def initialize_model_inference(device):
        return ModelInference(
            model=initialize_GINO_model(16),
            model_checkpoint_path=os.getenv("MODEL_CHECKPOINT_PATH"),
            dataprocessor_path=os.getenv("DATAPROCESSOR_PATH"),
            single_case_handling=single_case_handling,
            device=device    
        )

    file_path = './uploads/geometry.vtk'
    file_token = 'geometry'
    d_iso = 0.2
    plocs_xyz = np.array([[1.133, -2.835, -2.014]])
    pool = WarmPool([0,1], initialize_model_inference)
    pool.start()
    time.sleep(5)
    pool.predict_and_write(
                inp_file = str(file_path),
                Diso=d_iso,
                plocs=plocs_xyz,
                file_token=file_token,
                inp_meshdir='./uploads' + '/' + file_token + '.vtk',
                xdmf_dir='./predicted' + '/' + file_token + '.xdmf'
                )
    pool.stop()