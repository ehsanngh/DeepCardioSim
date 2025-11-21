import sys
import dolfin
import argparse
from scipy.spatial import cKDTree
import numpy as np
from vtkpy3utils import readUGrid, extractFeNiCsBiVFacet, addLVfiber, writeUGrid
from pathlib import Path
from vtk.util import numpy_support
import time

dolfin.flags = ["-O3", "-ffast-math", "-march=native"]
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

isepiflip = False
isendoflip = False
 
def main(token, LVangle_endo=60., LVangle_epi=-60., tol=0.5):
    uploads_dir = Path(__file__).resolve().parent.parent / "uploads"
    
    # Input and output paths
    input_vtk = uploads_dir / f"{token}.vtk"
    output_vtk = uploads_dir / f"{token}.vtk"
    hdf5_dir = output_vtk.as_posix().replace(".vtk", ".hdf5")
    tol_is_okay = True
    start = time.time()
    comm = dolfin.MPI.comm_world
    rank = dolfin.MPI.rank(comm)

    # np.random.seed(case_ID + 1001)
    # LVangle_1 = np.random.uniform(40, 80)
    # LVangle_2 = np.random.uniform(-80, -40)

    if tol_is_okay:
        try:
            if rank == 0:
                print(f"LVangle_endo: {LVangle_endo}, LVangle_epi: {LVangle_epi}")
                ugrid = readUGrid(input_vtk.as_posix())
                xmlgrid, _, _ = extractFeNiCsBiVFacet(ugrid, geometry="LV", tol=tol)
                with dolfin.HDF5File(dolfin.MPI.comm_self, hdf5_dir, "w") as f:
                    f.write(xmlgrid, "geometry")
        except Exception as e:
            tol_is_okay = False
        tol_is_okay = comm.allreduce(tol_is_okay, op=pyMPI.LAND)
        if not tol_is_okay:
            if rank == 0:
                print(f"Process failed in mesh processing with tol = {tol}")
            raise RuntimeError()
        dolfin.MPI.barrier(comm)

    xmlgrid = dolfin.Mesh()
    with dolfin.HDF5File(comm, hdf5_dir, "r") as f:
        f.read(xmlgrid, "geometry", False)
    if rank == 0:
        Path(hdf5_dir).unlink()

    fiberFS_nodal = dolfin.VectorFunctionSpace(xmlgrid, "CG", 1)
    ef_projected = dolfin.Function(fiberFS_nodal)

    isendoflip = False
    ef_projected, *_ = addLVfiber(
        xmlgrid, fiberFS_nodal, "geometry", LVangle_endo, LVangle_epi, [], isepiflip, isendoflip, ztol=tol
    )
    dolfin.MPI.barrier(comm)

    efx, efy, efz = ef_projected.split(deepcopy=True)
    S = efx.function_space()
    gdim = xmlgrid.geometry().dim()

    coords_loc = S.tabulate_dof_coordinates().reshape(-1, gdim)  
    gdofs_loc  = np.array(S.dofmap().dofs())                            
    own_start, own_end = S.dofmap().ownership_range()
    owned_mask = (gdofs_loc >= own_start) & (gdofs_loc < own_end)

    coords_owned = coords_loc[owned_mask]
    vals_owned = np.column_stack([
        efx.vector().get_local()[owned_mask],
        efy.vector().get_local()[owned_mask],
        efz.vector().get_local()[owned_mask],
    ])

    all_coords = comm.gather(coords_owned, root=0)
    all_vals = comm.gather(vals_owned, root=0)

    if rank == 0:
        coords_all = np.concatenate(all_coords, axis=0)
        vals_all = np.concatenate(all_vals, axis=0)

        norms = np.linalg.norm(vals_all, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vals_all = vals_all / norms

        vtk_pts = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData())
        kdt = cKDTree(coords_all)
        dist, idx = kdt.query(vtk_pts)

        if np.any(dist > 1e-8):
            raise ValueError(f"Point match tolerance exceeded: max Δ = {dist.max()}")


        ef_vtk = numpy_support.numpy_to_vtk(vals_all[idx], deep=True)
        ef_vtk.SetName("ef")
        ugrid.GetPointData().AddArray(ef_vtk)
        writeUGrid(ugrid, output_vtk.as_posix())

        print(f"Finished case {args.token} in {time.time() - start} seconds.")
    
    return None

if __name__ == "__main__":
    import sys
    from mpi4py import MPI as pyMPI
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True)
    args = parser.parse_args()
    
    tol = 0.5      # Initial tolerance
    tol_step = 0.1 # Step size for incrementing tolerance
    max_tol = 2.0  # Maximum tolerance limit
    success = False
    while not success and tol <= max_tol:
        try:
            main(
                args.token,
                tol=tol)
            success = True
        except Exception as e:
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                print(f"Process failed with tol = {tol}: {e}")
            success = False
        success = dolfin.MPI.comm_world.allreduce(success, op=pyMPI.LAND)
        if not success:
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                print(f"Retrying with increased tol. Current tol = {tol}")
            tol += tol_step

    if not success:
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            print(f"Failed to run successfully with tol up to {max_tol}")
        sys.exit(1)
    
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(f"Success with tol = {tol}")
    sys.exit(0)

        