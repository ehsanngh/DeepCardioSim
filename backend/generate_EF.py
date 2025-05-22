import sys
import os
import dolfin
import vtk
import argparse
import meshio
from collections import defaultdict
from scipy.spatial import cKDTree
import numpy as np
import vtkpy3utils
from pathlib import Path

isepiflip = False
isendoflip = True
 
def main(token, tol=0.5):
    uploads_dir = Path("uploads")
    
    # Input and output paths
    input_vtk = uploads_dir / f"{token}.vtk"
    output_vtk = uploads_dir / f"{token}.vtk"
    
    if not input_vtk.exists():
        raise FileNotFoundError(f"Input VTK file not found: {input_vtk}")
    
    mesh_meshio = meshio.read(str(input_vtk))
    
    ugrid = vtkpy3utils.readUGrid(str(input_vtk))

    xmlgrid, _, _ = vtkpy3utils.extractFeNiCsBiVFacet(ugrid, geometry="LV", tol=tol)
    
    VQuadelem = dolfin.VectorElement("DG", xmlgrid.ufl_cell(), degree=0)

    fiberFS = dolfin.FunctionSpace(xmlgrid, VQuadelem)

    isendoflip = False
    ef, _, _, _, _, _ = vtkpy3utils.addLVfiber(
        xmlgrid, fiberFS, "geometry", 60, -60, [], isepiflip, isendoflip, ztol=tol
    )

    dofmap = fiberFS.dofmap()
    node_values = defaultdict(lambda: {"values": [], "coords": None})

    for cell in dolfin.cells(xmlgrid):
        cell_index = cell.index()
        cell_dofs = dofmap.cell_dofs(cell_index)
        cell_value = ef.vector()[cell_dofs]
        for vertex in dolfin.vertices(cell):
            node_index = vertex.index()
            vertex_coords = vertex.point().array()
            if node_values[node_index]["coords"] is None:
                node_values[node_index]["coords"] = vertex_coords
            node_values[node_index]["values"].append(cell_value)

    
    average_node_values = {
        node: {
            "coords": data["coords"],
            "average_value": np.mean(data["values"], axis=0)
        }
        for node, data in node_values.items()
    }
    node_indices = list(average_node_values.keys())
    coords_array = np.array([average_node_values[node]["coords"] for node in node_indices])
    values_array = np.array([average_node_values[node]["average_value"] for node in node_indices])
    values_array /= np.linalg.norm(values_array, axis=1, keepdims=True)
    coords_tree = cKDTree(coords_array)
    distances, indices = coords_tree.query(mesh_meshio.points)
    if np.all(distances < 1e-8):  
        print("Ordering matches found.")
    else:
        print("Warning: Large discrepancies found in point matching.")
    
    mesh_meshio.point_data = {}
    mesh_meshio.point_data['ef'] = values_array[indices]
    meshio.write(str(output_vtk), mesh_meshio)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True)
    args = parser.parse_args()
    tol = 0.5
    tol_step = 0.1
    max_tol = 2.0      
    success = False

    while not success and tol <= max_tol:
        try:
            main(args.token, tol=tol)
            success = True
            print(f"Success with tol = {tol}")
        except Exception as e:
            print(f"Error with tol = {tol}: {e}")
            tol += tol_step

    if not success:
        print(f"Failed to run successfully with tol up to {max_tol}")
