import open3d as o3d
import numpy as np
import meshio
from scipy.spatial import cKDTree
from src.format_conversions import convert_vtk_to_plymesh, convert_pcd_to_plymesh, convert_ply_to_vtp, convert_vtk_to_vtp
import os

def LVsurface_detection(input_file):
    mesh = o3d.io.read_triangle_mesh(input_file)
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    z_coords = points[:, 2] 
    normal_z = normals[:, 2]
    max_z = np.max(z_coords)
    normal_threshold = 0.7
    z_threshold = 0.125 * (max_z - np.min(z_coords))
    computed_label = np.ones(len(points), dtype=int)  # 1 is epicardium
    close_to_top_normal = normal_z > normal_threshold
    close_to_max_z = z_coords > (max_z - z_threshold)
    computed_label[close_to_top_normal & close_to_max_z] = 4  # 4 is base

    center = np.mean(points, axis=0)
    center_to_point = points - center
    center_to_point_normalized = center_to_point / np.linalg.norm(center_to_point, axis=1, keepdims=True)
    dot_products = np.sum(normals * center_to_point_normalized, axis=1)
    threshold = 0.
    endocardium_mask = dot_products < threshold
    not_base = computed_label != 4
    computed_label[endocardium_mask & not_base] = 2  # 2 is endocardium

    return np.concatenate((points, computed_label[:, np.newaxis]), axis=1)


def handle_input_file(input_file, output_vtp_file):
    ext = input_file.lower()
    plymesh_file = input_file[:-4] + ".ply"
    
    if ext.endswith(".vtk"):
        convert_vtk_to_plymesh(input_file, plymesh_file)
        original_file = meshio.read(input_file)
        original_points = original_file.points
        file_type = "vtk"
        
    elif ext.endswith(".txt"):
        convert_pcd_to_plymesh(input_file, plymesh_file)
        original_file = np.loadtxt(input_file)
        original_points = original_file[:, :3]
        file_type = "txt"
        
    else:
        raise ValueError(f"Unsupported file type: {input_file}")
    
    updated_surface_data = LVsurface_detection(plymesh_file)
    surface_points = updated_surface_data[:, :3]
    surface_label = updated_surface_data[:, 3]
    
    point_label = np.zeros(len(original_points))  # 0 is myocardium
    tree = cKDTree(original_points)
    distances, indices = tree.query(surface_points, k=1)
    tolerance = 1e-10
    valid_matches = distances < tolerance
    point_label[indices[valid_matches]] = surface_label[valid_matches]
    
    if file_type == "vtk":
        if not hasattr(original_file, "point_data") or original_file.point_data is None:
            original_file.point_data = {}
        original_file.point_data["computed_labels"] = point_label[:, np.newaxis]
        meshio.write(input_file, original_file)
        convert_vtk_to_vtp(input_file, output_vtp_file)
        
    elif file_type == "txt":
        original_file = np.concatenate((original_file, point_label[:, np.newaxis]), axis=1)

        point_data = {"t_act": original_file[:, 3:4],
                      "computed_labels": point_label[:, np.newaxis]}
        np.savetxt(input_file, original_file)
        convert_ply_to_vtp(
            plymesh_file,
            output_vtp_file,
            original_points,
            point_data)
    
    os.remove(plymesh_file)
    return None

