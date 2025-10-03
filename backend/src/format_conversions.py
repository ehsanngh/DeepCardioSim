import vtk
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree


def convert_pcd_to_plymesh(input_file, output_file):
    input_data = np.loadtxt(input_file)
    points = input_data[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    alpha = 0.6
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    o3d.io.write_triangle_mesh(output_file, mesh)

    return None


def convert_vtk_to_plymesh(input_file, output_file):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()

    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(reader.GetOutputPort())
    surface.Update()

    writer = vtk.vtkPLYWriter()
    writer.SetFileName(output_file)
    writer.SetInputConnection(surface.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.Write()

    return None


def convert_vtk_to_vtp(input_file, output_file):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()

    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(reader.GetOutputPort())
    surface.Update()

    surface_data = surface.GetOutput()
    point_data = surface_data.GetPointData()
    
    computed_labels = point_data.GetArray("computed_labels")
    if computed_labels is not None:
        computed_labels.SetName("computed labels")

    t_act_array = point_data.GetArray("t_act")
    if t_act_array is not None:
        t_act_array.SetName("measured act map (ms)")

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(surface_data)
    writer.Write()

    return None


def convert_xdmf_to_vtp(input_file, output_file):
    print(input_file)
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(input_file)
    reader.Update()
    
    data = reader.GetOutput()
    
    if data.IsA("vtkMultiBlockDataSet"):
        data = data.GetBlock(0)
    
    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputData(data)
    surfaceFilter.Update()

    surface_data = surfaceFilter.GetOutput()
    
    point_data = surface_data.GetPointData()

    ef = point_data.GetArray("ef")
    if ef is not None:
        ef.SetName("fiber vectors")

    ploc_bool = point_data.GetArray("ploc_bool")
    if ploc_bool is not None:
        ploc_bool.SetName("pacing site(s)")

    computed_labels = point_data.GetArray("computed_labels")
    if computed_labels is not None:
        computed_labels.SetName("computed labels")

    y_true_array = point_data.GetArray("y_true")
    if y_true_array is not None:
        y_true_array.SetName("measured act map (ms)")
    
    y_est_array = point_data.GetArray("y_est")
    if y_est_array is not None:
        y_est_array.SetName("est act map (ms)")

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(surface_data)
    writer.Write()

    return None


def convert_ply_to_vtp(input_ply_file, output_vtp_file, original_pcd, pcd_data):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(input_ply_file)
    reader.Update()

    mesh = reader.GetOutput()

    mesh_points = []
    for i in range(mesh.GetNumberOfPoints()):
        point = mesh.GetPoint(i)
        mesh_points.append(point)
    mesh_points = np.array(mesh_points)

    tree = cKDTree(original_pcd)

    distances, indices = tree.query(mesh_points)

    for field_name, field_values in pcd_data.items():
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetName(field_name)
        vtk_array.SetNumberOfComponents(1 if field_values.ndim == 1 else field_values.shape[1])
    
        if field_values.ndim == 1:
            interpolated_values = field_values[indices]
            for value in interpolated_values:
                vtk_array.InsertNextValue(value)
        else:
            interpolated_values = field_values[indices]
            for value in interpolated_values:
                vtk_array.InsertNextTuple(value)
    
        mesh.GetPointData().AddArray(vtk_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_vtp_file)
    writer.SetInputData(mesh)
    writer.Write()

    return None