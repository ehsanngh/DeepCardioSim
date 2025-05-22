import vtk

def convert_vtk_to_vtp(input_file, output_file):
        
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()

    # Extract the surface as PolyData
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(reader.GetOutputPort())
    surface.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputConnection(surface.GetOutputPort())
    writer.Write()

    return None


def convert_xdmf_to_vtp(input_file, output_file):
    print(input_file)
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(input_file)
    reader.Update()
    
    # Get the output data
    data = reader.GetOutput()
    
    # If the data is composite, we need to extract the first block
    if data.IsA("vtkMultiBlockDataSet"):
        data = data.GetBlock(0)
    
    # Create surface filter
    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputData(data)
    surfaceFilter.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputConnection(surfaceFilter.GetOutputPort())
    writer.Write()

    return None