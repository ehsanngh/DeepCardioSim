import vtk
import dolfin
import numpy as np
from vtk.util import numpy_support
import math


def readUGrid(mesh_file_name, verbose=True):
    if (verbose): print('*** readUGrid ***')
    
    mesh_reader = vtk.vtkUnstructuredGridReader()
    mesh_reader.SetFileName(mesh_file_name)
    mesh_reader.Update()
    mesh = mesh_reader.GetOutput()
    
    if (verbose):
        nb_points = mesh.GetNumberOfPoints()
        print('nb_points =', nb_points)
        nb_cells = mesh.GetNumberOfCells()
        print('nb_cells =', nb_cells)
    
    return mesh


def writeUGrid(ugrid, ugrid_file_name, verbose=True):
    if (verbose): print('*** writeUGrid ***')

    ugrid_writer = vtk.vtkUnstructuredGridWriter()
    ugrid_writer.SetFileName(ugrid_file_name)
    
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        ugrid_writer.SetInputData(ugrid)
    else:
        ugrid_writer.SetInput(ugrid)
    ugrid_writer.Update()
    ugrid_writer.Write()


def extractCellFromPData(cellidlist, pdata):
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(cellidlist)
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)
    extractSelection = vtk.vtkExtractSelection()
    
    if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
        extractSelection.SetInput(0, pdata)
        extractSelection.SetInput(1, selection)
    else:	
        extractSelection.SetInputData(0, pdata)
        extractSelection.SetInputData(1, selection)
    extractSelection.Update()
    extractbase = vtk.vtkGeometryFilter()
    
    if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
        extractbase.SetInput(extractSelection.GetOutput())
    else:
        extractbase.SetInputData(extractSelection.GetOutput())
    extractbase.Update()

    return extractbase.GetOutput()


def convertUGridToXMLMesh(ugrid):
    num_pts = ugrid.GetNumberOfPoints()
    num_cells =  ugrid.GetNumberOfCells()
    
    celltypes = numpy_support.vtk_to_numpy(ugrid.GetCellTypesArray())
    num_tetra = np.count_nonzero(celltypes == 10)
    
    print("Number of points  = ", num_pts)
    print("Number of tetra  = ", num_tetra)
    
    mesh = dolfin.Mesh()
    editor = dolfin.MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)  # top. and geom. dimension are both 2
    editor.init_vertices(num_pts)  # number of vertices
    editor.init_cells(num_tetra)     # number of cells
    
    for p in range(0, num_pts):
        pt = ugrid.GetPoints().GetPoint(p)
        editor.add_vertex(p, dolfin.Point(pt[0], pt[1], pt[2]))
    
    cnt =  0
    for p in range(0, num_cells):
        pts = vtk.vtkIdList()
        ugrid.GetCellPoints(p, pts)
        if(pts.GetNumberOfIds() == 4):
            editor.add_cell(cnt, [pts.GetId(0),  pts.GetId(1), pts.GetId(2), pts.GetId(3)])
            cnt = cnt + 1
    
    editor.close()
    return mesh


def convertXMLMeshToUGrid(mesh, p_region=None):
    connectivity =  mesh.cells()
    coords =  mesh.coordinates()
    points = vtk.vtkPoints()

    for coord in coords:
        points.InsertNextPoint(coord[0], coord[1], coord[2])
    
    part_id = vtk.vtkIntArray()
    part_id.SetName("part_id")

    if(p_region):
        V = dolfin.FunctionSpace(mesh, "DG", 0)
        dm = V.dofmap()
        for cell in dolfin.cells(mesh):
            matid = p_region[cell]
            part_id.InsertNextValue(matid)
    
    cellarray = vtk.vtkCellArray()
    for cell in connectivity:
        if('tetrahedron' in str(mesh.type().cell_type())):
            vtkcell = vtk.vtkTetra()
            vtkcell.GetPointIds().SetId(0, cell[0])
            vtkcell.GetPointIds().SetId(1, cell[1])
            vtkcell.GetPointIds().SetId(2, cell[2])
            vtkcell.GetPointIds().SetId(3, cell[3])
    
        elif('interval' in str(mesh.type().cell_type())):
            vtkcell = vtk.vtkLine()
            vtkcell.GetPointIds().SetId(0, cell[0])
            vtkcell.GetPointIds().SetId(1, cell[1])
            
        else:
            print("element type not supported", flush=True)
            exit()

        cellarray.InsertNextCell(vtkcell)
    
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    ugrid.SetCells(vtkcell.GetCellType(), cellarray)
    
    if(p_region):
        ugrid.GetCellData().AddArray(part_id)
    
    return ugrid


def convertUGridtoPdata(ugrid):
    geometry = vtk.vtkGeometryFilter()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        geometry.SetInputData(ugrid)
    else:
        geometry.SetInput(ugrid)

    geometry.Update()
    return geometry.GetOutput()


def getcentroid(domain, verbose=True):
    if (verbose): print('*** Get Centroid ***')
    bds = domain.GetBounds()
    return [0.5*(bds[0]+bds[1]), 0.5*(bds[2]+bds[3]), 0.5*(bds[4]+bds[5])]


def clipheart(domain, C, N, isinsideout, verbose=True):
    if (verbose): print('*** Slice Heart ***')

    plane = vtk.vtkPlane()
    plane.SetOrigin(C)
    plane.SetNormal(N)

    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(plane)
    
    if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
        clipper.SetInput(domain)
    else:
        clipper.SetInputData(domain)
    
    clipper.SetInsideOut(isinsideout)
    clipper.Update()
    return clipper.GetOutput()


def splitDomainBetweenEndoAndEpi(domain, verbose=True):

    if (verbose): print('*** splitDomainBetweenEndoAndEpi ***')
    
    connectivity0 = vtk.vtkConnectivityFilter()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        connectivity0.SetInputData(domain)
    else:
        connectivity0.SetInput(domain)
    
    connectivity0.SetExtractionModeToSpecifiedRegions()
    connectivity0.AddSpecifiedRegion(0)
    connectivity0.Update()
    ugrid0_temp = connectivity0.GetOutput()
    
    geom0 = vtk.vtkGeometryFilter()
    
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        geom0.SetInputData(ugrid0_temp)
    else:
        geom0.SetInput(ugrid0_temp)
    geom0.Update()
    pdata0_temp = geom0.GetOutput()
    
    tfilter0 = vtk.vtkTriangleFilter()
    
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        tfilter0.SetInputData(pdata0_temp)
    else:
        tfilter0.SetInput(pdata0_temp)
    tfilter0.Update()
    
    connectivity1 = vtk.vtkConnectivityFilter()
    connectivity1.SetExtractionModeToSpecifiedRegions()
    connectivity1.AddSpecifiedRegion(1)
    
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        connectivity1.SetInputData(domain)
    else:
        connectivity1.SetInput(domain)
    connectivity1.Update()
    
    ugrid1_temp = connectivity1.GetOutput()
    geom1 = vtk.vtkGeometryFilter()
    
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        geom1.SetInputData(ugrid1_temp)
    else:
        geom1.SetInput(ugrid1_temp)
    geom1.Update()
    pdata1_temp = geom1.GetOutput()
    
    tfilter1 = vtk.vtkTriangleFilter()
    
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        tfilter1.SetInputData(pdata1_temp)
    else:
        tfilter1.SetInput(pdata1_temp)
    tfilter1.Update()
    
    pdata1 = tfilter1.GetOutput()
    pdata0 = tfilter0.GetOutput()    
    
    p0bd = pdata0.GetBounds()
    p1bd = pdata1.GetBounds()
    
    if (abs(p1bd[0] - p1bd[1]) < abs(p0bd[0] - p0bd[1])):
        pdata_epi = pdata0
        pdata_endo = pdata1
    else:
        pdata_epi = pdata1
        pdata_endo = pdata0
    
    return pdata_epi, pdata_endo


def CreateVertexFromPoint(ugrid):
    vertices = vtk.vtkCellArray()
    for p in range(ugrid.GetNumberOfPoints()):
        vert = vtk.vtkVertex()
        vert.GetPointIds().SetId(0, p)
        vertices.InsertNextCell(vert)
    ugrid.SetCells(1, vertices)


def createFloatArray(name, nb_components=1, nb_tuples=0, verbose=True):
    farray = vtk.vtkFloatArray()
    farray.SetName(name)
    farray.SetNumberOfComponents(nb_components)
    farray.SetNumberOfTuples(nb_tuples)
    return farray


def getABPointsFromBoundsAndCenter(ugrid_wall, verbose=True):

    if (verbose): print('*** getABPointsFromBoundsAndCenter ***')
    
    # Note that it is assumed here that the ventricle is vertical in the
    # global coordinates system
    bounds = ugrid_wall.GetBounds()
    center = ugrid_wall.GetCenter()

    point_A = [center[0], center[1], bounds[4]]
    point_B = [center[0], center[1], bounds[5]]

    points_AB = vtk.vtkPoints()
    points_AB.InsertNextPoint(point_A)
    points_AB.InsertNextPoint(point_B)

    return points_AB


def getCellCenters(mesh, verbose=1):
    print("*** getCellCenters ***")
    if (verbose): print("*** getCellCenters ***")

    filter_cell_centers = vtk.vtkCellCenters()

    if(vtk.vtkVersion().GetVTKMajorVersion() > 5):
        filter_cell_centers.SetInputData(mesh)
    else:
        filter_cell_centers.SetInput(mesh)
    filter_cell_centers.Update()
    return filter_cell_centers.GetOutput()


def getPDataNormals(pdata, flip=False, verbose=True):
    if (verbose): print('*** getPDataNormals ***')

    poly_data_normals = vtk.vtkPolyDataNormals()
    poly_data_normals.ComputePointNormalsOff()
    poly_data_normals.ComputeCellNormalsOn()

    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        poly_data_normals.SetInputData(pdata)
    else:
        poly_data_normals.SetInput(pdata)
    
    if (flip): poly_data_normals.FlipNormalsOn()
    else:      poly_data_normals.FlipNormalsOff()
    poly_data_normals.Update()

    return poly_data_normals.GetOutput()


def writePData(pdata, pdata_file_name, verbose=True):
    if (verbose): print('*** writePData ***')

    pdata_writer = vtk.vtkPolyDataWriter()
    pdata_writer.SetFileName(pdata_file_name)
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        pdata_writer.SetInputData(pdata)
    else:
        pdata_writer.SetInput(pdata)
    pdata_writer.Update()
    pdata_writer.Write()


def addLocalProlateSpheroidalDirections(
    ugrid_wall,
    pdata_end,
    pdata_epi,
    type_of_support="cell",
	epiflip=False,
	endoflip=False,
	apexflip=False,
    points_AB=None,
	eCCname="eCC",
	eLLname="eLL",
    eRRname="eRR",
    verbose=True):

    if (verbose): print('*** addLocalProlateSpheroidalDirections ***')

    if (points_AB == None):
        points_AB = getABPointsFromBoundsAndCenter(pdata_epi, verbose)
    assert (points_AB.GetNumberOfPoints() == 2), "points_AB must have two points. Aborting."
    point_A = np.array([0.]*3)
    point_B = np.array([0.]*3)
    points_AB.GetPoint(0, point_A)
    points_AB.GetPoint(1, point_B)
    if(apexflip):
        eL  = point_A - point_B
    else:
        eL  = point_B - point_A
    eL /= np.linalg.norm(eL)

    if (type_of_support == "cell"):
        pdata_cell_centers = getCellCenters(ugrid_wall)
        
    if (verbose): print("Computing cell normals...")

    if(epiflip):
        pdata_epi = getPDataNormals(pdata_epi, flip=1)
    else:
        pdata_epi = getPDataNormals(pdata_epi, flip=0)

    if(endoflip):
        pdata_end = getPDataNormals(pdata_end, flip=1)
    else:
        pdata_end = getPDataNormals(pdata_end, flip=0)
        
    if (verbose): print("Computing surface bounds...")
    
    bounds_end = pdata_end.GetBounds()
    bounds_epi = pdata_epi.GetBounds()
    z_min_end = bounds_end[4]
    z_min_epi = bounds_epi[4]
    z_max_end = bounds_end[5]
    z_max_epi = bounds_epi[5]
    L_end = z_max_end-z_min_end
    L_epi = z_max_epi-z_min_epi

    if (verbose): print("Initializing cell locators...")

    cell_locator_end = vtk.vtkCellLocator()
    cell_locator_end.SetDataSet(pdata_end)
    cell_locator_end.Update()

    cell_locator_epi = vtk.vtkCellLocator()
    cell_locator_epi.SetDataSet(pdata_epi)
    cell_locator_epi.Update()

    closest_point_end = [0.]*3
    closest_point_epi = [0.]*3
    generic_cell = vtk.vtkGenericCell()
    cellId_end = vtk.mutable(0)
    cellId_epi = vtk.mutable(0)
    subId = vtk.mutable(0)
    dist_end = vtk.mutable(0.)
    dist_epi = vtk.mutable(0.)

    if (verbose): print("Computing local prolate spheroidal directions...")

    if (type_of_support == "cell"):
        nb_cells = ugrid_wall.GetNumberOfCells()
    elif (type_of_support == "point"):
        nb_cells = ugrid_wall.GetNumberOfPoints()

    farray_norm_dist_end = createFloatArray("norm_dist_end", 1, nb_cells)
    farray_norm_dist_epi = createFloatArray("norm_dist_epi", 1, nb_cells)
    
    farray_norm_z_end = createFloatArray("norm_z_end", 1, nb_cells)
    farray_norm_z_epi = createFloatArray("norm_z_epi", 1, nb_cells)

    farray_eRR = createFloatArray(eRRname, 3, nb_cells)
    farray_eCC = createFloatArray(eCCname, 3, nb_cells)
    farray_eLL = createFloatArray(eLLname, 3, nb_cells)

    for num_cell in range(nb_cells):
        if (type_of_support == "cell"):
            cell_center = np.array(pdata_cell_centers.GetPoints().GetPoint(num_cell))
        elif (type_of_support == "point"):
            cell_center = np.array(ugrid_wall.GetPoints().GetPoint(num_cell))
        cell_locator_end.FindClosestPoint(cell_center, closest_point_end, generic_cell, cellId_end, subId, dist_end)
        cell_locator_epi.FindClosestPoint(cell_center, closest_point_epi, generic_cell, cellId_epi, subId, dist_epi)

        norm_dist_end = dist_end/(dist_end+dist_epi)
        norm_dist_epi = dist_epi/(dist_end+dist_epi)
        farray_norm_dist_end.InsertTuple(num_cell, [norm_dist_end])
        farray_norm_dist_epi.InsertTuple(num_cell, [norm_dist_epi])

        norm_z_end = (closest_point_end[2]-z_min_end)/L_end
        norm_z_epi = (closest_point_epi[2]-z_min_epi)/L_epi
        farray_norm_z_end.InsertTuple(num_cell, [norm_z_end])
        farray_norm_z_epi.InsertTuple(num_cell, [norm_z_epi])

        normal_end = np.reshape(pdata_end.GetCellData().GetNormals().GetTuple(cellId_end), (3))
        normal_epi = np.reshape(pdata_epi.GetCellData().GetNormals().GetTuple(cellId_epi), (3))
        eRR  = -1*(1.-norm_dist_end) * normal_end + (1.-norm_dist_epi) * normal_epi
        eRR /= np.linalg.norm(eRR)
        eCC  = np.cross(eL, eRR)
        eCC /= np.linalg.norm(eCC)
        eLL  = np.cross(eRR, eCC)
        farray_eRR.InsertTuple(num_cell, eRR)
        farray_eCC.InsertTuple(num_cell, eCC)
        farray_eLL.InsertTuple(num_cell, eLL)

    if (verbose): print("Filling mesh...")

    if (type_of_support == "cell"):
        ugrid_wall.GetCellData().AddArray(farray_norm_dist_end)
        ugrid_wall.GetCellData().AddArray(farray_norm_dist_epi)
        ugrid_wall.GetCellData().AddArray(farray_norm_z_end)
        ugrid_wall.GetCellData().AddArray(farray_norm_z_epi)
        ugrid_wall.GetCellData().AddArray(farray_eRR)
        ugrid_wall.GetCellData().AddArray(farray_eCC)
        ugrid_wall.GetCellData().AddArray(farray_eLL)
    elif (type_of_support == "point"):
        ugrid_wall.GetPointData().AddArray(farray_norm_dist_end)
        ugrid_wall.GetPointData().AddArray(farray_norm_dist_epi)
        ugrid_wall.GetPointData().AddArray(farray_norm_z_end)
        ugrid_wall.GetPointData().AddArray(farray_norm_z_epi)
        ugrid_wall.GetPointData().AddArray(farray_eRR)
        ugrid_wall.GetPointData().AddArray(farray_eCC)
        ugrid_wall.GetPointData().AddArray(farray_eLL)


def addLocalFiberOrientation(
    ugrid_wall,
    fiber_angle_end,
    fiber_angle_epi,
    points_AB=None,
    verbose=True):

    if (verbose): print('*** addLocalFiberOrientation ***')

    if (points_AB == None):
        points_AB = getABPointsFromBoundsAndCenter(ugrid_wall, verbose)
    
    assert (points_AB.GetNumberOfPoints() >= 2), "\"points_AB\" must have at least two points. Aborting."
    
    point_A = np.array([0.]*3)
    point_B = np.array([0.]*3)
    points_AB.GetPoint(                              0, point_A)
    points_AB.GetPoint(points_AB.GetNumberOfPoints()-1, point_B)
    eL  = point_B - point_A
    eL /= np.linalg.norm(eL)

    if (verbose): print("Computing local fiber orientation...")

    farray_norm_dist_end = ugrid_wall.GetCellData().GetArray("norm_dist_end")
    farray_norm_dist_epi = ugrid_wall.GetCellData().GetArray("norm_dist_epi")
    farray_eRR = ugrid_wall.GetCellData().GetArray("eRR")
    farray_eCC = ugrid_wall.GetCellData().GetArray("eCC")
    farray_eLL = ugrid_wall.GetCellData().GetArray("eLL")

    nb_cells = ugrid_wall.GetNumberOfCells()

    farray_fiber_angle = createFloatArray("fiber_angle", 1, nb_cells)

    farray_eF = createFloatArray("fiber vectors", 3, nb_cells)
    farray_eS = createFloatArray("sheet vectors", 3, nb_cells)
    farray_eN = createFloatArray("sheet normal vectors", 3, nb_cells)
       
    for num_cell in range(nb_cells):
        norm_dist_end = farray_norm_dist_end.GetTuple(num_cell)[0]
        norm_dist_epi = farray_norm_dist_epi.GetTuple(num_cell)[0]

        fiber_angle_in_degrees = (1.-norm_dist_end) * fiber_angle_end + (1.-norm_dist_epi) * fiber_angle_epi

        farray_fiber_angle.InsertTuple(num_cell, [fiber_angle_in_degrees])

        eRR = np.array(farray_eRR.GetTuple(num_cell))
        eCC = np.array(farray_eCC.GetTuple(num_cell))
        eLL = np.array(farray_eLL.GetTuple(num_cell))

        fiber_angle_in_radians = math.pi*fiber_angle_in_degrees/180
        eF = math.cos(fiber_angle_in_radians) * eCC + math.sin(fiber_angle_in_radians) * eLL
        eS = eRR
        eN = np.cross(eF, eS)
        farray_eF.InsertTuple(num_cell, eF)
        farray_eS.InsertTuple(num_cell, eS)
        farray_eN.InsertTuple(num_cell, eN)

    if (verbose): print("Filling mesh...")

    ugrid_wall.GetCellData().AddArray(farray_fiber_angle)
    ugrid_wall.GetCellData().AddArray(farray_eF)
    ugrid_wall.GetCellData().AddArray(farray_eS)
    ugrid_wall.GetCellData().AddArray(farray_eN)


def extractFeNiCsBiVFacet(ugrid, geometry="BiV", tol=1e-2):
    geom = vtk.vtkGeometryFilter()
    if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
        geom.SetInput(ugrid)
    else:
        geom.SetInputData(ugrid)
    geom.Update()
    surf = geom.GetOutput()

    bc_pts_locator = []
    bc_pts = []
    bc_pts_range = []
    bc_pts_map = []

    # Extract Surface Normal
    normal = vtk.vtkPolyDataNormals()
    if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
        normal.SetInput(surf)
    else:
        normal.SetInputData(surf)

    normal.ComputeCellNormalsOn()
    normal.Update()
    surf_w_norm = normal.GetOutput()

    zmax = surf_w_norm.GetBounds()[5]

    surf_w_norm.BuildLinks()
    idlist = vtk.vtkIdList()
    basecellidlist = vtk.vtkIdTypeArray()
    basesurf = vtk.vtkPolyData()
    for p in range(0, surf_w_norm.GetNumberOfCells()):
        zvec = surf_w_norm.GetCellData().GetNormals().GetTuple3(p)[2]
        surf_w_norm.GetCellPoints(p, idlist)
        zpos = surf_w_norm.GetPoints().GetPoint(idlist.GetId(0))[2]
        if((abs(zvec - 1.0) < tol or abs(zvec + 1.0) < tol) and (abs(zmax - zpos) < tol)):
            surf_w_norm.DeleteCell(p)
            basecellidlist.InsertNextValue(p)
            
    basesurf = extractCellFromPData(basecellidlist, surf)
    baseptlocator = vtk.vtkPointLocator()
    baseptlocator.SetDataSet(basesurf)
    baseptlocator.BuildLocator()

    surf_w_norm.RemoveDeletedCells()
    cleanpdata = vtk.vtkCleanPolyData()

    if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
        cleanpdata.SetInput(surf_w_norm)
    else:
        cleanpdata.SetInputData(surf_w_norm)
    cleanpdata.Update()

    connfilter = vtk.vtkPolyDataConnectivityFilter()

    if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
        connfilter.SetInput(cleanpdata.GetOutput())
    else:
        connfilter.SetInputData(cleanpdata.GetOutput())
    connfilter.Update()

    print("Total_num_points = ",  cleanpdata.GetOutput().GetNumberOfPoints())
    tpt = 0

    if(geometry=="BiV"):
        nsurf = 3
    else:
        nsurf = 2

    for p in range(0,nsurf):
        pts = vtk.vtkPolyData()
        connfilter.SetExtractionModeToSpecifiedRegions()
        [connfilter.DeleteSpecifiedRegion(k) for k in range(0,nsurf)]
        connfilter.AddSpecifiedRegion(p)
        connfilter.ScalarConnectivityOff()
        connfilter.FullScalarConnectivityOff()
        connfilter.Update()

        cleanpdata2 = vtk.vtkCleanPolyData()
        if(vtk.vtkVersion().GetVTKMajorVersion() < 6):
            cleanpdata2.SetInput(connfilter.GetOutput())
        else:
            cleanpdata2.SetInputData(connfilter.GetOutput())
        cleanpdata2.Update()

        pts.DeepCopy(cleanpdata2.GetOutput())

        tpt = tpt + cleanpdata2.GetOutput().GetNumberOfPoints()

        ptlocator = vtk.vtkPointLocator()
        ptlocator.SetDataSet(pts)
        ptlocator.BuildLocator()

        bc_pts_locator.append(ptlocator)
        bc_pts.append(pts)
        bc_pts_range.append([abs(pts.GetBounds()[k+1] - pts.GetBounds()[k]) for k in range(0, 6, 2)])

    print("Total_num_points = ",  tpt)

    Epiid = np.argmax(np.array([max(pts) for pts in bc_pts_range]))
    maxzrank =  np.array([pts[2] for pts in bc_pts_range]).argsort()

    if(geometry=="BiV"):
        LVid = maxzrank[1] 
        RVid = 3 - (LVid + Epiid)
        bc_pts_map = [4, 4, 4, 4]
        bc_pts_map[Epiid] = 1; bc_pts_map[LVid] = 2; bc_pts_map[RVid] = 3
        baseid  = 3
    else:
        LVid = maxzrank[0]
        bc_pts_map = [4, 4, 4]
        bc_pts_map[Epiid] = 1; bc_pts_map[LVid] = 2
        baseid  = 2

    bc_pts_locator.append(baseptlocator)
    bc_pts.append(basesurf)

    dolfin_mesh = convertUGridToXMLMesh(ugrid)
    dolfin_facets = dolfin.MeshFunction('size_t', dolfin_mesh, 2)
    dolfin_facets.set_all(0)

    for facet in dolfin.SubsetIterator(dolfin_facets, 0):
        for locator in range(0,nsurf+1):
            cnt = 0
            for p in range(0,3):
                v0 =  dolfin.Vertex(dolfin_mesh, facet.entities(0)[p]).x(0)
                v1 =  dolfin.Vertex(dolfin_mesh, facet.entities(0)[p]).x(1)
                v2 =  dolfin.Vertex(dolfin_mesh, facet.entities(0)[p]).x(2)
                ptid = bc_pts_locator[locator].FindClosestPoint(v0, v1, v2)
                x0 =  bc_pts[locator].GetPoints().GetPoint(ptid)
                dist = vtk.vtkMath.Distance2BetweenPoints([v0,v1,v2], x0)
                if(dist < 1e-5):
                    cnt = cnt + 1
            if(cnt == 3):
                dolfin_facets[facet] = bc_pts_map[locator]
                    
    dolfin_edges = dolfin.MeshFunction('size_t', dolfin_mesh, 1)
    dolfin_edges.set_all(0)

    epilocator = Epiid
    lvendolocator = LVid

    for edge in dolfin.SubsetIterator(dolfin_edges, 0):
        cnt_epi = 0; cnt_lvendo = 0
        for p in range(0,2):
            v0 =  dolfin.Vertex(dolfin_mesh, edge.entities(0)[p]).x(0)
            v1 =  dolfin.Vertex(dolfin_mesh, edge.entities(0)[p]).x(1)
            v2 =  dolfin.Vertex(dolfin_mesh, edge.entities(0)[p]).x(2)

            epiptid = bc_pts_locator[epilocator].FindClosestPoint(v0, v1, v2)
            epix0 =  bc_pts[epilocator].GetPoints().GetPoint(epiptid)
            epidist = vtk.vtkMath.Distance2BetweenPoints([v0,v1,v2], epix0)

            topptid = bc_pts_locator[baseid].FindClosestPoint(v0, v1, v2)
            topx0 =  bc_pts[baseid].GetPoints().GetPoint(topptid)
            topdist = vtk.vtkMath.Distance2BetweenPoints([v0,v1,v2], topx0)

            lvendoptid = bc_pts_locator[lvendolocator].FindClosestPoint(v0, v1, v2)
            lvendox0 =  bc_pts[lvendolocator].GetPoints().GetPoint(lvendoptid)
            lvendodist = vtk.vtkMath.Distance2BetweenPoints([v0,v1,v2], lvendox0)

            if(topdist < 1e-5 and epidist < 1e-5):
                cnt_epi = cnt_epi + 1

            if(topdist < 1e-5 and lvendodist < 1e-5):
                cnt_lvendo = cnt_lvendo + 1

            if(cnt_epi == 2):
                dolfin_edges[edge] = 1

            if(cnt_lvendo == 2):
                dolfin_edges[edge] = 2

    return dolfin_mesh, dolfin_facets, dolfin_edges


def addLVfiber(
    mesh, V, casename, endo_angle, epi_angle,  casedir, isepiflip,
    isendoflip, isapexflip=False, ztol=0.05):
    fiberV = dolfin.Function(V)
    sheetV = dolfin.Function(V)
    sheetnormV = dolfin.Function(V)
    cV = dolfin.Function(V)
    lV = dolfin.Function(V)
    rV = dolfin.Function(V)
    
    ugrid = convertXMLMeshToUGrid(mesh)
    pdata = convertUGridtoPdata(ugrid)
    C = getcentroid(pdata)
    if(isapexflip):
        ztop = pdata.GetBounds()[4]
        C = [C[0], C[1], ztop+ztol]
        clippedheart = clipheart(pdata, C, [0,0,-1], True)
    else:
        ztop = pdata.GetBounds()[5]
        C = [C[0], C[1], ztop-ztol]
        clippedheart = clipheart(pdata, C, [0,0,1], True)
    epi, endo= splitDomainBetweenEndoAndEpi(clippedheart)
    
    cleanepipdata = vtk.vtkCleanPolyData()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
            cleanepipdata.SetInputData(epi)
    else:
            cleanepipdata.SetInput(epi)
    cleanepipdata.Update()
    pdata_epi = cleanepipdata.GetOutput()
    
    cleanendopdata = vtk.vtkCleanPolyData()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
            cleanendopdata.SetInputData(endo)
    else:
            cleanendopdata.SetInput(endo)
    cleanendopdata.Update()
    pdata_endo = cleanendopdata.GetOutput()
    
    L_epi = pdata_epi.GetBounds()[5]  -  pdata_epi.GetBounds()[4]
    L_endo = pdata_endo.GetBounds()[5] - pdata_endo.GetBounds()[4]
    
    if(L_endo > L_epi):
        temp = pdata_epi
        pdata_epi = pdata_endo
        pdata_endo = temp
        
    # Quad points
    gdim = mesh.geometry().dim()
    xdofmap = V.sub(0).dofmap().dofs()
    ydofmap = V.sub(1).dofmap().dofs()
    zdofmap = V.sub(2).dofmap().dofs()
    
    if(dolfin.__version__ != '1.6.0'):
            xq = V.tabulate_dof_coordinates().reshape((-1, gdim))
            xq0 = xq[xdofmap]  
    else:
            xq = V.dofmap().tabulate_all_coordinates(mesh).reshape((-1, gdim))
            xq0 = xq[xdofmap]  
    
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    ugrid = vtk.vtkUnstructuredGrid()
    
    cnt = 0
    for pt in xq0:
        points.InsertNextPoint([pt[0], pt[1], pt[2]])
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, cnt)
        vertices.InsertNextCell(vertex)
        cnt += 1
    
    ugrid.SetPoints(points)
    ugrid.SetCells(0, vertices)
    
    CreateVertexFromPoint(ugrid)
    addLocalProlateSpheroidalDirections(
        ugrid, pdata_endo, pdata_epi, type_of_support="cell",
        epiflip=isepiflip, endoflip=isendoflip, apexflip=isapexflip)
    addLocalFiberOrientation(ugrid, endo_angle, epi_angle)
    
    fiber_vector =  ugrid.GetCellData().GetArray("fiber vectors")
    sheet_vector =  ugrid.GetCellData().GetArray("sheet vectors")
    sheetnorm_vector =  ugrid.GetCellData().GetArray("sheet normal vectors")
    
    eCC_vector =  ugrid.GetCellData().GetArray("eCC")
    eLL_vector =  ugrid.GetCellData().GetArray("eLL")
    eRR_vector =  ugrid.GetCellData().GetArray("eRR")
    
    cnt = 0
    for pt in xq0:
        fvec = fiber_vector.GetTuple(cnt)
        svec = sheet_vector.GetTuple(cnt)
        nvec = sheetnorm_vector.GetTuple(cnt)

        cvec = eCC_vector.GetTuple(cnt)
        lvec = eLL_vector.GetTuple(cnt)
        rvec = eRR_vector.GetTuple(cnt)

        fvecnorm = math.sqrt(fvec[0]**2 + fvec[1]**2 + fvec[2]**2)
        svecnorm = math.sqrt(svec[0]**2 + svec[1]**2 + svec[2]**2)
        nvecnorm = math.sqrt(nvec[0]**2 + nvec[1]**2 + nvec[2]**2)

        if(abs(fvecnorm - 1.0) > 1e-7 or  abs(svecnorm - 1.0) > 1e-6 or abs(nvecnorm - 1.0) > 1e-7):
            print(fvecnorm)
            print(svecnorm)
            print(nvecnorm)

        fiberV.vector()[xdofmap[cnt]] = fvec[0]; fiberV.vector()[ydofmap[cnt]] = fvec[1]; fiberV.vector()[zdofmap[cnt]] = fvec[2];
        sheetV.vector()[xdofmap[cnt]] = svec[0]; sheetV.vector()[ydofmap[cnt]] = svec[1]; sheetV.vector()[zdofmap[cnt]] = svec[2];
        sheetnormV.vector()[xdofmap[cnt]] = nvec[0]; sheetnormV.vector()[ydofmap[cnt]] = nvec[1]; sheetnormV.vector()[zdofmap[cnt]] = nvec[2];

        cV.vector()[xdofmap[cnt]] = cvec[0];  cV.vector()[ydofmap[cnt]] = cvec[1]; cV.vector()[zdofmap[cnt]] = cvec[2]; 
        lV.vector()[xdofmap[cnt]] = lvec[0];  lV.vector()[ydofmap[cnt]] = lvec[1]; lV.vector()[zdofmap[cnt]] = lvec[2]; 
        rV.vector()[xdofmap[cnt]] = rvec[0];  rV.vector()[ydofmap[cnt]] = rvec[1]; rV.vector()[zdofmap[cnt]] = rvec[2]; 
        cnt += 1
    
    return fiberV, sheetV, sheetnormV, cV, lV, rV
