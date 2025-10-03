import vtk
import dolfin
import numpy as np
from vtk.util import numpy_support
import time

def checknormal(pdata, pdata_end, pdata_epi, verbose=True):
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** check surface normal ***')

    pdata_centroid = getCellCenters(pdata)

    pdatalocator = vtk.vtkPointLocator()
    pdatalocator.SetDataSet(pdata_centroid)
    pdatalocator.BuildLocator()

    # Check endo
    pdata_end_centroid = getCellCenters(pdata_end)
    testpt = pdata_end_centroid.GetPoints().GetPoint(0)
    closestptid = pdatalocator.FindClosestPoint(testpt[0], testpt[1], testpt[2])
    checkpt = pdata_centroid.GetPoints().GetPoint(closestptid)
    testvec = pdata_end.GetCellData().GetNormals().GetTuple(0)
    vec = pdata.GetCellData().GetNormals().GetTuple(closestptid)

    if(abs(vec[0] - testvec[0]) > 1e-6):
        pdata_end = getPDataNormals(pdata_end, flip=1)

    # Check epi
    pdata_epi_centroid = getCellCenters(pdata_epi)
    testpt = pdata_epi_centroid.GetPoints().GetPoint(0)
    closestptid = pdatalocator.FindClosestPoint(testpt[0], testpt[1], testpt[2])
    checkpt = pdata_centroid.GetPoints().GetPoint(closestptid)
    testvec = pdata_epi.GetCellData().GetNormals().GetTuple(0)
    vec = pdata.GetCellData().GetNormals().GetTuple(closestptid)

    if(abs(vec[0] - testvec[0]) > 1e-6):
        pdata_epi = getPDataNormals(pdata_epi, flip=1)

    return pdata_end, pdata_epi


def readUGrid(mesh_file_name, verbose=True):
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** readUGrid ***')
    
    mesh_reader = vtk.vtkUnstructuredGridReader()
    mesh_reader.SetFileName(mesh_file_name)
    mesh_reader.Update()
    mesh = mesh_reader.GetOutput()
    
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)):
        nb_points = mesh.GetNumberOfPoints()
        print('nb_points =', nb_points)
        nb_cells = mesh.GetNumberOfCells()
        print('nb_cells =', nb_cells)
    
    return mesh


def writeUGrid(ugrid, ugrid_file_name, verbose=True):
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** writeUGrid ***')

    ugrid_writer = vtk.vtkUnstructuredGridWriter()
    ugrid_writer.SetFileName(ugrid_file_name)
    
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        ugrid_writer.SetInputData(ugrid)
    else:
        ugrid_writer.SetInput(ugrid)
    ugrid_writer.Update()
    ugrid_writer.Write()


def writeXMLUGrid(ugrid, ugrid_file_name):
    ugrid_writer = vtk.vtkXMLUnstructuredGridWriter()
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
    if (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0):
        print("Number of points  = ", num_pts)
        print("Number of tetra  = ", num_tetra)
    mesh = dolfin.Mesh(dolfin.MPI.comm_self)
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


def convertXMLMeshToUGrid(mesh, p_region=None, replicate_global=True, comm=None):
    """
    Build a global vtkUnstructuredGrid from a (possibly distributed) FEniCS mesh.
    If replicate_global=True and MPI size > 1, gather on rank 0 then broadcast
    arrays so every rank reconstructs the same global ugrid.
    """
    # --- MPI bits
    try:
        from mpi4py import MPI as pyMPI
        mpi_comm = comm if comm is not None else pyMPI.COMM_WORLD
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()
    except Exception:
        mpi_comm = None
        rank, size = 0, 1
        replicate_global = False

    # --- Local data (IMPORTANT: use global vertex ids)
    coords_local = mesh.coordinates()               # (nloc_verts, 3) local
    cells_local  = mesh.cells()                     # (nloc_cells, 4) local vertex indices
    gvid_local   = mesh.topology().global_indices(0)  # (nloc_verts,) global vertex ids

    # Map local vertex index -> global vertex id
    # cells as global-vertex-id connectivity
    cells_glob = gvid_local[cells_local]           # (nloc_cells, 4)

    # Optional cell part_id aligned with cells_local order
    part_local = None
    if p_region is not None:
        # safest: material id array in the same ordering as cells_local
        try:
            part_local = p_region.array().copy()
        except Exception:
            # fallback: iterate cells; order should match mesh.cells() for legacy dolfin
            part_local = np.array([p_region[cell] for cell in dolfin.cells(mesh)], dtype=np.int32)

    # Pack vertices as (global_id, x, y, z) so we can deduplicate on root
    verts_local = np.empty((gvid_local.size, 4), dtype=np.float64)
    verts_local[:, 0] = gvid_local.astype(np.float64)
    verts_local[:, 1:] = coords_local

    if not replicate_global or size == 1 or mpi_comm is None:
        # ------- Serial / already-global path (your original logic) -------
        points = vtk.vtkPoints()
        for x, y, z in coords_local:
            points.InsertNextPoint(x, y, z)

        cellarray = vtk.vtkCellArray()
        for cell in cells_local:
            tet = vtk.vtkTetra()
            for k in range(4):
                tet.GetPointIds().SetId(k, int(cell[k]))
            cellarray.InsertNextCell(tet)

        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.SetCells(tet.GetCellType(), cellarray)

        if part_local is not None:
            part_id = numpy_support.numpy_to_vtk(part_local.astype(np.int32), deep=True)
            part_id.SetName("part_id")
            ugrid.GetCellData().AddArray(part_id)
        return ugrid

    # ------- Parallel gather on root -------
    all_verts = mpi_comm.gather(verts_local, root=0)
    all_cells = mpi_comm.gather(cells_glob,  root=0)
    all_part  = mpi_comm.gather(part_local,  root=0) if p_region is not None else None

    if rank == 0:
        # Deduplicate vertices by global id
        import numpy as _np
        verts_concat = _np.concatenate(all_verts, axis=0)
        gids = verts_concat[:, 0].astype(_np.int64)
        xyz  = verts_concat[:, 1:].astype(_np.float64)

        # Keep last occurrence on duplicate IDs (ownership ambiguity doesn’t matter if coords match)
        # Build map: global_id -> compressed_id
        unique_gids, first_idx = _np.unique(gids, return_index=True)
        # (use first occurrence; if you prefer last, use return_index on reversed array)
        xyz_unique = xyz[first_idx]

        gid2cid = {int(gid): i for i, gid in enumerate(unique_gids.tolist())}

        # Reindex cells
        cells_concat = _np.concatenate(all_cells, axis=0)
        cells_c = _np.vectorize(lambda g: gid2cid[int(g)], otypes=[_np.int32])(cells_concat)

        # Optional part ids
        part_c = None
        if all_part is not None and any(p is not None for p in all_part):
            part_c = _np.concatenate([p for p in all_part if p is not None], axis=0).astype(_np.int32)
            assert part_c.shape[0] == cells_c.shape[0], "part_id length doesn’t match cells"

        # Build arrays to broadcast
        P = xyz_unique                         # (N,3)
        C = cells_c.astype(_np.int32)          # (M,4)
        A = part_c                              # (M,) or None
    else:
        P = C = A = None

    # ------- Broadcast compact global arrays to everyone -------
    P = mpi_comm.bcast(P, root=0)
    C = mpi_comm.bcast(C, root=0)
    A = mpi_comm.bcast(A, root=0)

    # ------- Rebuild identical VTK ugrid on each rank -------
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(P.shape[0])
    for i in range(P.shape[0]):
        px, py, pz = P[i]
        points.SetPoint(i, float(px), float(py), float(pz))

    cellarray = vtk.vtkCellArray()
    for c in C:
        tet = vtk.vtkTetra()
        tet.GetPointIds().SetId(0, int(c[0]))
        tet.GetPointIds().SetId(1, int(c[1]))
        tet.GetPointIds().SetId(2, int(c[2]))
        tet.GetPointIds().SetId(3, int(c[3]))
        cellarray.InsertNextCell(tet)

    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    ugrid.SetCells(vtk.vtkTetra().GetCellType(), cellarray)

    if A is not None:
        part_id = numpy_support.numpy_to_vtk(A.astype(np.int32), deep=True)
        part_id.SetName("part_id")
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
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** Get Centroid ***')
    bds = domain.GetBounds()
    return [0.5*(bds[0]+bds[1]), 0.5*(bds[2]+bds[3]), 0.5*(bds[4]+bds[5])]


def clipheart(domain, C, N, isinsideout, verbose=True):
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** Slice Heart ***')

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

    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** splitDomainBetweenEndoAndEpi ***')
    
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


def createFloatArray(name, nb_components=1, nb_tuples=0):
    farray = vtk.vtkFloatArray()
    farray.SetName(name)
    farray.SetNumberOfComponents(nb_components)
    farray.SetNumberOfTuples(nb_tuples)
    return farray


def getABPointsFromBoundsAndCenter(ugrid_wall, verbose=True):

    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** getABPointsFromBoundsAndCenter ***')
    
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
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("*** getCellCenters ***")

    filter_cell_centers = vtk.vtkCellCenters()

    if(vtk.vtkVersion().GetVTKMajorVersion() > 5):
        filter_cell_centers.SetInputData(mesh)
    else:
        filter_cell_centers.SetInput(mesh)
    filter_cell_centers.Update()
    return filter_cell_centers.GetOutput()


def getPDataNormals(pdata, flip=False, verbose=True):
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** getPDataNormals ***')

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
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** writePData ***')

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
    ugrid_surf=None,
    verbose=True):

    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** addLocalProlateSpheroidalDirections ***')

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
        
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("Computing cell normals...")

    # Check if normal is oriented in the correction direction
    if(ugrid_surf):
        pdata = getPDataNormals(ugrid_surf, flip=0)
        pdata_epi = getPDataNormals(pdata_epi, flip=0)
        pdata_end = getPDataNormals(pdata_end, flip=0)
        pdata_end, pdata_epi = checknormal(pdata, pdata_end, pdata_epi) 
    
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("Computing surface bounds...")
    
    bounds_end = pdata_end.GetBounds()
    bounds_epi = pdata_epi.GetBounds()
    z_min_end = bounds_end[4]
    z_min_epi = bounds_epi[4]
    z_max_end = bounds_end[5]
    z_max_epi = bounds_epi[5]
    L_end = z_max_end-z_min_end
    L_epi = z_max_epi-z_min_epi

    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("Initializing cell locators...")

    cell_locator_end = vtk.vtkCellLocator()
    cell_locator_end.SetDataSet(pdata_end)
    cell_locator_end.Update()

    cell_locator_epi = vtk.vtkCellLocator()
    cell_locator_epi.SetDataSet(pdata_epi)
    cell_locator_epi.Update()


    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("Computing local prolate spheroidal directions...")


    start = time.time()  # record start time

    code = r"""
    #include <vtkPolyData.h>
    #include <vtkSmartPointer.h>
    #include <vtkCellLocator.h>
    #include <vtkPolyData.h>
    #include <vtkUnstructuredGrid.h>
    #include <vtkPoints.h>
    #include <vtkCellData.h>
    #include <vtkDataArray.h>
    #include <vtkFloatArray.h>
    #include <vtkMath.h>
    #include <array>
    #include <vector>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/stl.h>
    namespace py = pybind11;

    std::tuple<
        py::array_t<float>, py::array_t<float>, py::array_t<float>,  // eRR, eCC, eLL
        py::array_t<float>, py::array_t<float>,                      // norm_dist_end, norm_dist_epi
        py::array_t<float>, py::array_t<float>                       // norm_z_end, norm_z_epi
    >
    iterate(
        const std::string& type_of_support,
        uintptr_t addr_cell_locator_end,
        uintptr_t addr_cell_locator_epi,
        uintptr_t addr_pdata_cell_centers,
        uintptr_t addr_ugrid_wall,
        uintptr_t addr_pdata_end,
        uintptr_t addr_pdata_epi,
        double z_min_end,
        double L_end,
        double z_min_epi,
        double L_epi,
        const std::vector<double>& eL_vec)
    {
        if (eL_vec.size() != 3)
            throw std::runtime_error("eL must have length 3");
        std::array<double,3> eL = { eL_vec[0], eL_vec[1], eL_vec[2] };

        auto cell_locator_end = reinterpret_cast<vtkCellLocator*>(addr_cell_locator_end);
        auto cell_locator_epi = reinterpret_cast<vtkCellLocator*>(addr_cell_locator_epi);
        auto pdata_cell_centers = reinterpret_cast<vtkPolyData*>(addr_pdata_cell_centers);
        auto ugrid_wall = reinterpret_cast<vtkUnstructuredGrid*>(addr_ugrid_wall);
        auto pdata_end = reinterpret_cast<vtkPolyData*>(addr_pdata_end);
        auto pdata_epi = reinterpret_cast<vtkPolyData*>(addr_pdata_epi);

        int nb_cells = ugrid_wall->GetNumberOfCells();

        vtkGenericCell* generic_cell = vtkGenericCell::New();
        vtkIdType cellId_end, cellId_epi;
        int subId;
        double dist_end, dist_epi;

        // Allocate NumPy arrays
        py::array_t<float> eRR_array({nb_cells,3});
        py::array_t<float> eCC_array({nb_cells,3});
        py::array_t<float> eLL_array({nb_cells,3});
        py::array_t<float> norm_dist_end_array({nb_cells});
        py::array_t<float> norm_dist_epi_array({nb_cells});
        py::array_t<float> norm_z_end_array({nb_cells});
        py::array_t<float> norm_z_epi_array({nb_cells});

        auto eRR_ptr = eRR_array.mutable_data();
        auto eCC_ptr = eCC_array.mutable_data();
        auto eLL_ptr = eLL_array.mutable_data();
        auto nd_end_ptr = norm_dist_end_array.mutable_data();
        auto nd_epi_ptr = norm_dist_epi_array.mutable_data();
        auto nz_end_ptr = norm_z_end_array.mutable_data();
        auto nz_epi_ptr = norm_z_epi_array.mutable_data();

        // Loop over cells
        for (int i=0; i<nb_cells; ++i)
        {
            double cell_center[3];
            if (type_of_support == "cell")
                pdata_cell_centers->GetPoints()->GetPoint(i, cell_center);
            else
                ugrid_wall->GetPoints()->GetPoint(i, cell_center);

            double closest_end[3], closest_epi[3];
            cell_locator_end->FindClosestPoint(cell_center, closest_end, generic_cell, cellId_end, subId, dist_end);
            cell_locator_epi->FindClosestPoint(cell_center, closest_epi, generic_cell, cellId_epi, subId, dist_epi);

            // normalized distances
            double nd_end = dist_end / (dist_end + dist_epi);
            double nd_epi = dist_epi / (dist_end + dist_epi);
            nd_end_ptr[i] = static_cast<float>(nd_end);
            nd_epi_ptr[i] = static_cast<float>(nd_epi);

            // normalized Z
            double nz_end = (closest_end[2] - z_min_end)/L_end;
            double nz_epi = (closest_epi[2] - z_min_epi)/L_epi;
            nz_end_ptr[i] = static_cast<float>(nz_end);
            nz_epi_ptr[i] = static_cast<float>(nz_epi);

            // Normals
            double normal_end[3], normal_epi[3];
            pdata_end->GetCellData()->GetNormals()->GetTuple(cellId_end, normal_end);
            pdata_epi->GetCellData()->GetNormals()->GetTuple(cellId_epi, normal_epi);

            // eRR
            double eRR[3];
            for (int j=0; j<3; ++j)
                eRR[j] = -1.0*(1.0-nd_end)*normal_end[j] + (1.0-nd_epi)*normal_epi[j];
            vtkMath::Normalize(eRR);

            // eCC = eL x eRR
            double eCC[3];
            vtkMath::Cross(eL.data(), eRR, eCC);
            vtkMath::Normalize(eCC);

            // eLL = eRR x eCC
            double eLL[3];
            vtkMath::Cross(eRR, eCC, eLL);

            for (int j=0;j<3;++j)
            {
                eRR_ptr[3*i+j] = static_cast<float>(eRR[j]);
                eCC_ptr[3*i+j] = static_cast<float>(eCC[j]);
                eLL_ptr[3*i+j] = static_cast<float>(eLL[j]);
            }
        }

        generic_cell->Delete();

        return std::make_tuple(
            eRR_array, eCC_array, eLL_array,
            norm_dist_end_array, norm_dist_epi_array,
            norm_z_end_array, norm_z_epi_array
        );



    }

    PYBIND11_MODULE(SIGNATURE, m) {
    m.def("iterate", [](
        const std::string& type_of_support,
        uintptr_t addr_cell_locator_end,
        uintptr_t addr_cell_locator_epi,
        uintptr_t addr_pdata_cell_centers,
        uintptr_t addr_ugrid_wall,
        uintptr_t addr_pdata_end,
        uintptr_t addr_pdata_epi,
        double z_min_end,
        double L_end,
        double z_min_epi,
        double L_epi,
        py::object eL_obj) 
    {
        std::vector<double> eL_vec = py::cast<std::vector<double>>(eL_obj);
        return iterate(type_of_support,
                       addr_cell_locator_end, addr_cell_locator_epi,
                       addr_pdata_cell_centers, addr_ugrid_wall,
                       addr_pdata_end, addr_pdata_epi,
                       z_min_end, L_end, z_min_epi, L_epi, eL_vec);
    });
    }
    """

    ext_module = dolfin.compile_cpp_code(code,  
                                         include_dirs=["/usr/local/include/vtk-9.2", "/usr/local/include"],
                                         library_dirs=["/usr/local/lib"],
                                         libraries=[
                                             "vtkCommonCore-9.2",
                                             "vtkCommonDataModel-9.2",
                                             "vtkIOXML-9.2",
                                             "vtkFiltersCore-9.2"
                                         ])

    cell_locator_end_addr = int(cell_locator_end.GetAddressAsString("vtkCellLocator").replace("Addr=", ""), 16)
    cell_locator_epi_addr = int(cell_locator_epi.GetAddressAsString("vtkCellLocator").replace("Addr=", ""), 16)
    pdata_cell_centers_addr = int(pdata_cell_centers.GetAddressAsString("vtkPolyData").replace("Addr=", ""), 16)
    ugrid_wall_addr = int(ugrid_wall.GetAddressAsString("vtkUnstructuredGrid").replace("Addr=", ""), 16)
    pdata_end_addr = int(pdata_end.GetAddressAsString("vtkPolyData").replace("Addr=", ""), 16)
    pdata_epi_addr = int(pdata_epi.GetAddressAsString("vtkPolyData").replace("Addr=", ""), 16)

    eRR, eCC, eLL, nd_end, nd_epi, nz_end, nz_epi = ext_module.iterate(
                                                            type_of_support,
                                                            cell_locator_end_addr,
                                                            cell_locator_epi_addr,
                                                            pdata_cell_centers_addr,
                                                            ugrid_wall_addr,
                                                            pdata_end_addr,
                                                            pdata_epi_addr,
                                                            z_min_end, L_end, z_min_epi, L_epi, eL
                                                           ) 

    eRR = eRR.astype(np.float32)
    eRR_flat = eRR.reshape(-1, 3)
    farray_eRR = numpy_support.numpy_to_vtk(num_array=eRR_flat, deep=True, array_type=vtk.VTK_FLOAT)
    farray_eRR.SetName("eRR")

    eCC = eCC.astype(np.float32)
    eCC_flat = eCC.reshape(-1, 3)
    farray_eCC = numpy_support.numpy_to_vtk(num_array=eCC_flat, deep=True, array_type=vtk.VTK_FLOAT)
    farray_eCC.SetName("eCC")

    eLL = eLL.astype(np.float32)
    eLL_flat = eLL.reshape(-1, 3)
    farray_eLL = numpy_support.numpy_to_vtk(num_array=eLL_flat, deep=True, array_type=vtk.VTK_FLOAT)
    farray_eLL.SetName("eLL")

    eRR = eRR.astype(np.float32)
    eRR_flat = eRR.reshape(-1, 3)
    farray_eRR = numpy_support.numpy_to_vtk(num_array=eRR_flat, deep=True, array_type=vtk.VTK_FLOAT)
    farray_eRR.SetName("eRR")

    farray_norm_dist_end = numpy_support.numpy_to_vtk(num_array=nd_end, deep=True, array_type=vtk.VTK_FLOAT)
    farray_norm_dist_end.SetName("norm_dist_end")

    farray_norm_dist_epi = numpy_support.numpy_to_vtk(num_array=nd_epi, deep=True, array_type=vtk.VTK_FLOAT)
    farray_norm_dist_epi.SetName("norm_dist_epi")

    farray_norm_z_end = numpy_support.numpy_to_vtk(num_array=nz_end, deep=True, array_type=vtk.VTK_FLOAT)
    farray_norm_z_end.SetName("norm_z_end")

    farray_norm_z_epi = numpy_support.numpy_to_vtk(num_array=nz_epi, deep=True, array_type=vtk.VTK_FLOAT)
    farray_norm_z_epi.SetName("norm_z_epi")

    end = time.time()  # record end time
    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print(f"Elapsed time: {end - start:.6f} seconds")


    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("Filling mesh...")

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

    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print('*** addLocalFiberOrientation ***')

    if (points_AB == None):
        points_AB = getABPointsFromBoundsAndCenter(ugrid_wall, verbose)
    
    assert (points_AB.GetNumberOfPoints() >= 2), "\"points_AB\" must have at least two points. Aborting."
    
    point_A = np.array([0.]*3)
    point_B = np.array([0.]*3)
    points_AB.GetPoint(                              0, point_A)
    points_AB.GetPoint(points_AB.GetNumberOfPoints()-1, point_B)
    eL  = point_B - point_A
    eL /= np.linalg.norm(eL)

    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("Computing local fiber orientation...")

    # Extract data from vtkFloatArray objects into NumPy arrays
    norm_dist_end = numpy_support.vtk_to_numpy(ugrid_wall.GetCellData().GetArray("norm_dist_end"))
    norm_dist_epi = numpy_support.vtk_to_numpy(ugrid_wall.GetCellData().GetArray("norm_dist_epi"))
    eRR = numpy_support.vtk_to_numpy(ugrid_wall.GetCellData().GetArray("eRR"))
    eCC = numpy_support.vtk_to_numpy(ugrid_wall.GetCellData().GetArray("eCC"))
    eLL = numpy_support.vtk_to_numpy(ugrid_wall.GetCellData().GetArray("eLL"))

    # Compute fiber angles in degrees
    fiber_angle_in_degrees = (1.0 - norm_dist_end) * fiber_angle_end + (1.0 - norm_dist_epi) * fiber_angle_epi

    # Compute fiber angles in radians
    fiber_angle_in_radians = np.pi * fiber_angle_in_degrees / 180.0

    # Compute eF, eS, and eN vectors
    eF = np.cos(fiber_angle_in_radians)[:, None] * eCC + np.sin(fiber_angle_in_radians)[:, None] * eLL
    eS = eRR
    eN = np.cross(eF, eS)

    if (verbose and (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0)): print("Filling mesh...")

    farray_fiber_angle = numpy_support.numpy_to_vtk(fiber_angle_in_degrees, deep=True)
    farray_fiber_angle.SetName("fiber_angle")

    farray_eF = numpy_support.numpy_to_vtk(eF, deep=True)
    farray_eF.SetName("fiber vectors")

    farray_eS = numpy_support.numpy_to_vtk(eS, deep=True)
    farray_eS.SetName("sheet vectors")

    farray_eN = numpy_support.numpy_to_vtk(eN, deep=True)
    farray_eN.SetName("sheet normal vectors")

    # Add the arrays back to the ugrid_wall
    ugrid_wall.GetCellData().AddArray(farray_fiber_angle)
    ugrid_wall.GetCellData().AddArray(farray_eF)
    ugrid_wall.GetCellData().AddArray(farray_eS)
    ugrid_wall.GetCellData().AddArray(farray_eN)


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
        
    own_beg, own_end = V.dofmap().ownership_range()
    gdim = mesh.geometry().dim()

    xq_all = V.tabulate_dof_coordinates().reshape((-1, gdim))

    owned_global_all = np.array(V.dofmap().dofs(), dtype=np.int64)

    g2l = {int(g): i for i, g in enumerate(owned_global_all)}

    xdofmap = np.array(V.sub(0).dofmap().dofs(), dtype=np.int64)
    ydofmap = np.array(V.sub(1).dofmap().dofs(), dtype=np.int64)
    zdofmap = np.array(V.sub(2).dofmap().dofs(), dtype=np.int64)

    xq0 = xq_all[[g2l[g] for g in xdofmap if g in g2l]]
    
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    ugrid = vtk.vtkUnstructuredGrid()
    
    points.SetData(numpy_support.numpy_to_vtk(xq0, deep=True))
    vertices.InsertNextCell(len(xq0))
    for i in range(len(xq0)):
        vertices.InsertCellPoint(i)
    
    ugrid.SetPoints(points)
    ugrid.SetCells(0, vertices)
    
    CreateVertexFromPoint(ugrid)
    addLocalProlateSpheroidalDirections(
        ugrid, pdata_endo, pdata_epi, type_of_support="cell",
        epiflip=isepiflip, endoflip=isendoflip, apexflip=isapexflip, ugrid_surf=pdata)
    addLocalFiberOrientation(ugrid, endo_angle, epi_angle)
    
    fiber_vector =  ugrid.GetCellData().GetArray("fiber vectors")
    sheet_vector =  ugrid.GetCellData().GetArray("sheet vectors")
    sheetnorm_vector =  ugrid.GetCellData().GetArray("sheet normal vectors")
    
    eCC_vector =  ugrid.GetCellData().GetArray("eCC")
    eLL_vector =  ugrid.GetCellData().GetArray("eLL")
    eRR_vector =  ugrid.GetCellData().GetArray("eRR")
    
    fiber_vector_np = numpy_support.vtk_to_numpy(fiber_vector)
    sheet_vector_np = numpy_support.vtk_to_numpy(sheet_vector)
    sheetnorm_vector_np = numpy_support.vtk_to_numpy(sheetnorm_vector)
    eCC_vector_np = numpy_support.vtk_to_numpy(eCC_vector)
    eLL_vector_np = numpy_support.vtk_to_numpy(eLL_vector)
    eRR_vector_np = numpy_support.vtk_to_numpy(eRR_vector)

    own0 = (xdofmap >= own_beg) & (xdofmap < own_end)
    own1 = (ydofmap >= own_beg) & (ydofmap < own_end)
    own2 = (zdofmap >= own_beg) & (zdofmap < own_end)

    # Local indices (into this rank's local vector) for the owned entries
    lid0 = np.fromiter((g2l[int(g)] for g in xdofmap[own0]), dtype=np.int64)
    lid1 = np.fromiter((g2l[int(g)] for g in ydofmap[own1]), dtype=np.int64)
    lid2 = np.fromiter((g2l[int(g)] for g in zdofmap[own2]), dtype=np.int64)

    # Assign fiber components
    loc = fiberV.vector().get_local()  # local segment (owned only, in DOLFIN 2018.x)
    loc[lid0] = fiber_vector_np[own0, 0]
    loc[lid1] = fiber_vector_np[own1, 1]
    loc[lid2] = fiber_vector_np[own2, 2]
    fiberV.vector().set_local(loc)
    fiberV.vector().apply("insert")

    # Assign sheet components
    loc = sheetV.vector().get_local()
    loc[lid0] = sheet_vector_np[own0, 0]
    loc[lid1] = sheet_vector_np[own1, 1]
    loc[lid2] = sheet_vector_np[own2, 2]
    sheetV.vector().set_local(loc)
    sheetV.vector().apply("insert")

    # Assign sheet-normal components
    loc = sheetnormV.vector().get_local()
    loc[lid0] = sheetnorm_vector_np[own0, 0]
    loc[lid1] = sheetnorm_vector_np[own1, 1]
    loc[lid2] = sheetnorm_vector_np[own2, 2]
    sheetnormV.vector().set_local(loc)
    sheetnormV.vector().apply("insert")

    # Assign orthonormal triad eC/eL
    loc = cV.vector().get_local()
    loc[lid0] = eCC_vector_np[own0, 0]
    loc[lid1] = eCC_vector_np[own1, 1]
    loc[lid2] = eCC_vector_np[own2, 2]
    cV.vector().set_local(loc)
    cV.vector().apply("insert")

    loc = lV.vector().get_local()
    loc[lid0] = eLL_vector_np[own0, 0]
    loc[lid1] = eLL_vector_np[own1, 1]
    loc[lid2] = eLL_vector_np[own2, 2]
    lV.vector().set_local(loc)
    lV.vector().apply("insert")

    loc = rV.vector().get_local()
    loc[lid0] = eRR_vector_np[own0, 0]
    loc[lid1] = eRR_vector_np[own1, 1]
    loc[lid2] = eRR_vector_np[own2, 2]
    rV.vector().set_local(loc)
    rV.vector().apply("insert")

    # if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
    #     writeXMLUGrid(ugrid, "fiber3.vtu")
    
    return fiberV, sheetV, sheetnormV, cV, lV, rV


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

    if (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0): print("Total_num_points = ",  cleanpdata.GetOutput().GetNumberOfPoints())
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

    if (dolfin.MPI.rank(dolfin.MPI.comm_world) == 0): print("Total_num_points = ",  tpt)

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
