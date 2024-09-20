# %%
import argparse
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio
import numpy as np
from dolfinx import fem, default_scalar_type
import ufl
from dolfinx.fem.petsc import LinearProblem
from pathlib import Path

# %%

def main(case_ID):
    np.random.seed(case_ID)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("geometry")

    L1, L2, L3, L4 = 4.5 + np.random.uniform(0.5), 4.5 + np.random.uniform(0.5), 4.5 + np.random.uniform(0.5), 4.5 + np.random.uniform(0.5)
    h, w = 4.5 + np.random.uniform(0.5), 4.5 + np.random.uniform(0.5)

    p1 = gmsh.model.geo.addPoint(-L1/2, -L4/2, 0, meshSize=0.25)
    p2 = gmsh.model.geo.addPoint(0, -h/2, 0, meshSize=0.25)
    p3 = gmsh.model.geo.addPoint(L1/2, -L2/2, 0, meshSize=0.25)
    p4 = gmsh.model.geo.addPoint(w/2, 0, 0, meshSize=0.25)
    p5 = gmsh.model.geo.addPoint(L3/2, L2/2, 0, meshSize=0.25)
    p6 = gmsh.model.geo.addPoint(0, h/2, 0, meshSize=0.25)
    p7 = gmsh.model.geo.addPoint(-L3/2, L4/2, 0, meshSize=0.25)
    p8 = gmsh.model.geo.addPoint(-w/2, 0, 0, meshSize=0.25)

    l1 = gmsh.model.geo.add_bspline([p1, p2, p3])
    l2 = gmsh.model.geo.add_bspline([p3, p4, p5])
    l3 = gmsh.model.geo.add_bspline([p5, p6, p7])
    l4 = gmsh.model.geo.add_bspline([p7, p8, p1])

    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()

    bottom_tag_num, left_tag_num, top_tag_num, \
        right_tag_num, interior_tag_num = 1, 2, 3, 4, 5

    gmsh.model.addPhysicalGroup(1, [l1], bottom_tag_num, name="bottom")
    gmsh.model.addPhysicalGroup(1, [l2], left_tag_num, name="left")
    gmsh.model.addPhysicalGroup(1, [l3], top_tag_num, name="top")
    gmsh.model.addPhysicalGroup(1, [l4], right_tag_num, name="right")
    gmsh.model.addPhysicalGroup(2, [surface], interior_tag_num, name="interior")

    gmsh.model.mesh.generate(2)

    domain, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    gmsh.finalize()

    points = domain.geometry.x
    cells = domain.topology.connectivity(2, 0).array.reshape((-1, 3))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # %%
    V = fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)

    src_term = -(4* x[0] + 2 * x[1])

    # %%
    dofs_bottom_b = fem.locate_dofs_topological(V, fdim, facet_tags.find(bottom_tag_num))
    dofs_left_b = fem.locate_dofs_topological(V, fdim, facet_tags.find(left_tag_num))
    dofs_top_b = fem.locate_dofs_topological(V, fdim, facet_tags.find(top_tag_num))
    dofs_right_b = fem.locate_dofs_topological(V, fdim, facet_tags.find(right_tag_num))

    dofs_drchlt_BC = np.concatenate([dofs_left_b, dofs_right_b])
    dofs_neumnn_BC = np.concatenate([dofs_bottom_b, dofs_top_b])
    # %%
    f_left = fem.Function(V)
    f_left.interpolate(lambda x: np.full(x.shape[1], 1))

    f_right = fem.Function(V)
    f_right.interpolate(lambda x: np.full(x.shape[1], 0))
    
    drchlt_bc_left = fem.dirichletbc(f_left, dofs_left_b)
    drchlt_bc_right = fem.dirichletbc(f_right, dofs_right_b)

    # %%
    n = ufl.FacetNormal(domain)
    g_bottom = fem.Constant(domain, default_scalar_type(0))
    g_top = -fem.Constant(domain, default_scalar_type(1))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # %%
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = src_term * v * ufl.dx \
        - g_bottom * v * ds(bottom_tag_num) - g_top * v * ds(top_tag_num)

    # %%
    problem = LinearProblem(
    a, L, bcs=[drchlt_bc_left, drchlt_bc_right], 
    petsc_options={"ksp_type": "preonly",
                   "pc_type": "lu",
                   "ksp_monitor": None})
    print(f'-------------------- solving for case_ID {case_ID} --------------------')
    uh = problem.solve()

    # %%
    drchlt_bool = np.zeros(points.shape[0])
    neumnn_bool = np.zeros(points.shape[0])
    drchlt_bool[dofs_drchlt_BC] = 1.
    neumnn_bool[dofs_neumnn_BC] = 1.
    data = np.concatenate(
        [points[:, 0].reshape((-1, 1)),
         points[:, 1].reshape((-1, 1)),
         drchlt_bool.reshape((-1, 1)),
         neumnn_bool.reshape((-1, 1)),
         uh.vector.getArray().reshape((-1, 1))], axis=1)

    data_folder = Path("./data")
    data_folder.mkdir(exist_ok=True, parents=True)
    filename = data_folder / f"case_ID_{case_ID}"

    np.save(filename, data)

    data_folder = Path("./data/cells")
    data_folder.mkdir(exist_ok=True, parents=True)
    filename = data_folder / f"case_ID_{case_ID}"

    np.save(filename, cells)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_ID', type=int, required=True)
    args = parser.parse_args()
    main(args.case_ID)