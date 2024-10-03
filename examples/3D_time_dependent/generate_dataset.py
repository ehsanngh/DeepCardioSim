# %%
import argparse
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio
import numpy as np
from dolfinx import fem, default_scalar_type
import ufl
from pathlib import Path
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, \
    create_vector, apply_lifting, set_bc

from petsc4py import PETSc
# %%

def main(case_ID, write_xdmf=False):
    np.random.seed(case_ID)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("geometry")

    L1, L2, L3, L4 = 4. + np.random.uniform(1.), 4. + np.random.uniform(4.), 4. + np.random.uniform(1.), 4. + np.random.uniform(1.)
    h = 6. + np.random.uniform(1.5)
    angle = 0. + np.random.uniform(90.)
    meshsize = 0.25
    
    p1 = gmsh.model.geo.addPoint(-L1/2, -L4/2, 0, meshSize=meshsize)
    p2 = gmsh.model.geo.addPoint(L1/2, -L2/2, 0, meshSize=meshsize)
    p3 = gmsh.model.geo.addPoint(L3/2, L2/2, 0, meshSize=meshsize)
    p4 = gmsh.model.geo.addPoint(-L3/2, L4/2, 0, meshSize=meshsize)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    curve_loop_bottom = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    bottom_surface = gmsh.model.geo.addPlaneSurface([curve_loop_bottom])

    gmsh.model.geo.synchronize()
    bottom_tag_num = gmsh.model.addPhysicalGroup(2, [bottom_surface], name="bottom_surface")

    ov = gmsh.model.geo.twist([(2, bottom_surface)], 0., 0., 0, 0, 0, h, 0, 0, 1,
                            (angle * np.pi / 360.), [15], [], False)

    gmsh.model.geo.synchronize()

    top_tag_num = gmsh.model.addPhysicalGroup(2, [ov[0][1]], name="top_surface")
    volume = gmsh.model.addPhysicalGroup(3, [ov[1][1]], name="Volume")
    lateral1_tag_num = gmsh.model.addPhysicalGroup(2, [ov[2][1]], name="lateral_surface1")
    lateral2_tag_num = gmsh.model.addPhysicalGroup(2, [ov[3][1]], name="lateral_surface2")
    lateral3_tag_num = gmsh.model.addPhysicalGroup(2, [ov[4][1]], name="lateral_surface3")
    lateral4_tag_num = gmsh.model.addPhysicalGroup(2, [ov[5][1]], name="lateral_surface4")

    gmsh.model.mesh.generate(3)

    data_folder = Path("./data/mesh")
    data_folder.mkdir(exist_ok=True, parents=True)
    filename = data_folder / f"case_ID_{case_ID}.msh"
    gmsh.write(str(filename))

    domain, _, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)

    gmsh.finalize()

    points = domain.geometry.x
    cells = domain.topology.connectivity(2, 0).array.reshape((-1, 3))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # %%
    t = 0  # Start time
    T = 5  # End time
    num_steps = 50  # Number of time steps
    dt = (T - t) / num_steps  # Time step size

    # %%
    V2 = fem.functionspace(domain, ("Lagrange", 2))

    dofs_bottom_b = fem.locate_dofs_topological(
        V2, fdim, facet_tags.find(bottom_tag_num))
    dofs_lateral1_b = fem.locate_dofs_topological(
        V2, fdim, facet_tags.find(lateral1_tag_num))
    dofs_lateral2_b = fem.locate_dofs_topological(
        V2, fdim, facet_tags.find(lateral2_tag_num))
    dofs_lateral3_b = fem.locate_dofs_topological(
        V2, fdim, facet_tags.find(lateral3_tag_num))
    dofs_lateral4_b = fem.locate_dofs_topological(
        V2, fdim, facet_tags.find(lateral4_tag_num))
    dofs_top_b = fem.locate_dofs_topological(
        V2, fdim, facet_tags.find(top_tag_num))

    dofs_drchlt_BC = np.concatenate([dofs_bottom_b])
    dofs_neumnn_BC = np.concatenate(
        [dofs_lateral1_b,
         dofs_lateral2_b,
         dofs_lateral3_b,
         dofs_lateral4_b,
         dofs_top_b])


    f_bottom = fem.Function(V2)
    f_bottom.interpolate(lambda x: np.full(x.shape[1], 0))

    drchlt_bc_bottom = fem.dirichletbc(f_bottom, dofs_bottom_b)

    # %%
    def initial_condition(x):
        return (4 * x[0] + 2 * x[1] + 6 * x[0] * x[1] * x[2]) * 10

    u_n = fem.Function(V2)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)
    
    # %%
    class SourceTerm():
        def __init__(self, t):
            self.t = t

        def __call__(self, x):
            return (1 + np.sin(self.t / (0.5 * T) * np.pi)) * \
                (4 * x[0] + 2 * x[1] + 6 * x[0] * x[1] * x[2]) * 10
        
    source_term = SourceTerm(t=t)
    src_term = fem.Function(V2)
    src_term.interpolate(source_term)
    
    g_lateral1 = fem.Constant(domain, default_scalar_type(0))
    g_lateral2 = fem.Constant(domain, default_scalar_type(0))
    g_lateral3 = fem.Constant(domain, default_scalar_type(0))
    g_lateral4 = fem.Constant(domain, default_scalar_type(0))
    g_top = -fem.Constant(domain, default_scalar_type(5))
    
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # %%
    u, v = ufl.TrialFunction(V2), ufl.TestFunction(V2)
    a = dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + u * v * ufl.dx
    L = (u_n + dt * src_term) * v * ufl.dx \
        + dt * (- g_lateral1 * v * ds(lateral1_tag_num) \
                - g_lateral2 * v * ds(lateral2_tag_num) \
                - g_lateral3 * v * ds(lateral3_tag_num) \
                - g_lateral4 * v * ds(lateral4_tag_num) \
                - g_top * v * ds(top_tag_num))
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    # %%
    A = assemble_matrix(bilinear_form, bcs=[drchlt_bc_bottom])
    A.assemble()
    b = create_vector(linear_form)
    uh2 = fem.Function(V2)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU) 
    solver.setTolerances(rtol=1e-12)
    print(f'-------------------- solving for case_ID {case_ID} --------------------')

    V1 = fem.functionspace(domain, ("Lagrange", 1))
    uh1 = fem.Function(V1)
    uh1.interpolate(initial_condition)
    uh1s = np.zeros((points.shape[0], num_steps + 1))
    uh1s[:, 0] = uh1.x.array

    if write_xdmf:
        from dolfinx import io
        results_folder = Path("./data/xdmf") / f"case_ID_{case_ID}"
        results_folder.mkdir(exist_ok=True, parents=True)
        filename = results_folder / f"case_ID_{case_ID}"
        xdmf = io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w")
        xdmf.write_mesh(domain)
        xdmf.write_function(uh1, source_term.t)
    
    for n in range(num_steps):
        # Update Diriclet boundary condition
        source_term.t += dt
        src_term.interpolate(source_term)

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [bilinear_form], [[drchlt_bc_bottom]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [drchlt_bc_bottom])

        # Solve linear problem
        solver.solve(b, uh2.x.petsc_vec)
        print(f"n: {n}, Final residual norm: {solver.getResidualNorm()}")
        uh2.x.scatter_forward()

        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh2.x.array

        uh1.interpolate(uh2)
        uh1s[:, n + 1] = uh1.x.array
        if write_xdmf:
            xdmf.write_function(uh1, source_term.t)
    # %%
    if write_xdmf:
        xdmf.close()

    drchlt_bool = np.zeros(points.shape[0])
    neumnn_bool = np.zeros(points.shape[0])

    dofs_bottom_b = fem.locate_dofs_topological(V1, fdim, facet_tags.find(bottom_tag_num))
    dofs_lateral1_b = fem.locate_dofs_topological(V1, fdim, facet_tags.find(lateral1_tag_num))
    dofs_lateral2_b = fem.locate_dofs_topological(V1, fdim, facet_tags.find(lateral2_tag_num))
    dofs_lateral3_b = fem.locate_dofs_topological(V1, fdim, facet_tags.find(lateral3_tag_num))
    dofs_lateral4_b = fem.locate_dofs_topological(V1, fdim, facet_tags.find(lateral4_tag_num))
    dofs_top_b = fem.locate_dofs_topological(V1, fdim, facet_tags.find(top_tag_num))

    dofs_drchlt_BC = np.concatenate([dofs_bottom_b])
    dofs_neumnn_BC = np.concatenate(
        [dofs_lateral1_b, dofs_lateral2_b, dofs_lateral3_b, dofs_lateral4_b, dofs_top_b])
    drchlt_bool[dofs_drchlt_BC] = 1.
    neumnn_bool[dofs_neumnn_BC] = 1.

    data = np.concatenate(
        [points,
         drchlt_bool.reshape((-1, 1)),
         neumnn_bool.reshape((-1, 1)),
         uh1s
        ], axis=1)

    data_folder = Path("./data/npy")
    data_folder.mkdir(exist_ok=True, parents=True)
    filename = data_folder / f"case_ID_{case_ID}.npy"

    np.save(file=filename, arr=data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_ID', type=int, default=0, required=False)
    parser.add_argument('--write_xdmf', type=bool, default=False, required=False)
    args = parser.parse_args()

    main(args.case_ID, args.write_xdmf)
