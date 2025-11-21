import numpy as np
import pyvista as pv
from pathlib import Path

COARSE = '/mnt/home/naghavis/Documents/Research/DeepCardioSim/deepcardio/LVmean/LV_mean.vtk'
RR1_FILE = '/mnt/home/naghavis/Documents/Research/DeepCardioSim/deepcardio/LVmean/LV_mean_RR1.vtk'
RR2_FILE = '/mnt/home/naghavis/Documents/Research/DeepCardioSim/deepcardio/LVmean/LV_mean_RR2.vtk'
TEMPLATE = '/mnt/home/naghavis/Documents/Research/DeepCardioSim/cardiac_models/electrophysio/data/interpolated/refine_template_rr1_rr2.npz'

def _cells_to_tets(grid: pv.UnstructuredGrid) -> np.ndarray:
    return grid.cells.reshape(-1, 5)[:, 1:].astype(np.int64)

def _vertex_neighbors_from_tets(n_pts: int, tets: np.ndarray):
    nbrs = [set() for _ in range(n_pts)]
    for a, b, c, d in tets:
        tri = (a, b, c, d)
        for i in range(4):
            v = tri[i]
            for j in range(4):
                if i != j:
                    nbrs[v].add(tri[j])
    return nbrs

def build_refine_templates_from_existing(verbose: bool = True):
    coarse = pv.read(COARSE)
    rr1 = pv.read(RR1_FILE)
    rr2 = pv.read(RR2_FILE)

    N0 = coarse.n_points
    N1 = rr1.n_points
    N2 = rr2.n_points

    tets_rr1 = _cells_to_tets(rr1)
    tets_rr2 = _cells_to_tets(rr2)

    cells_rr1 = tets_rr1.copy()
    cells_rr2 = tets_rr2.copy()

    nbrs_rr1 = _vertex_neighbors_from_tets(N1, tets_rr1)
    edge_pairs_rr1 = np.zeros((N1 - N0, 2), dtype=np.int64)
    for v in range(N0, N1):
        coarse_nbrs = sorted([u for u in nbrs_rr1[v] if u < N0])
        if len(coarse_nbrs) != 2:
            raise RuntimeError(f'RR1 vertex {v} does not have exactly two coarse neighbors: {coarse_nbrs}')
        edge_pairs_rr1[v - N0] = coarse_nbrs

    nbrs_rr2 = _vertex_neighbors_from_tets(N2, tets_rr2)
    edge_pairs_rr2 = np.zeros((N2 - N1, 2), dtype=np.int64)
    for v in range(N1, N2):
        rr1_nbrs = sorted([u for u in nbrs_rr2[v] if u < N1])
        if len(rr1_nbrs) != 2:
            raise RuntimeError(f'RR2 vertex {v} does not have exactly two RR1 neighbors: {rr1_nbrs}')
        edge_pairs_rr2[v - N1] = rr1_nbrs

    Path(TEMPLATE).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        TEMPLATE,
        N0=np.int64(N0), N1=np.int64(N1), N2=np.int64(N2),
        edge_pairs_rr1=edge_pairs_rr1,
        edge_pairs_rr2=edge_pairs_rr2,
        cells_rr1=cells_rr1,
        cells_rr2=cells_rr2
    )
    if verbose:
        print(f'Template saved: N0={N0}, N1={N1}, N2={N2}, '
              f'RR1 new={N1-N0}, RR2 new={N2-N1}')

def _interp_point_data(edge_pairs: np.ndarray, pd: dict) -> dict:
    out = {}
    for k, arr in pd.items():
        a2 = arr if arr.ndim > 1 else arr.reshape(-1, 1)
        if k == 'x_c' and ('x_l' in pd):
            x_l = pd['x_l'].reshape(-1, 1)
            d1 = a2[edge_pairs[:, 0]]
            d2 = a2[edge_pairs[:, 1]]
            is_p1_apex = np.isclose(x_l[edge_pairs[:, 0]], 0.0)
            is_p2_apex = np.isclose(x_l[edge_pairs[:, 1]], 0.0)
            x1, y1 = np.cos(d1), np.sin(d1)
            x2, y2 = np.cos(d2), np.sin(d2)
            x_avg, y_avg = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            mid = np.arctan2(y_avg, x_avg)
            mid[is_p1_apex & ~is_p2_apex] = d2[is_p1_apex & ~is_p2_apex]
            mid[~is_p1_apex & is_p2_apex] = d1[~is_p1_apex & is_p2_apex]
            mid[is_p1_apex & is_p2_apex] = 0.0
            new_data = mid
        else:
            new_data = 0.5 * (a2[edge_pairs[:, 0]] + a2[edge_pairs[:, 1]])
        out[k] = np.vstack([a2, new_data]).squeeze()
    return out

def refine_case_using_templates(points_coarse: np.ndarray,
                                coarse_point_data: dict,
                                level: str = 'RR2',
                                need_quality: bool = False) -> pv.UnstructuredGrid:
    tpl = np.load(TEMPLATE)
    N0 = int(tpl['N0']); N1 = int(tpl['N1']); N2 = int(tpl['N2'])
    assert points_coarse.shape[0] == N0

    ep1 = tpl['edge_pairs_rr1']
    new_rr1 = 0.5 * (points_coarse[ep1[:, 0]] + points_coarse[ep1[:, 1]])
    pts_rr1 = np.vstack([points_coarse, new_rr1])
    pd_rr1 = _interp_point_data(ep1, coarse_point_data)
    cells_rr1 = tpl['cells_rr1'].astype(int)
    mesh_rr1 = pv.UnstructuredGrid({10: cells_rr1}, pts_rr1)
    for k, v in pd_rr1.items():
        mesh_rr1.point_data[k] = v
    if level == 'RR1':
        if need_quality:
            _ = mesh_rr1.cell_quality('scaled_jacobian')
        return mesh_rr1

    ep2 = tpl['edge_pairs_rr2']
    new_rr2 = 0.5 * (pts_rr1[ep2[:, 0]] + pts_rr1[ep2[:, 1]])
    pts_rr2 = np.vstack([pts_rr1, new_rr2])
    pd_rr2 = _interp_point_data(ep2, pd_rr1)
    cells_rr2 = tpl['cells_rr2'].astype(int)
    mesh_rr2 = pv.UnstructuredGrid({10: cells_rr2}, pts_rr2)
    for k, v in pd_rr2.items():
        mesh_rr2.point_data[k] = v
    if need_quality:
        _ = mesh_rr2.cell_quality('scaled_jacobian')
    return mesh_rr2

if __name__ == '__main__':
    case_ID = '2'
    npyfile = f'/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data/npy/case{case_ID}_nplocs1.npy'
    if not Path(npyfile).exists():
        npyfile = f'/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data/npy/case{case_ID}_nplocs2.npy'
        if not Path(npyfile).exists():
            raise ValueError('No default case-specific npy file found for setting up the mesh please pass the input mesh directory and try again.')
    npydata = np.load(npyfile)
    points_coarse = npydata[:, :3]

    coarse_pd = {
        'ploc_bool': npydata[:, 3:4],
        'D_iso': npydata[:, 4:5],
        'ef': npydata[:, 5:8],
        'activation_time': npydata[:, 8:9]
    }
    mesh = refine_case_using_templates(points_coarse, coarse_pd, level='RR2')
    mesh.save(f'./case{case_ID}_RR2.vtk')