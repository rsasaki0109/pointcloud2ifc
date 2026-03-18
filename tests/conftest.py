"""Shared fixtures for pointcloud2ifc tests."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest


@pytest.fixture
def synthetic_room_pcd() -> o3d.geometry.PointCloud:
    """Create a synthetic room point cloud with floor, ceiling, and 4 walls.

    Room dimensions: 5m x 4m x 3m (x, y, z).
    Floor at z=0, ceiling at z=3.
    Walls at x=0, x=5, y=0, y=4.
    Each surface has ~500 points with small noise.
    """
    rng = np.random.default_rng(42)

    def _plane_points(
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
        fixed_axis: int,
        fixed_val: float,
        n: int = 500,
    ) -> np.ndarray:
        pts = np.zeros((n, 3))
        axes = [0, 1, 2]
        axes.remove(fixed_axis)
        ranges = {0: x_range, 1: y_range, 2: z_range}
        for ax in axes:
            lo, hi = ranges[ax]
            pts[:, ax] = rng.uniform(lo, hi, n)
        pts[:, fixed_axis] = fixed_val + rng.normal(0, 0.005, n)
        return pts

    floor = _plane_points((0, 5), (0, 4), (0, 0), fixed_axis=2, fixed_val=0.0)
    ceiling = _plane_points((0, 5), (0, 4), (3, 3), fixed_axis=2, fixed_val=3.0)
    wall_x0 = _plane_points((0, 0), (0, 4), (0, 3), fixed_axis=0, fixed_val=0.0)
    wall_x5 = _plane_points((5, 5), (0, 4), (0, 3), fixed_axis=0, fixed_val=5.0)
    wall_y0 = _plane_points((0, 5), (0, 0), (0, 3), fixed_axis=1, fixed_val=0.0)
    wall_y4 = _plane_points((0, 5), (4, 4), (0, 3), fixed_axis=1, fixed_val=4.0)

    all_points = np.vstack([floor, ceiling, wall_x0, wall_x5, wall_y0, wall_y4])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )
    return pcd
