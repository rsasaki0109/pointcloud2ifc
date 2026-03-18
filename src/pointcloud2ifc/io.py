"""Point cloud loading and IFC file writing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    import ifcopenshell


def load_point_cloud(path: Path, voxel_size: float = 0.0) -> o3d.geometry.PointCloud:
    """Load a point cloud from PLY, PCD, or LAS/LAZ file.

    Parameters
    ----------
    path : Path
        Input file path. Supported extensions: .ply, .pcd, .las, .laz
    voxel_size : float
        If > 0, downsample with this voxel size (metres).

    Returns
    -------
    o3d.geometry.PointCloud
    """
    suffix = path.suffix.lower()

    if suffix in (".ply", ".pcd"):
        pcd = o3d.io.read_point_cloud(str(path))
    elif suffix in (".las", ".laz"):
        pcd = _load_las(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {path}")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def _load_las(path: Path) -> o3d.geometry.PointCloud:
    """Load a LAS/LAZ file into an Open3D point cloud.

    Requires the ``laspy`` package.
    """
    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy is required to read LAS/LAZ files. Install with: pip install laspy[lazrs]"
        )

    las = laspy.read(str(path))
    points = np.vstack((las.x, las.y, las.z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Transfer colours if available
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        colors = np.vstack((las.red, las.green, las.blue)).T.astype(np.float64)
        # LAS stores 16-bit colour; normalise to [0, 1]
        if colors.max() > 1.0:
            colors /= 65535.0 if colors.max() > 255 else 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def write_ifc(model: "ifcopenshell.file", path: Path) -> None:
    """Write an IFC model to disk.

    Parameters
    ----------
    model : ifcopenshell.file
        The IFC model to write.
    path : Path
        Destination file path (.ifc).
    """
    model.write(str(path))
