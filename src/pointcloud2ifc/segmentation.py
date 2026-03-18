"""Semantic segmentation of point clouds into BIM categories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from pointcloud2ifc import BIMNET_CATEGORIES


@dataclass
class Segment:
    """A labelled subset of a point cloud."""

    label: str
    label_id: int
    points: np.ndarray  # (N, 3)
    colors: np.ndarray | None = None  # (N, 3) or None
    metadata: dict = field(default_factory=dict)


def segment(
    pcd: o3d.geometry.PointCloud,
    method: str = "dbscan",
) -> list[Segment]:
    """Segment a point cloud into BIM-category clusters.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud (optionally with normals).
    method : str
        One of ``"dbscan"``, ``"ransac"``, ``"ml"``.

    Returns
    -------
    list[Segment]
    """
    if method == "dbscan":
        return _segment_dbscan(pcd)
    elif method == "ransac":
        return _segment_ransac(pcd)
    elif method == "ml":
        return _segment_ml(pcd)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


# ---------------------------------------------------------------------------
# DBSCAN clustering (geometric only, no semantics)
# ---------------------------------------------------------------------------

def _segment_dbscan(pcd: o3d.geometry.PointCloud) -> list[Segment]:
    """Cluster using DBSCAN and assign heuristic labels.

    This is a baseline that groups spatially close points. Horizontal planes
    are labelled as floor/ceiling, large vertical planes as walls, and the
    rest as 'other'.
    """
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    labels_arr = np.array(
        pcd.cluster_dbscan(eps=0.1, min_points=50, print_progress=False)
    )
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    max_label = int(labels_arr.max())
    segments: list[Segment] = []

    for cluster_id in range(max_label + 1):
        mask = labels_arr == cluster_id
        cluster_pts = points[mask]
        cluster_normals = normals[mask]
        cluster_colors = colors[mask] if colors is not None else None

        label_id, label = _heuristic_label(cluster_pts, cluster_normals)

        segments.append(
            Segment(
                label=label,
                label_id=label_id,
                points=cluster_pts,
                colors=cluster_colors,
            )
        )

    return segments


def _heuristic_label(points: np.ndarray, normals: np.ndarray) -> tuple[int, str]:
    """Assign a BIMNet category based on simple geometric heuristics."""
    mean_normal = normals.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal) + 1e-8

    vertical = abs(mean_normal[2])  # z-component

    z_range = points[:, 2].max() - points[:, 2].min()
    z_mean = points[:, 2].mean()

    # Mostly horizontal surface
    if vertical > 0.8:
        if z_mean < np.median(points[:, 2]):
            return 1, "floor"
        else:
            return 2, "ceiling"

    # Tall, vertical surface -> wall
    if vertical < 0.3 and z_range > 1.0:
        return 0, "wall"

    return 13, "other"


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def _segment_ransac(pcd: o3d.geometry.PointCloud) -> list[Segment]:
    """Iteratively extract planes using RANSAC.

    Planes are classified by orientation (horizontal -> floor/ceiling,
    vertical -> wall).
    """
    remaining = pcd
    segments: list[Segment] = []

    for _ in range(20):  # extract at most 20 planes
        if len(remaining.points) < 100:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inliers) < 100:
            break

        inlier_cloud = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)

        pts = np.asarray(inlier_cloud.points)
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal) + 1e-8

        # Classify by normal direction
        if abs(normal[2]) > 0.8:
            z_mean = pts[:, 2].mean()
            all_z_mean = np.asarray(pcd.points)[:, 2].mean()
            if z_mean < all_z_mean:
                label_id, label = 1, "floor"
            else:
                label_id, label = 2, "ceiling"
        elif abs(normal[2]) < 0.3:
            label_id, label = 0, "wall"
        else:
            label_id, label = 13, "other"

        colors = np.asarray(inlier_cloud.colors) if inlier_cloud.has_colors() else None
        segments.append(Segment(label=label, label_id=label_id, points=pts, colors=colors))

    return segments


# ---------------------------------------------------------------------------
# ML-based segmentation (stub)
# ---------------------------------------------------------------------------

def _segment_ml(pcd: o3d.geometry.PointCloud) -> list[Segment]:
    """Segment using a trained ML model (BIMNet-style).

    Not yet implemented - requires a trained point cloud segmentation model.
    """
    raise NotImplementedError(
        "ML-based segmentation is not yet implemented. "
        "Use --method dbscan or --method ransac as a baseline."
    )
