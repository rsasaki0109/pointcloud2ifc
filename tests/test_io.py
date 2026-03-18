"""Tests for pointcloud2ifc.io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from pointcloud2ifc.io import load_point_cloud, write_ifc


def _write_test_ply(path: Path, n: int = 200) -> None:
    """Write a minimal PLY point cloud."""
    rng = np.random.default_rng(0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    o3d.io.write_point_cloud(str(path), pcd)


def _write_test_pcd(path: Path, n: int = 200) -> None:
    """Write a minimal PCD point cloud."""
    rng = np.random.default_rng(0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    o3d.io.write_point_cloud(str(path), pcd)


class TestLoadPointCloud:
    """Test point cloud loading."""

    def test_load_ply(self, tmp_path: Path):
        ply_path = tmp_path / "test.ply"
        _write_test_ply(ply_path)
        pcd = load_point_cloud(ply_path)
        assert not pcd.is_empty()
        assert len(pcd.points) == 200

    def test_load_pcd(self, tmp_path: Path):
        pcd_path = tmp_path / "test.pcd"
        _write_test_pcd(pcd_path)
        pcd = load_point_cloud(pcd_path)
        assert not pcd.is_empty()
        assert len(pcd.points) == 200

    def test_voxel_downsampling(self, tmp_path: Path):
        ply_path = tmp_path / "test.ply"
        _write_test_ply(ply_path, n=5000)
        pcd = load_point_cloud(ply_path, voxel_size=0.1)
        # After downsampling, should have fewer points
        assert len(pcd.points) < 5000

    def test_no_downsampling_with_zero(self, tmp_path: Path):
        ply_path = tmp_path / "test.ply"
        _write_test_ply(ply_path, n=200)
        pcd = load_point_cloud(ply_path, voxel_size=0.0)
        assert len(pcd.points) == 200

    def test_unsupported_format(self, tmp_path: Path):
        bad_path = tmp_path / "test.xyz"
        bad_path.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported point cloud format"):
            load_point_cloud(bad_path)

    def test_empty_cloud_raises(self, tmp_path: Path):
        ply_path = tmp_path / "empty.ply"
        pcd = o3d.geometry.PointCloud()
        o3d.io.write_point_cloud(str(ply_path), pcd)
        with pytest.raises(ValueError, match="empty"):
            load_point_cloud(ply_path)


class TestWriteIfc:
    """Test IFC writing."""

    def test_write_creates_file(self, tmp_path: Path):
        from pointcloud2ifc.ifc_builder import build_ifc_model
        from pointcloud2ifc.segmentation import Segment

        seg = Segment(
            label="wall",
            label_id=0,
            points=np.random.default_rng(0).uniform(0, 1, (50, 3)),
        )
        model = build_ifc_model([seg])
        out = tmp_path / "output.ifc"
        write_ifc(model, out)
        assert out.exists()
        assert out.stat().st_size > 0
