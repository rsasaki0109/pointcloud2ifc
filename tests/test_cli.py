"""Tests for pointcloud2ifc.cli."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from click.testing import CliRunner

from pointcloud2ifc.cli import cli


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_ply(tmp_path: Path) -> Path:
    """Create a small but segmentable PLY point cloud (floor + wall)."""
    rng = np.random.default_rng(42)

    # Floor plane at z=0
    floor = np.column_stack([
        rng.uniform(0, 3, 300),
        rng.uniform(0, 3, 300),
        rng.normal(0, 0.005, 300),
    ])
    # Wall plane at x=0
    wall = np.column_stack([
        rng.normal(0, 0.005, 300),
        rng.uniform(0, 3, 300),
        rng.uniform(0, 2, 300),
    ])
    pts = np.vstack([floor, wall])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    ply_path = tmp_path / "room.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    return ply_path


class TestCLI:
    """Test the Click CLI commands."""

    def test_version(self, runner: CliRunner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "convert" in result.output
        assert "evaluate" in result.output

    def test_convert_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--method" in result.output
        assert "--voxel-size" in result.output

    def test_convert_ply_dbscan(self, runner: CliRunner, sample_ply: Path):
        output_ifc = sample_ply.with_suffix(".ifc")
        result = runner.invoke(cli, [
            "convert", str(sample_ply),
            "-o", str(output_ifc),
            "--method", "dbscan",
            "--voxel-size", "0",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Done." in result.output
        assert output_ifc.exists()

    def test_convert_ply_ransac(self, runner: CliRunner, sample_ply: Path):
        output_ifc = sample_ply.with_suffix(".ifc")
        result = runner.invoke(cli, [
            "convert", str(sample_ply),
            "-o", str(output_ifc),
            "--method", "ransac",
            "--voxel-size", "0",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Done." in result.output

    def test_convert_default_output(self, runner: CliRunner, sample_ply: Path):
        result = runner.invoke(cli, [
            "convert", str(sample_ply),
            "--voxel-size", "0",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        default_output = sample_ply.with_suffix(".ifc")
        assert default_output.exists()

    def test_convert_nonexistent_file(self, runner: CliRunner):
        result = runner.invoke(cli, ["convert", "/nonexistent/file.ply"])
        assert result.exit_code != 0

    @pytest.mark.skipif(
        not _has_torch(), reason="torch not installed"
    )
    def test_convert_ml_succeeds(self, runner: CliRunner, sample_ply: Path):
        result = runner.invoke(cli, [
            "convert", str(sample_ply),
            "--method", "ml",
        ])
        assert result.exit_code == 0
        assert "Segments found" in result.output
