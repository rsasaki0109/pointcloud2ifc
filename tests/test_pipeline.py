"""Tests for pointcloud2ifc.pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from click.testing import CliRunner

from pointcloud2ifc.cli import cli
from pointcloud2ifc.pipeline import Scan2IFCPipeline


@pytest.fixture
def sample_ply(tmp_path: Path) -> Path:
    """Create a small PLY file with floor + wall."""
    rng = np.random.default_rng(42)
    floor = np.column_stack([
        rng.uniform(0, 3, 300),
        rng.uniform(0, 3, 300),
        rng.normal(0, 0.005, 300),
    ])
    wall = np.column_stack([
        rng.normal(0, 0.005, 300),
        rng.uniform(0, 3, 300),
        rng.uniform(0, 2, 300),
    ])
    pts = np.vstack([floor, wall])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ply_path = tmp_path / "test.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    return ply_path


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a directory with two PLY files."""
    rng = np.random.default_rng(42)
    input_dir = tmp_path / "scans"
    input_dir.mkdir()

    for name in ["room1.ply", "room2.ply"]:
        floor = np.column_stack([
            rng.uniform(0, 3, 200),
            rng.uniform(0, 3, 200),
            rng.normal(0, 0.005, 200),
        ])
        wall = np.column_stack([
            rng.normal(0, 0.005, 200),
            rng.uniform(0, 3, 200),
            rng.uniform(0, 2, 200),
        ])
        pts = np.vstack([floor, wall])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(input_dir / name), pcd)

    return input_dir


class TestScan2IFCPipeline:
    """Test the Scan2IFCPipeline class."""

    def test_run_produces_ifc_and_json(self, sample_ply: Path, tmp_path: Path):
        pipeline = Scan2IFCPipeline()
        output = tmp_path / "output.ifc"
        report = pipeline.run(sample_ply, output, method="ransac", voxel_size=0)

        assert report.success
        assert report.segments_found > 0
        assert report.ifc_elements_created > 0
        assert report.points_after_downsample > 0
        assert report.processing_time_seconds > 0
        assert output.exists()

        json_path = output.with_suffix(".json")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["success"] is True
        assert data["segments_found"] > 0

    def test_run_default_output(self, sample_ply: Path):
        pipeline = Scan2IFCPipeline()
        report = pipeline.run(sample_ply, method="dbscan", voxel_size=0)

        assert report.success
        default_ifc = sample_ply.with_suffix(".ifc")
        assert default_ifc.exists()
        assert sample_ply.with_suffix(".json").exists()

    def test_run_invalid_file(self, tmp_path: Path):
        pipeline = Scan2IFCPipeline()
        report = pipeline.run(tmp_path / "nonexistent.ply", method="dbscan")

        assert not report.success
        assert report.error is not None

    def test_run_batch(self, sample_dir: Path, tmp_path: Path):
        pipeline = Scan2IFCPipeline()
        output_dir = tmp_path / "output"
        reports = pipeline.run_batch(sample_dir, output_dir, method="ransac", voxel_size=0)

        assert len(reports) == 2
        assert all(r.success for r in reports)

        summary_path = output_dir / "batch_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_files"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0

    def test_run_batch_empty_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        pipeline = Scan2IFCPipeline()
        reports = pipeline.run_batch(empty_dir)

        assert len(reports) == 0


class TestBatchCLI:
    """Test the batch CLI command."""

    def test_batch_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--method" in result.output
        assert "--output-dir" in result.output

    def test_batch_converts_files(self, sample_dir: Path, tmp_path: Path):
        runner = CliRunner()
        output_dir = tmp_path / "batch_out"
        result = runner.invoke(cli, [
            "batch", str(sample_dir),
            "--output-dir", str(output_dir),
            "--method", "ransac",
            "--voxel-size", "0",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Batch complete" in result.output
        assert (output_dir / "batch_summary.json").exists()
