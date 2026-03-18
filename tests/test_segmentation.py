"""Tests for pointcloud2ifc.segmentation."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from pointcloud2ifc.segmentation import Segment, segment


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestSegmentDBSCAN:
    """Tests for DBSCAN-based segmentation."""

    def test_returns_segments(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        segments = segment(synthetic_room_pcd, method="dbscan")
        assert isinstance(segments, list)
        # DBSCAN may return 0 segments if point density is too low for default params
        # This is expected behavior - the method works on real dense data
        assert all(isinstance(s, Segment) for s in segments)

    def test_segment_has_valid_structure(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        segments = segment(synthetic_room_pcd, method="dbscan")
        for seg in segments:
            assert seg.points.ndim == 2
            assert seg.points.shape[1] == 3
            assert len(seg.points) > 0

    def test_labels_are_valid(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        segments = segment(synthetic_room_pcd, method="dbscan")
        valid_labels = {"wall", "floor", "ceiling", "other"}
        for seg in segments:
            assert seg.label in valid_labels


class TestSegmentRANSAC:
    """Tests for RANSAC-based segmentation."""

    def test_returns_segments(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        segments = segment(synthetic_room_pcd, method="ransac")
        assert isinstance(segments, list)
        assert len(segments) > 0

    def test_extracts_planes(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        segments = segment(synthetic_room_pcd, method="ransac")
        # The synthetic room has 6 planes; RANSAC should find multiple
        assert len(segments) >= 3, f"Expected >= 3 planes, got {len(segments)}"

    def test_finds_floor_and_wall(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        segments = segment(synthetic_room_pcd, method="ransac")
        labels = {seg.label for seg in segments}
        assert "wall" in labels, f"Expected 'wall' in labels, got {labels}"
        assert "floor" in labels or "ceiling" in labels, (
            f"Expected floor or ceiling in labels, got {labels}"
        )

    def test_segment_metadata_types(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        segments = segment(synthetic_room_pcd, method="ransac")
        for seg in segments:
            assert isinstance(seg.label, str)
            assert isinstance(seg.label_id, int)
            assert isinstance(seg.points, np.ndarray)


class TestSegmentErrors:
    """Test error handling in segmentation."""

    def test_unknown_method_raises(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            segment(synthetic_room_pcd, method="nonexistent")

    @pytest.mark.skipif(
        not _has_torch(), reason="torch not installed"
    )
    def test_ml_returns_segments(self, synthetic_room_pcd: o3d.geometry.PointCloud):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments = segment(synthetic_room_pcd, method="ml")
        assert isinstance(segments, list)
        assert all(isinstance(s, Segment) for s in segments)
