"""Tests for pointcloud2ifc.pretrained."""

from __future__ import annotations

import warnings

import numpy as np
import open3d as o3d
import pytest

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


class TestPointNetModel:
    """Tests for the PointNet segmentation model."""

    def test_forward_pass_shape(self):
        """Model forward pass with random data produces correct output shape."""
        from pointcloud2ifc.pretrained import _build_pointnet_model, NUM_CLASSES

        ModelClass, torch = _build_pointnet_model()
        model = ModelClass(num_classes=NUM_CLASSES)
        model.eval()

        B, N = 2, 256
        x = torch.randn(B, N, 3)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (B, N, NUM_CLASSES)

    def test_single_point_cloud(self):
        """Model handles a single point cloud (batch size 1)."""
        from pointcloud2ifc.pretrained import _build_pointnet_model, NUM_CLASSES

        ModelClass, torch = _build_pointnet_model()
        model = ModelClass(num_classes=NUM_CLASSES)
        model.eval()

        x = torch.randn(1, 100, 3)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 100, NUM_CLASSES)


class TestPretrainedSegmenter:
    """Tests for the PretrainedSegmenter wrapper."""

    def test_segment_returns_valid_segments(self):
        """segment() returns a list of valid Segment objects."""
        from pointcloud2ifc.pretrained import PretrainedSegmenter
        from pointcloud2ifc.segmentation import Segment
        from pointcloud2ifc import BIMNET_CATEGORIES

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segmenter = PretrainedSegmenter(backend="pointnet")

        # Create a small synthetic point cloud
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((200, 3)).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        segments = segmenter.segment(pcd)

        assert isinstance(segments, list)
        assert len(segments) > 0  # random weights should produce at least one class
        for seg in segments:
            assert isinstance(seg, Segment)
            assert seg.points.ndim == 2
            assert seg.points.shape[1] == 3
            assert len(seg.points) > 0
            assert seg.label in BIMNET_CATEGORIES.values()
            assert seg.label_id in BIMNET_CATEGORIES

    def test_unknown_backend_raises(self):
        """Requesting an unknown backend raises ValueError."""
        from pointcloud2ifc.pretrained import PretrainedSegmenter

        with pytest.raises(ValueError, match="Unknown backend"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                PretrainedSegmenter(backend="nonexistent")

    def test_total_points_match(self):
        """All points are accounted for across segments."""
        from pointcloud2ifc.pretrained import PretrainedSegmenter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segmenter = PretrainedSegmenter(backend="pointnet")

        rng = np.random.default_rng(1)
        N = 150
        pts = rng.standard_normal((N, 3)).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        segments = segmenter.segment(pcd)
        total = sum(len(s.points) for s in segments)
        assert total == N
