"""Pretrained point cloud segmentation backends."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import open3d as o3d

from pointcloud2ifc import BIMNET_CATEGORIES
from pointcloud2ifc.segmentation import Segment

logger = logging.getLogger(__name__)

NUM_CLASSES = len(BIMNET_CATEGORIES)  # 14


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class SegmentationBackend(Protocol):
    """Protocol that all segmentation backends must satisfy."""

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Return per-point class predictions (N,) int array with values in [0, NUM_CLASSES)."""
        ...


# ---------------------------------------------------------------------------
# PointNet-style backbone (torch)
# ---------------------------------------------------------------------------

def _require_torch():
    """Import and return torch, raising a clear error if unavailable."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for ML-based segmentation. "
            "Install it with:  pip install 'pointcloud2ifc[ml]'  "
            "or:  pip install torch>=2.0"
        )


def _build_pointnet_model():
    """Build a PointNet-style segmentation model and return (model, torch)."""
    torch = _require_torch()
    import torch.nn as nn

    class PointNetSegmentation(nn.Module):
        """Minimal PointNet segmentation network.

        Architecture:
          Shared MLPs (3 -> 64 -> 128 -> 256 -> 512 -> 1024) on each point,
          global max-pool, concatenate global feature to per-point features,
          per-point MLPs (1024+128 -> 512 -> 256 -> NUM_CLASSES).
        """

        def __init__(self, num_classes: int = NUM_CLASSES):
            super().__init__()
            self.shared_mlp = nn.Sequential(
                nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            )
            # After concatenating global feature (1024) with intermediate feature (128)
            self.seg_mlp = nn.Sequential(
                nn.Linear(1024 + 128, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                (B, N, 3) point coordinates.

            Returns
            -------
            torch.Tensor
                (B, N, num_classes) logits.
            """
            B, N, _ = x.shape

            # Shared MLPs applied per-point (reshape to (B*N, 3))
            flat = x.reshape(B * N, 3)

            # We need the intermediate feature at dim=128 for skip connection
            h = flat
            # Manual forward through shared_mlp layers to grab intermediate
            layers = list(self.shared_mlp.children())
            # layers: Linear,BN,ReLU, Linear,BN,ReLU, Linear,BN,ReLU, ...
            # After second group (idx 3,4,5) we have 128-dim features
            for i, layer in enumerate(layers):
                h = layer(h)
                if i == 5:  # after ReLU of second group (128-dim)
                    intermediate = h.clone()  # (B*N, 128)

            global_feat = h.reshape(B, N, 1024)  # (B, N, 1024)
            global_feat = global_feat.max(dim=1, keepdim=True).values  # (B, 1, 1024)
            global_feat = global_feat.expand(B, N, 1024)  # (B, N, 1024)

            intermediate = intermediate.reshape(B, N, 128)  # (B, N, 128)

            concat = torch.cat([global_feat, intermediate], dim=2)  # (B, N, 1152)
            concat_flat = concat.reshape(B * N, 1024 + 128)

            logits = self.seg_mlp(concat_flat)  # (B*N, num_classes)
            return logits.reshape(B, N, -1)

    return PointNetSegmentation, torch


@dataclass
class PointNetBackend:
    """PointNet segmentation backend using a simple MLP architecture."""

    weights_path: str | None = None
    device: str = "cpu"

    def __post_init__(self):
        ModelClass, torch = _build_pointnet_model()
        self._torch = torch
        self.model = ModelClass(num_classes=NUM_CLASSES)

        if self.weights_path is not None:
            try:
                state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                logger.info("Loaded pretrained weights from %s", self.weights_path)
            except Exception as exc:
                warnings.warn(
                    f"Failed to load weights from {self.weights_path}: {exc}. "
                    "Using random initialization -- results will be poor.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "No pretrained weights provided for PointNet. "
                "Using random initialization -- predictions will be arbitrary. "
                "Train or download weights for meaningful results.",
                stacklevel=2,
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Run inference on (N, 3) points and return (N,) class ids."""
        torch = self._torch
        with torch.no_grad():
            tensor = torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(tensor)  # (1, N, C)
            preds = logits.argmax(dim=2).squeeze(0).cpu().numpy()
        return preds


# ---------------------------------------------------------------------------
# High-level segmenter
# ---------------------------------------------------------------------------

class PretrainedSegmenter:
    """Pluggable pretrained segmentation wrapper.

    Parameters
    ----------
    backend : str
        Backend name. Currently supported: ``"pointnet"``.
    weights_path : str | None
        Path to pretrained weights file.  If *None*, the model runs with
        random initialisation (useful for testing the pipeline).
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """

    _BACKENDS = {
        "pointnet": PointNetBackend,
    }

    def __init__(
        self,
        backend: str = "pointnet",
        weights_path: str | None = None,
        device: str = "cpu",
    ):
        if backend not in self._BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {list(self._BACKENDS)}"
            )
        self._backend: SegmentationBackend = self._BACKENDS[backend](
            weights_path=weights_path, device=device
        )

    def segment(self, pcd: o3d.geometry.PointCloud, backend: str = "pointnet") -> list[Segment]:
        """Segment a point cloud using the pretrained model.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Input point cloud.
        backend : str
            Ignored (kept for API compatibility); the backend is fixed at
            construction time.

        Returns
        -------
        list[Segment]
            One ``Segment`` per predicted class that has at least one point.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        preds = self._backend.predict(points)  # (N,)

        segments: list[Segment] = []
        for class_id in range(NUM_CLASSES):
            mask = preds == class_id
            if not mask.any():
                continue
            seg_colors = colors[mask] if colors is not None else None
            segments.append(
                Segment(
                    label=BIMNET_CATEGORIES[class_id],
                    label_id=class_id,
                    points=points[mask],
                    colors=seg_colors,
                )
            )
        return segments
