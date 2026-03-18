"""Tests for pointcloud2ifc.ifc_builder."""

from __future__ import annotations

import numpy as np
import ifcopenshell
import pytest

from pointcloud2ifc.ifc_builder import build_ifc_model
from pointcloud2ifc.segmentation import Segment


def _make_segment(label: str, label_id: int, n: int = 100) -> Segment:
    """Create a simple test segment with random points in a unit cube."""
    rng = np.random.default_rng(label_id)
    points = rng.uniform(0, 1, (n, 3))
    # Offset in z to separate floor/ceiling
    if label == "ceiling":
        points[:, 2] += 2.5
    return Segment(label=label, label_id=label_id, points=points)


class TestBuildIfcModel:
    """Test IFC model creation from segments."""

    def test_returns_ifc_file(self):
        segments = [_make_segment("wall", 0)]
        model = build_ifc_model(segments)
        assert isinstance(model, ifcopenshell.file)

    def test_has_project_structure(self):
        segments = [_make_segment("floor", 1)]
        model = build_ifc_model(segments)
        projects = model.by_type("IfcProject")
        sites = model.by_type("IfcSite")
        buildings = model.by_type("IfcBuilding")
        storeys = model.by_type("IfcBuildingStorey")
        assert len(projects) == 1
        assert len(sites) == 1
        assert len(buildings) == 1
        assert len(storeys) == 1

    def test_wall_creates_ifc_wall(self):
        segments = [_make_segment("wall", 0)]
        model = build_ifc_model(segments)
        walls = model.by_type("IfcWall")
        assert len(walls) == 1
        assert walls[0].Name == "wall_0000"

    def test_floor_creates_ifc_slab(self):
        segments = [_make_segment("floor", 1)]
        model = build_ifc_model(segments)
        slabs = model.by_type("IfcSlab")
        assert len(slabs) == 1

    def test_ceiling_creates_ifc_covering(self):
        segments = [_make_segment("ceiling", 2)]
        model = build_ifc_model(segments)
        coverings = model.by_type("IfcCovering")
        assert len(coverings) == 1

    def test_unknown_label_creates_proxy(self):
        segments = [_make_segment("other", 13)]
        model = build_ifc_model(segments)
        proxies = model.by_type("IfcBuildingElementProxy")
        assert len(proxies) == 1

    def test_multiple_segments(self):
        segments = [
            _make_segment("wall", 0),
            _make_segment("floor", 1),
            _make_segment("ceiling", 2),
        ]
        model = build_ifc_model(segments)
        walls = model.by_type("IfcWall")
        slabs = model.by_type("IfcSlab")
        coverings = model.by_type("IfcCovering")
        assert len(walls) == 1
        assert len(slabs) == 1
        assert len(coverings) == 1

    def test_elements_have_geometry(self):
        segments = [_make_segment("wall", 0)]
        model = build_ifc_model(segments)
        wall = model.by_type("IfcWall")[0]
        assert wall.Representation is not None
        assert wall.ObjectPlacement is not None

    def test_empty_segments_list(self):
        model = build_ifc_model([])
        assert isinstance(model, ifcopenshell.file)
        # Should still have project structure but no building elements
        assert len(model.by_type("IfcProject")) == 1
        assert len(model.by_type("IfcWall")) == 0

    def test_write_and_read_roundtrip(self, tmp_path):
        segments = [_make_segment("wall", 0), _make_segment("floor", 1)]
        model = build_ifc_model(segments)
        ifc_path = tmp_path / "test.ifc"
        model.write(str(ifc_path))

        reloaded = ifcopenshell.open(str(ifc_path))
        assert len(reloaded.by_type("IfcWall")) == 1
        assert len(reloaded.by_type("IfcSlab")) == 1
