"""Build IFC models from segmented point clouds."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
import ifcopenshell
import ifcopenshell.api

if TYPE_CHECKING:
    from pointcloud2ifc.segmentation import Segment


def _guid() -> str:
    return ifcopenshell.guid.compress(uuid.uuid1().hex)


def build_ifc_model(segments: list["Segment"]) -> ifcopenshell.file:
    """Create an IFC file from segmented point cloud data.

    Each segment becomes an IfcBuildingElement with a bounding-box
    representation (IfcExtrudedAreaSolid).

    Parameters
    ----------
    segments : list[Segment]
        Labelled point-cloud segments.

    Returns
    -------
    ifcopenshell.file
    """
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4")

    project = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcProject", name="PointCloud2IFC")
    ifcopenshell.api.run("unit.assign_unit", ifc)
    ctx = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context",
        ifc,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=ctx,
    )

    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuildingStorey", name="Ground Floor")

    ifcopenshell.api.run("aggregate.assign_object", ifc, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", ifc, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", ifc, relating_object=building, products=[storey])

    _LABEL_TO_IFC_CLASS = {
        "wall": "IfcWall",
        "floor": "IfcSlab",
        "ceiling": "IfcCovering",
        "door": "IfcDoor",
        "window": "IfcWindow",
        "column": "IfcColumn",
        "beam": "IfcBeam",
        "stair": "IfcStairFlight",
        "railing": "IfcRailing",
        "furniture": "IfcFurnishingElement",
        "slab": "IfcSlab",
        "curtain_wall": "IfcCurtainWall",
        "roof": "IfcRoof",
        "other": "IfcBuildingElementProxy",
    }

    for i, seg in enumerate(segments):
        ifc_class = _LABEL_TO_IFC_CLASS.get(seg.label, "IfcBuildingElementProxy")
        element_name = f"{seg.label}_{i:04d}"

        element = ifcopenshell.api.run(
            "root.create_entity", ifc, ifc_class=ifc_class, name=element_name
        )
        ifcopenshell.api.run(
            "spatial.assign_container", ifc, relating_structure=storey, products=[element]
        )

        # Create bounding-box geometry
        _assign_bbox_geometry(ifc, element, seg.points, body)

    return ifc


def _assign_bbox_geometry(
    ifc: ifcopenshell.file,
    element,
    points: np.ndarray,
    context,
) -> None:
    """Assign an axis-aligned bounding box as the element geometry."""
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    dx, dy, dz = mx - mn

    # Avoid degenerate boxes
    dx = max(dx, 0.01)
    dy = max(dy, 0.01)
    dz = max(dz, 0.01)

    cx, cy, cz = (mn + mx) / 2.0

    # Create a simple extruded rectangle
    rectangle = ifc.createIfcRectangleProfileDef(
        ProfileType="AREA",
        XDim=float(dx),
        YDim=float(dy),
    )

    extrusion_direction = ifc.createIfcDirection((0.0, 0.0, 1.0))
    position = ifc.createIfcAxis2Placement3D(
        ifc.createIfcCartesianPoint((float(mn[0]), float(mn[1]), float(mn[2]))),
        None,
        None,
    )

    solid = ifc.createIfcExtrudedAreaSolid(
        SweptArea=rectangle,
        Position=position,
        ExtrudedDirection=extrusion_direction,
        Depth=float(dz),
    )

    shape_rep = ifc.createIfcShapeRepresentation(
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid],
    )

    product_shape = ifc.createIfcProductDefinitionShape(Representations=[shape_rep])
    element.Representation = product_shape

    # Set local placement at origin
    origin = ifc.createIfcAxis2Placement3D(
        ifc.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None
    )
    local_placement = ifc.createIfcLocalPlacement(RelativePlacement=origin)
    element.ObjectPlacement = local_placement
