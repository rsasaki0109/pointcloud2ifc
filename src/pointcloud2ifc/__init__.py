"""pointcloud2ifc - Automatic conversion of point clouds to IFC (BIM) files."""

__version__ = "0.1.0"

# BIMNet 14-category semantic labels
BIMNET_CATEGORIES = {
    0: "wall",
    1: "floor",
    2: "ceiling",
    3: "door",
    4: "window",
    5: "column",
    6: "beam",
    7: "stair",
    8: "railing",
    9: "furniture",
    10: "slab",
    11: "curtain_wall",
    12: "roof",
    13: "other",
}
