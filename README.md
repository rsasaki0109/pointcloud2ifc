# pointcloud2ifc

[![CI](https://github.com/MapIV/pointcloud2ifc/actions/workflows/ci.yml/badge.svg)](https://github.com/MapIV/pointcloud2ifc/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automatic conversion of point clouds to IFC (BIM) files using semantic segmentation.

## Pipeline

```
Point Cloud (PLY/PCD/LAS)
  |
  v
[1. Load & Downsample]  -- open3d voxel downsampling
  |
  v
[2. Segment]             -- DBSCAN clustering / RANSAC plane extraction
  |
  v
[3. Label]               -- Heuristic classification into BIMNet categories
  |
  v
[4. Build IFC]           -- Bounding-box geometry per segment (IfcExtrudedAreaSolid)
  |
  v
IFC File (.ifc)
```

### Segmentation methods

| Method   | Description |
|----------|-------------|
| `dbscan` | DBSCAN clustering followed by heuristic labelling based on surface normals and geometry |
| `ransac` | Iterative RANSAC plane extraction; planes classified by normal orientation |
| `ml`     | Pretrained PointNet segmentation (requires `pip install 'pointcloud2ifc[ml]'`) |

### BIMNet 14-category labels

| ID | Category     | IFC Class                |
|----|--------------|--------------------------|
| 0  | wall         | IfcWall                  |
| 1  | floor        | IfcSlab                  |
| 2  | ceiling      | IfcCovering              |
| 3  | door         | IfcDoor                  |
| 4  | window       | IfcWindow                |
| 5  | column       | IfcColumn                |
| 6  | beam         | IfcBeam                  |
| 7  | stair        | IfcStairFlight           |
| 8  | railing      | IfcRailing               |
| 9  | furniture    | IfcFurnishingElement     |
| 10 | slab         | IfcSlab                  |
| 11 | curtain_wall | IfcCurtainWall           |
| 12 | roof         | IfcRoof                  |
| 13 | other        | IfcBuildingElementProxy  |

## Installation

```bash
pip install -e .
```

For development (includes pytest and ruff):

```bash
pip install -e ".[dev]"
```

For ML-based segmentation (PointNet backbone, requires PyTorch):

```bash
pip install -e ".[ml]"
```

For LAS/LAZ support:

```bash
pip install laspy[lazrs]
```

## Usage

### Convert a point cloud to IFC

```bash
pointcloud2ifc convert input.ply -o output.ifc --method ransac --voxel-size 0.05
```

Supported input formats: PLY, PCD, LAS/LAZ.

### Evaluate against ground truth

```bash
pointcloud2ifc evaluate generated.ifc ground_truth.ifc --iou-threshold 0.5
```

### Run tests

```bash
pytest tests/ -v
```

## Architecture

```
src/pointcloud2ifc/
  __init__.py        # BIMNet category definitions
  cli.py             # Click CLI (convert, evaluate)
  io.py              # Point cloud loading, IFC writing
  segmentation.py    # Semantic segmentation (DBSCAN, RANSAC, ML)
  pretrained.py      # Pretrained PointNet segmentation backend
  ifc_builder.py     # IFC model construction from segments
```

## License

MIT
