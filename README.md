# pointcloud2ifc -- Automated Scan-to-BIM Pipeline

[![CI](https://github.com/MapIV/pointcloud2ifc/actions/workflows/ci.yml/badge.svg)](https://github.com/MapIV/pointcloud2ifc/actions/workflows/ci.yml)
[![Scan-to-IFC](https://github.com/MapIV/pointcloud2ifc/actions/workflows/scan2ifc.yml/badge.svg)](https://github.com/MapIV/pointcloud2ifc/actions/workflows/scan2ifc.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automated point cloud to IFC (BIM) conversion with CI/CD integration.
The first open-source tool that offers **automated IFC generation in CI** -- drop a scan into your repo and get BIM output as a build artifact.

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
IFC File (.ifc) + JSON Report
```

## CI/CD: Automated Scan-to-IFC in GitHub Actions

**This is the key differentiator.** No other tool offers automated IFC generation as a CI/CD pipeline.

### Use from another repository (reusable workflow)

```yaml
# .github/workflows/convert.yml in YOUR repo
name: Convert scans to IFC
on:
  push:
    paths: ["scans/**"]

jobs:
  scan2ifc:
    uses: MapIV/pointcloud2ifc/.github/workflows/scan2ifc.yml@main
    with:
      point_cloud_path: scans/building.ply
      method: ransac
      voxel_size: "0.05"
```

The IFC file and a JSON report are uploaded as build artifacts automatically.

### Manual trigger (ad-hoc conversion)

Go to **Actions > Scan-to-IFC Pipeline > Run workflow** and specify:
- Point cloud file path
- Segmentation method (`dbscan`, `ransac`, or `ml`)
- Output path (optional)
- Voxel size (optional)

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

### Convert a single point cloud to IFC

```bash
pointcloud2ifc convert input.ply -o output.ifc --method ransac --voxel-size 0.05
```

Supported input formats: PLY, PCD, LAS/LAZ.

### Batch-convert a directory

```bash
pointcloud2ifc batch scans/ --output-dir output/ --method ransac --voxel-size 0.05
```

Processes all PLY, PCD, and LAS/LAZ files in the directory. Generates an IFC file and JSON report for each input, plus a `batch_summary.json`.

### Evaluate against ground truth

```bash
pointcloud2ifc evaluate generated.ifc ground_truth.ifc --iou-threshold 0.5
```

### Python API

```python
from pointcloud2ifc.pipeline import Scan2IFCPipeline

pipeline = Scan2IFCPipeline()

# Single file
report = pipeline.run("scan.ply", "output.ifc", method="ransac")
print(f"Segments: {report.segments_found}, IFC elements: {report.ifc_elements_created}")

# Batch
reports = pipeline.run_batch("scans/", "output/", method="ransac")
```

### Run tests

```bash
pytest tests/ -v
```

## Architecture

```
src/pointcloud2ifc/
  __init__.py        # BIMNet category definitions
  cli.py             # Click CLI (convert, evaluate, batch)
  io.py              # Point cloud loading, IFC writing
  segmentation.py    # Semantic segmentation (DBSCAN, RANSAC, ML)
  pretrained.py      # Pretrained PointNet segmentation backend
  ifc_builder.py     # IFC model construction from segments
  pipeline.py        # End-to-end pipeline with batch processing and JSON reporting
```

## License

MIT
