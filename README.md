# pointcloud2ifc

Automatic conversion of point clouds to IFC (BIM) files, leveraging BIMNet semantic categories (14-class: wall, floor, ceiling, door, window, column, beam, stair, railing, etc.).

## Installation

```bash
pip install -e .
```

For LAS/LAZ support:

```bash
pip install laspy[lazrs]
```

## Usage

### Convert a point cloud to IFC

```bash
pointcloud2ifc convert input.ply -o output.ifc --method ransac
```

Supported input formats: PLY, PCD, LAS/LAZ.

Segmentation methods:
- `dbscan` (default) - DBSCAN clustering with heuristic labelling
- `ransac` - Iterative RANSAC plane extraction
- `ml` - ML-based segmentation (not yet implemented)

### Evaluate against ground truth

```bash
pointcloud2ifc evaluate generated.ifc ground_truth.ifc --iou-threshold 0.5
```

## Architecture

```
src/pointcloud2ifc/
  __init__.py        # BIMNet category definitions
  cli.py             # Click CLI (convert, evaluate)
  io.py              # Point cloud loading, IFC writing
  segmentation.py    # Semantic segmentation (DBSCAN, RANSAC, ML stub)
  ifc_builder.py     # IFC model construction from segments
```

## License

MIT
