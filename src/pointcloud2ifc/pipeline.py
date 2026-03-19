"""End-to-end scan-to-IFC pipeline with batch processing and JSON reporting."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pointcloud2ifc.io import load_point_cloud, write_ifc
from pointcloud2ifc.segmentation import segment
from pointcloud2ifc.ifc_builder import build_ifc_model


SUPPORTED_EXTENSIONS = {".ply", ".pcd", ".las", ".laz"}


@dataclass
class ConversionReport:
    """Statistics from a single scan-to-IFC conversion."""

    input_path: str
    output_path: str
    method: str
    voxel_size: float
    points_after_downsample: int
    segments_found: int
    ifc_elements_created: int
    segment_labels: dict[str, int]
    processing_time_seconds: float
    success: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Scan2IFCPipeline:
    """End-to-end pipeline for converting point clouds to IFC files."""

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        method: str = "dbscan",
        voxel_size: float = 0.05,
    ) -> ConversionReport:
        """Run a single scan-to-IFC conversion.

        Parameters
        ----------
        input_path : str | Path
            Path to the input point cloud file.
        output_path : str | Path | None
            Path for the output IFC file. Defaults to ``<input_stem>.ifc``.
        method : str
            Segmentation method: ``"dbscan"``, ``"ransac"``, or ``"ml"``.
        voxel_size : float
            Voxel size for downsampling in metres. 0 disables downsampling.

        Returns
        -------
        ConversionReport
            Statistics about the conversion.
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".ifc")
        else:
            output_path = Path(output_path)

        t0 = time.monotonic()

        try:
            pcd = load_point_cloud(input_path, voxel_size=voxel_size)
            num_points = len(pcd.points)

            segments = segment(pcd, method=method)
            num_segments = len(segments)

            model = build_ifc_model(segments)
            write_ifc(model, output_path)

            # Count IFC building elements
            ifc_elements = len(model.by_type("IfcBuildingElement"))

            # Tally segment labels
            label_counts: dict[str, int] = {}
            for seg in segments:
                label_counts[seg.label] = label_counts.get(seg.label, 0) + 1

            elapsed = time.monotonic() - t0

            report = ConversionReport(
                input_path=str(input_path),
                output_path=str(output_path),
                method=method,
                voxel_size=voxel_size,
                points_after_downsample=num_points,
                segments_found=num_segments,
                ifc_elements_created=ifc_elements,
                segment_labels=label_counts,
                processing_time_seconds=round(elapsed, 3),
                success=True,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            report = ConversionReport(
                input_path=str(input_path),
                output_path=str(output_path),
                method=method,
                voxel_size=voxel_size,
                points_after_downsample=0,
                segments_found=0,
                ifc_elements_created=0,
                segment_labels={},
                processing_time_seconds=round(elapsed, 3),
                success=False,
                error=str(exc),
            )

        # Write JSON report alongside the IFC
        report_path = Path(report.output_path).with_suffix(".json")
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        return report

    def run_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        method: str = "dbscan",
        voxel_size: float = 0.05,
    ) -> list[ConversionReport]:
        """Process all supported point cloud files in a directory.

        Parameters
        ----------
        input_dir : str | Path
            Directory containing point cloud files.
        output_dir : str | Path | None
            Output directory for IFC and JSON files. Defaults to *input_dir*.
        method : str
            Segmentation method.
        voxel_size : float
            Voxel size for downsampling.

        Returns
        -------
        list[ConversionReport]
        """
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        reports: list[ConversionReport] = []
        for f in files:
            out_path = output_dir / f.with_suffix(".ifc").name
            report = self.run(f, out_path, method=method, voxel_size=voxel_size)
            reports.append(report)

        # Write batch summary
        summary_path = output_dir / "batch_summary.json"
        summary = {
            "total_files": len(reports),
            "successful": sum(1 for r in reports if r.success),
            "failed": sum(1 for r in reports if not r.success),
            "reports": [r.to_dict() for r in reports],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return reports
