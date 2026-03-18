"""CLI interface for pointcloud2ifc."""

from pathlib import Path

import click

from pointcloud2ifc import __version__


@click.group()
@click.version_option(version=__version__)
def cli():
    """Convert point clouds to IFC (BIM) files automatically."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output IFC file path. Defaults to <input_stem>.ifc",
)
@click.option(
    "--method",
    type=click.Choice(["dbscan", "ransac", "ml"]),
    default="dbscan",
    help="Segmentation method to use.",
)
@click.option("--voxel-size", type=float, default=0.05, help="Voxel size for downsampling (m).")
def convert(input_path: Path, output: Path | None, method: str, voxel_size: float):
    """Convert a point cloud file to IFC.

    Supported input formats: PLY, PCD, LAS/LAZ.
    """
    from pointcloud2ifc.io import load_point_cloud, write_ifc
    from pointcloud2ifc.segmentation import segment
    from pointcloud2ifc.ifc_builder import build_ifc_model

    if output is None:
        output = input_path.with_suffix(".ifc")

    click.echo(f"Loading point cloud: {input_path}")
    pcd = load_point_cloud(input_path, voxel_size=voxel_size)
    click.echo(f"  Points after downsampling: {len(pcd.points)}")

    click.echo(f"Segmenting with method: {method}")
    if method == "ml":
        click.echo("  Note: pretrained weights improve results significantly.")
        click.echo("  Without weights the model uses random initialisation.")
    segments = segment(pcd, method=method)
    click.echo(f"  Segments found: {len(segments)}")

    click.echo("Building IFC model...")
    model = build_ifc_model(segments)

    click.echo(f"Writing IFC: {output}")
    write_ifc(model, output)

    click.echo("Done.")


@cli.command()
@click.argument("generated_ifc", type=click.Path(exists=True, path_type=Path))
@click.argument("ground_truth_ifc", type=click.Path(exists=True, path_type=Path))
@click.option("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching.")
def evaluate(generated_ifc: Path, ground_truth_ifc: Path, iou_threshold: float):
    """Evaluate a generated IFC against a ground-truth IFC.

    Reports per-category precision, recall, F1 and overall accuracy.
    """
    import ifcopenshell

    click.echo(f"Generated:    {generated_ifc}")
    click.echo(f"Ground truth: {ground_truth_ifc}")

    gen_model = ifcopenshell.open(str(generated_ifc))
    gt_model = ifcopenshell.open(str(ground_truth_ifc))

    gen_elements = gen_model.by_type("IfcBuildingElement")
    gt_elements = gt_model.by_type("IfcBuildingElement")

    click.echo(f"Generated elements:    {len(gen_elements)}")
    click.echo(f"Ground truth elements: {len(gt_elements)}")

    # TODO: Implement geometric IoU matching and per-category metrics
    click.echo("Detailed evaluation metrics not yet implemented.")


if __name__ == "__main__":
    cli()
