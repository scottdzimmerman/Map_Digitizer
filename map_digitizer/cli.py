"""
Command-line driver.

    python -m map_digitizer.cli run my_map.yaml
    python -m map_digitizer.cli assess my_map.pdf
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import run_pipeline
from .assess import assess_pdf


def main(argv=None):
    parser = argparse.ArgumentParser(prog="map_digitizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline for a config file")
    p_run.add_argument("config_path")
    p_run.add_argument("--raster-image", default=None, help="Pre-extracted raster image (skips native pull)")
    p_run.add_argument("--page", type=int, default=0)

    p_assess = sub.add_parser("assess", help="Stage 0 only: classify a PDF")
    p_assess.add_argument("pdf_path")
    p_assess.add_argument("--page", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "assess":
        result = assess_pdf(args.pdf_path, page_index=args.page)
        print(f"Page size: {result.page_width_pt:.0f} x {result.page_height_pt:.0f} pt")
        print(f"Vector paths: {result.n_vector_paths}")
        print(f"Embedded images: {result.n_embedded_images}")
        print(f"Extractable text chars: {result.extractable_text_chars}")
        print(f"Likely kind: {result.likely_kind}")
        print(f"Needs OCR: {result.needs_ocr}")
        if result.native_image_size_px:
            print(f"Native raster size: {result.native_image_size_px[0]} x {result.native_image_size_px[1]} px")
        return 0

    if args.command == "run":
        config = load_config(args.config_path)
        result = run_pipeline(config, raster_image_path=args.raster_image, page_index=args.page)
        print(f"Extracted {len(result.extracted_records)} features "
              f"(georeference mean residual {result.georef.mean_residual_km:.1f} km, "
              f"max {result.georef.max_residual_km:.1f} km)")
        if result.sanity and result.sanity.flagged:
            print(f"Sanity filter flagged {len(result.sanity.flagged)} features as geographically implausible")
        for path in result.excel_paths:
            print(f"Wrote: {path}")
        print(f"Wrote: {result.html_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
