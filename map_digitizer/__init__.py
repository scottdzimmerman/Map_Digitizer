"""
map_digitizer
=============

A reproducible, config-driven pipeline for turning infrastructure map PDFs
into georeferenced GIS data and crosschecking it against a tracker database.

Systematizes the six-stage procedure (assess -> extract -> OCR -> georeference
-> convert -> crosscheck) as deterministic Python, with no calls to any AI
API. The only inputs that vary per map are captured in a single config file
(see `map_digitizer.config`) — color/stroke values, exclusion regions,
control points, and matching thresholds — so a new map or a new feature type
(area / point / line) is a new config, not new code.

Feature types supported:
    area  - polygons (oil/gas fields, concessions, protected areas, ...)
    point - point facilities (plants, terminals, compressor stations, ...)
    line  - linear features (pipelines, transmission lines, roads, ...)

See README.md for the end-to-end workflow and examples/ for sample configs.
"""

__version__ = "0.1.0"
