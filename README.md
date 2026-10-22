# GEM Map Digitizer

A Streamlit front end for the `map_digitizer` pipeline: turn an infrastructure
map (PDF or image) into georeferenced WGS84 features, and optionally crosscheck
them against a GOGET (or any) tracker export — the same config-driven procedure
used in the Qatar and Algeria worked examples, with no AI extraction calls.

## What it does

1. You upload a map file (PDF or PNG/JPG) and a tracker export (CSV/XLSX, optional).
2. You tell it, once, the handful of things only a human can read off *this*
   map: legend swatch colors per feature class, 3+ control points (pixel
   position + known real-world lon/lat), and anything to exclude (legend box,
   inset map, title block).
3. It runs the same six-stage procedure as the manual workflow — assess,
   extract, OCR, georeference, sanity-check, crosscheck — with a progress bar,
   and gives back:
   - a **raw extraction workbook** (`<name>_extraction.xlsx`)
   - a **tracker crosscheck workbook** (`<name>_crosscheck.xlsx`), if you
     uploaded a tracker export
   - an **interactive QA map** (`<name>_map.html`), also shown inline
   - the **run config** (`<name>_config.yaml`), so a power user can rerun or
     tweak it later from the command line with `map_digitizer`'s own CLI

These are the exact same three deliverables as the Qatar/Algeria worked
examples (`example_1_raw_extraction` / `example_2_goget_crosscheck`), just
produced from the browser instead of a one-off script.

## Run it locally

```bash
pip install -r requirements.txt
# Tesseract OCR binary (not a pip package):
#   macOS:         brew install tesseract
#   Debian/Ubuntu: apt-get install tesseract-ocr
streamlit run app.py
```

## Sharing it with your team

See `DEPLOY.md` — the short version is: push this folder to a GitHub repo and
deploy it for free on Streamlit Community Cloud (`share.streamlit.io`), the
same way the GEM production dashboard (`OG_Prod`) is deployed. `packages.txt`
already lists the one system dependency (`tesseract-ocr`) Streamlit Cloud needs
to install.

## Notes / limitations

- This wraps the same `map_digitizer` package used in the Qatar/Algeria
  worked examples — pixel-traced shapes are approximate, not survey-grade,
  and OCR-derived names are unverified until a human checks them. Treat this
  as a triage tool, same as the manual procedure's own caveats say.
- Finding control points and legend colors still needs a human to look at the
  map once — that part isn't automated, by design (same as the original
  procedure). The app just makes reading off pixel positions and running the
  rest of the pipeline faster.
- The interactive QA map loads Leaflet from a CDN (unpkg) rather than
  vendoring it locally, so it needs an internet connection to render (same as
  the OpenStreetMap basemap tiles it already depends on).
