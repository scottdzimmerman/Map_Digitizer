# Sharing this app with your team

Same pattern as `OG_Prod` (the GEM production dashboard) — push this folder to
its own GitHub repo and deploy it for free on Streamlit Community Cloud.

## Recommended: Streamlit Community Cloud (free, live URL)

What colleagues get: a link they open in a browser. No install.

1. Create a new GitHub repo (public or private) and push this whole folder
   (`app.py`, `map_digitizer/`, `requirements.txt`, `packages.txt`,
   `.streamlit/`, `assets/`) to it.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch, and `app.py` as the entry point. Deploy.
4. Share the resulting `https://<name>.streamlit.app` URL.

`packages.txt` tells Streamlit Cloud to `apt-get install tesseract-ocr` during
the build — this is required for the OCR step to work; nothing else to
configure.

Notes:
- Free tier apps are public by default; Streamlit Cloud also supports
  restricting viewers by email (Settings → Sharing) if you want it
  internal-only, or a simple password gate added to `app.py` if that's not
  available on your plan.
- To update later: push changes to the repo and the app redeploys
  automatically (reboot manually from share.streamlit.io if it doesn't).

## Alternatives

- **Hugging Face Spaces** (free): create a Space (SDK = Streamlit), upload the
  same files. Supports private Spaces.
- **Quick temporary share (no hosting):** run locally and expose it briefly —
  `streamlit run app.py` then in another terminal `npx localtunnel --port 8501`
  (or `ngrok http 8501`). Gives a temporary public URL; only live while your
  machine runs.
- **Internal server / Docker:** for a longer-lived internal deployment,
  containerize with a small Dockerfile (`apt-get install tesseract-ocr`,
  `pip install -r requirements.txt`, `streamlit run app.py`).

## Uploaded-file size limits

Streamlit's default max upload is 200 MB per file (map PDFs/images and tracker
exports both go through `st.file_uploader`), configurable via
`.streamlit/config.toml`'s `[server] maxUploadSize` if a map scan is larger
than that.
