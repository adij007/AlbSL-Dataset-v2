# AlbSL Dataset v2

Albanian sign language (AlbSL) hand landmarks, training pipelines, and live apps.

- **Web UI:** [`web/`](web/) (React + Vite + FastAPI on port 8765). From repo root with `.venv` active: `pip install -r Dependencies/requirements.txt` and `pip install -r Dependencies/requirements-web.txt`. Then from `web/`: `npm install`, `npm run dev`, and `npm run api` (two terminals).
- **What to keep in this repo:** [PROJECT_INVENTORY.md](PROJECT_INVENTORY.md) (directory sizes, essential vs optional paths).
- **Data layout:** [datasets/README.md](datasets/README.md) · **Checkpoints:** [models/README.md](models/README.md)
- **Docs / tests / Docker:** [`off-app/`](off-app/README.md) — [DOCUMENTATION.md](off-app/docs/DOCUMENTATION.md), [README-TRAINING-PIPELINE.md](off-app/docs/README-TRAINING-PIPELINE.md), [PERSHKRIM_PROJEKTI_SQ.md](off-app/docs/PERSHKRIM_PROJEKTI_SQ.md).

## Windows: `albsl_app_v2` live (`cv2.imshow`)

If OpenCV reports **The function is not implemented** for `imshow` / `destroyAllWindows`, the wheel in use has **no HighGUI**. Common causes:

1. **`opencv-python-headless`** is installed (alone or **together** with `opencv-python` — the headless build can win).
2. Multiple OpenCV packages from different projects.

**Fix — remove every OpenCV wheel, then install one GUI build:**

```powershell
pip uninstall -y opencv-python-headless opencv-contrib-python-headless opencv-python opencv-contrib-python
pip install "opencv-python>=4.8.0,<5"
pip list | findstr /i opencv
```

You should see **only** `opencv-python` (one line). Then:

```powershell
cd D:\AlbSl-Dataset-v2
python Script\albsl_app_v2.py live
```

The app now calls **`_require_opencv_highgui()`** at the start of `live`; if GUI is missing you get a **RuntimeError** with these same hints **before** loading the camera loop.

## Docker

See [off-app/docker/albsl-app-v2/README.md](off-app/docker/albsl-app-v2/README.md).
