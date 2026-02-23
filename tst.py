import json
import time
from pathlib import Path

import suite2p

# ---------- sessions to process ----------
BASE = Path("/mnt/data/test_run")
SESSIONS = [
    "2026-01-29-17-35-36-141436",
]

for session_name in SESSIONS:
    session_dir = BASE / session_name
    data_dir = session_dir / "raw_data" / "mesoscope_data"
    save_dir = session_dir / "suite2p_original"
    save_dir.mkdir(exist_ok=True)

    # Load acquisition parameters for mesoscope ROI layout
    with open(data_dir / "suite2p_parameters.json") as f:
        acq = json.load(f)

    # ── Database dict (paths + acquisition layout) ──
    db = suite2p.default_db()
    db["data_path"] = [str(data_dir)]
    db["save_path0"] = str(save_dir)
    db["file_list"] = sorted(f.name for f in data_dir.glob("mesoscope_*.tiff"))
    db["nplanes"] = acq["plane_number"]  # 1
    db["nchannels"] = acq["channel_number"]  # 1
    db["nrois"] = acq["roi_number"]  # 3
    db["lines"] = acq["roi_lines"]  # per-ROI scan lines
    db["dy"] = acq["roi_y_coordinates"]  # ROI y offsets
    db["dx"] = acq["roi_x_coordinates"]  # ROI x offsets

    # ── Processing settings ──
    settings = suite2p.default_settings()

    # Top-level
    settings["fs"] = acq["frame_rate"]  # 10.01 Hz
    settings["tau"] = 0.4  # GCaMP6f decay constant (s)

    # Registration
    reg = settings["registration"]
    reg["nimg_init"] = 500  # frames for reference image
    reg["batch_size"] = 100  # frames per batch
    reg["maxregshift"] = 0.1  # max shift fraction
    reg["smooth_sigma"] = 1.15  # spatial smoothing (px)
    reg["smooth_sigma_time"] = 0.0  # temporal smoothing (0=off)
    reg["two_step_registration"] = False
    reg["th_badframes"] = 1.0  # bad-frame threshold
    reg["norm_frames"] = True  # clip to 1-99th %ile
    reg["do_bidiphase"] = False
    reg["bidiphase"] = 0.0
    reg["nonrigid"] = True
    reg["block_size"] = [128, 128]
    reg["snr_thresh"] = 1.2  # SNR threshold for blocks
    reg["maxregshiftNR"] = 5.0  # max block shift (px)

    # Detection
    det = settings["detection"]
    det["denoise"] = False
    det["threshold_scaling"] = 2.0
    det["max_overlap"] = 0.75
    det["highpass_time"] = 100  # temporal high-pass (frames)
    det["nbins"] = 5000  # max binned frames
    det["soma_crop"] = True
    det["sparsery_settings"]["highpass_neuropil"] = 25  # spatial HP for detection

    # Classification
    settings["classification"]["preclassify"] = 0.5

    # Extraction
    ext = settings["extraction"]
    ext["neuropil_extract"] = True
    ext["allow_overlap"] = False
    ext["min_neuropil_pixels"] = 350
    ext["inner_neuropil_radius"] = 2
    ext["neuropil_coefficient"] = 0.7
    ext["lam_percentile"] = 50.0

    # Deconvolution
    dcnv = settings["dcnv_preprocess"]
    dcnv["baseline"] = "maximin"
    dcnv["win_baseline"] = 60.0
    dcnv["sig_baseline"] = 10.0
    dcnv["prctile_baseline"] = 8.0

    # ── Run ──
    print(f"\n{'=' * 60}")
    print(f"Processing: {session_name}")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    suite2p.run_s2p(db=db, settings=settings)
    elapsed = time.perf_counter() - t0
    print(f"Finished {session_name} in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
