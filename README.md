# 📡 Radar Data Processing & Optimization

**Author:** Moshe Ohana

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-vectorized-013243?logo=numpy)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-interactive%20charts-3F4F75?logo=plotly)](https://plotly.com/)

---

## 🚀 Live Demo

🔗 **[Open the Interactive Dashboard →](https://radar-data.streamlit.app)**

---

## 📋 Overview

A comprehensive Python project that demonstrates **real-world radar signal processing** skills, covering the full pipeline from raw data I/O through performance optimization and interactive visualization.

The project is built around a radar `.npz` dataset containing three arrays:
- **`signal`** — raw continuous radar signal (float64)
- **`signal_quality`** — discrete quality indicator per sample
- **`heart_state`** — discrete cardiac-state label per sample (0–4)

The project addresses four engineering challenges in a systematic and measurable way:

| Challenge | Technique | Result |
|---|---|---|
| Storage bloat | Dtype shrinking + ZIP compression | Significant file-size reduction |
| Slow RMS computation | NumPy vectorization (loop-free) | Large speed-up vs. baseline |
| Single-core batch bottleneck | `multiprocessing.Pool` | Near-linear parallel speed-up |
| Data exploration | Streamlit + Plotly dashboard | Interactive, deployable web app |

---

## ✨ Key Features

- **Lossless storage optimization** — automatically downcasts float arrays to the smallest safe integer dtype (e.g. float64 → bool for 0/1 columns, float64 → uint8 for 0–4 columns) and repackages with ZIP compression
- **Optional lossy compression** — reduces continuous float64 signals to float32 when slight precision loss is acceptable
- **Vectorized RMS computation** — replaces a nested Python loop with a single NumPy `reshape` + `mean` operation, measured with a built-in benchmark
- **Parallel batch processing** — benchmarks serial vs. `multiprocessing.Pool` on 1 000 identical NPZ files, printing speed-up factor
- **Interactive Streamlit dashboard** — upload any `.npz` file, inspect metadata, explore time-series plots, and view histograms with adjustable subsampling and bin controls

---

## 🗂️ Project Structure

```
Radar-data/
├── dashboard_app.py      # Streamlit web dashboard (upload & visualize .npz files)
├── processing.py         # NPZ storage optimizer (dtype shrink + ZIP compression)
├── optimization.py       # Vectorized RMS computation + runtime benchmark
├── RMS_computation.py    # Baseline RMS (Python loops — used for comparison)
├── parallelization.py    # Serial vs. multiprocessing batch benchmark
├── io.py                 # Generic .npz loader (field names + metadata)
├── utils.py              # Shared helpers (human_size, timeit_avg, chunk_nonoverlap)
├── radar_data_optimized.npz  # Sample optimized dataset
└── requirements.txt      # Python dependencies
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **NumPy** | Array math, dtype inspection, vectorized RMS, reshape tricks |
| **Streamlit** | Zero-boilerplate web dashboard |
| **Plotly** | Interactive time-series charts and histograms |
| **multiprocessing** | CPU-parallel batch file processing |

---

## ⚙️ Installation

```bash
git clone https://github.com/mosheohana/Radar-data.git
cd Radar-data
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Run the interactive dashboard locally

```bash
streamlit run dashboard_app.py
```

Upload any `.npz` file containing `signal`, `signal_quality`, and `heart_state` arrays to visualize them instantly.

### Optimize an NPZ file

```bash
# Dtype shrink + ZIP (recommended — best compression)
python processing.py --src radar_data.npz --dst radar_data_optimized.npz --op both

# Dtype shrink only (measure dtype benefit separately)
python processing.py --src radar_data.npz --dst out_dtype.npz --op dtype

# ZIP only (no dtype changes)
python processing.py --src radar_data.npz --dst out_zip.npz --op zip

# Optional: also reduce float64 signal → float32 (small precision trade-off)
python processing.py --src radar_data.npz --dst out_lossy.npz --op both --lossy-signal
```

### Benchmark RMS computation (baseline vs. vectorized)

```python
from RMS_computation import RMS_computation
from optimization import RMS_computation_optimized, benchmark_rms
import numpy as np

with np.load("radar_data_optimized.npz", allow_pickle=False) as f:
    signal = f["signal"]

benchmark_rms(RMS_computation, RMS_computation_optimized, signal, repeats=5)
# Example output: Baseline avg: 312.40 ms | Optimized avg: 4.82 ms | Speed-up: 64.81x
```

### Benchmark serial vs. parallel batch processing

```bash
python parallelization.py --src radar_data_optimized.npz --n 1000 --workers 8
# Example output:
#   Serial:   1000 files in 42.3s
#   Parallel: 1000 files in  7.1s  ⚡ Speed-up: 5.96x
```

---

## 📊 How the Optimization Works

### Storage Optimization (processing.py)

Each array is inspected at load time:

```
signal_quality (float64, values 0–4)
  → all values are integer-like  →  cast to int64  →  downcast to uint8  ✅ 8× smaller

heart_state (float64, values 0 or 1)
  → all values are 0/1  →  cast to bool  ✅ 64× smaller

signal (float64, continuous)
  → kept as float64 (or float32 with --lossy-signal)
```

After dtype shrinking, `np.savez_compressed` adds ZIP compression on top, yielding the maximum total reduction.

### RMS Vectorization (optimization.py)

**Baseline** — explicit Python loop over every value in every segment:
```python
for seg in segments:
    for val in seg:
        square_values.append(val**2)
    rms.append(sqrt(mean(square_values)))
```

**Optimized** — single NumPy expression, no Python loops:
```python
segments = signal[:n*L].reshape(n, L)   # 2D view, zero copy
rms = np.sqrt(np.mean(segments ** 2, axis=1))
```

---

## 👤 Author

**Moshe Ohana**  
[GitHub](https://github.com/mosheohana) · [LinkedIn](https://www.linkedin.com/in/moshe-ohana)
