# src/dashboard_app.py
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Radar NPZ Dashboard", layout="wide")

st.title("Radar Data — Interactive Dashboard")

st.markdown(
    "Upload an `.npz` file, visualize the three arrays (`signal`, `signal_quality`, `heart_state`), "
    "and interact with the plots (zoom/pan). Use subsampling if rendering is slow."
)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Controls")
    default_sub = 10
    sub = st.number_input(
        "Subsample (plot every N-th sample)",
        min_value=1, max_value=500, value=default_sub, step=1,
        help="Increase if plots feel slow (e.g., 10, 20, 50)."
    )
    nbins_signal = st.slider("Histogram bins (signal)", 10, 200, 60)
    nbins_quality = st.slider("Histogram bins (signal_quality)", 5, 50, 20)
    nbins_heart = st.slider("Histogram bins (heart_state)", 2, 10, 5)
    st.caption("Tip: try larger subsample on slow machines.")

# ---------------------------
# File uploader
# ---------------------------
file = st.file_uploader("Choose a .npz file", type=["npz"])

@st.cache_data(show_spinner=False)
def load_npz(file_obj):
    # Streamlit gives a BytesIO-like object; np.load can read it directly
    with np.load(file_obj, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    # small meta for display
    meta = {k: dict(dtype=str(v.dtype), shape=tuple(v.shape), nbytes=int(v.nbytes)) for k, v in arrays.items()}
    return arrays, meta

def pick_key(arrays, expected_name):
    """Return expected_name if present, else pick the first 1D numeric array."""
    if expected_name in arrays:
        return expected_name
    # Fallback: best-effort pick
    for k, v in arrays.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and np.issubdtype(v.dtype, np.number):
            return k
    return None

if not file:
    st.info("Upload a .npz file to begin.")
else:
    arrays, meta = load_npz(file)

    # Choose fields (handle different naming gracefully)
    signal_key = pick_key(arrays, "signal")
    quality_key = "signal_quality" if "signal_quality" in arrays else pick_key(arrays, "signal_quality")
    heart_key = "heart_state" if "heart_state" in arrays else pick_key(arrays, "heart_state")

    # Basic sanity
    missing = [name for name, key in [("signal", signal_key), ("signal_quality", quality_key), ("heart_state", heart_key)] if key is None]
    if missing:
        st.error(f"Could not locate required arrays: {', '.join(missing)}. "
                 f"Available keys: {list(arrays.keys())}")
        st.stop()

    # Show meta
    with st.expander("File summary", expanded=False):
        st.write("**Arrays found:**", list(arrays.keys()))
        st.json(meta, expanded=False)

    # Subsample for plotting
    sig = arrays[signal_key][::sub]
    sq = arrays[quality_key][::sub]
    hs = arrays[heart_key][::sub]

    # ---------------------------
    # Time-series (interactive)
    # ---------------------------
    st.subheader("Time Series")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig_sig = px.line(y=sig, labels={"y": signal_key}, title=f"{signal_key} (subsample={sub})")
        st.plotly_chart(fig_sig, use_container_width=True)
    with c2:
        fig_sq = px.line(y=sq, labels={"y": quality_key}, title=f"{quality_key} (subsample={sub})")
        st.plotly_chart(fig_sq, use_container_width=True)
    with c3:
        fig_hs = px.line(y=hs, labels={"y": heart_key}, title=f"{heart_key} (subsample={sub})")
        st.plotly_chart(fig_hs, use_container_width=True)

    # ---------------------------
    # Histograms
    # ---------------------------
    st.subheader("Histograms")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.caption("Signal distribution")
        st.plotly_chart(px.histogram(x=arrays[signal_key], nbins=nbins_signal, title="Signal Histogram"),
                        use_container_width=True)
    with c5:
        st.caption("Signal quality distribution")
        st.plotly_chart(px.histogram(x=arrays[quality_key], nbins=nbins_quality, title="Signal Quality Histogram"),
                        use_container_width=True)
    with c6:
        st.caption("Heart state distribution")
        st.plotly_chart(px.histogram(x=arrays[heart_key], nbins=nbins_heart, title="Heart State Histogram"),
                        use_container_width=True)

    # ---------------------------
    # Small stats
    # ---------------------------
    with st.expander("Quick stats"):
        def stats(a):
            return dict(count=a.size, mean=float(np.mean(a)), std=float(np.std(a)),
                        min=float(np.min(a)), max=float(np.max(a)))
        st.write({
            signal_key: stats(arrays[signal_key]),
            quality_key: stats(arrays[quality_key]),
            heart_key: stats(arrays[heart_key]),
        })
