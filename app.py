import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ========= Loading (cache) =========
@st.cache_resource
def load_models():
    cal = joblib.load("artifacts/model_lgbm_isotonic.joblib")  # untuk prediksi proba
    lgbm_raw = joblib.load("artifacts/model_lgbm_raw.joblib")  # untuk SHAP/explain
    feats = joblib.load("artifacts/feature_order.joblib")
    thr = json.load(open("artifacts/threshold.json"))["thr_final"]
    return cal, lgbm_raw, feats, float(thr)

cal, lgbm_raw, FEATURE_ORDER, THRESHOLD = load_models()

st.set_page_config(page_title="ICU AMS – CDS", page_icon="🩺", layout="wide")
st.title("🩺 ICU Antimicrobial Stewardship – CDS")

st.markdown("""
CDS ini memberi **probabilitas infeksi bakteri** berbasis vital & lab, lalu memetakan ke rekomendasi:
- Prob rendah → **Stop / De-escalate**
- Prob menengah → **Evaluate in 24h**
- Prob tinggi → **Continue / Broaden if worsening**
""")

# ========= Helper =========
GUARD_TEMP_MAX = 38.0
GUARD_LACTATE_MAX = 2.0

def map_action(prob, row, thr=THRESHOLD):
    # default
    action = "Evaluate in 24h"
    if prob < thr:
        stable = True
        t = row.get("median_temp", np.nan)
        l = row.get("median_lactate", np.nan)
        if not np.isnan(t): stable &= (t < GUARD_TEMP_MAX)
        if not np.isnan(l): stable &= (l < GUARD_LACTATE_MAX)
        action = "Stop / De-escalate" if stable else "Evaluate in 24h"
    else:
        action = "Continue / Broaden if worsening"
    return action

def ensure_feature_df(df):
    # pastikan semua kolom fitur ada & urutannya konsisten
    for c in FEATURE_ORDER:
        if c not in df.columns:
            df[c] = np.nan
    df = df[FEATURE_ORDER]
    return df

# ========= Tabs =========
tab1, tab2 = st.tabs(["Single patient (manual)", "Batch CSV"])

with tab1:
    st.subheader("Single patient input")
    st.caption("Isi fitur minimal yang kamu punya. Kosongkan jika tidak tersedia (akan diperlakukan sebagai NaN).")

    # Buat input dinamis untuk beberapa fitur umum terlebih dulu
    defaults = {
        "median_temp": 37.0,
        "median_lactate": 1.5,
        "median_wbc": 10.0,
        "median_heartrate": 90.0,
        "median_resprate": 20.0,
        "median_sysbp": 110.0,
        "median_diasbp": 70.0
    }

    cols = st.columns(3)
    user_vals = {}
    for i, f in enumerate(FEATURE_ORDER):
        # hanya tampilkan beberapa fitur "umum"; sisanya lewat expander
        if f in defaults:
            with cols[i % 3]:
                val = st.number_input(f, value=float(defaults[f]), step=0.1, format="%.3f")
                user_vals[f] = val

    with st.expander("Isi fitur lain (opsional)"):
        for f in FEATURE_ORDER:
            if f in user_vals:
                continue
            user_vals[f] = st.number_input(f, value=float("nan"), step=0.1, format="%.3f")

    if st.button("Predict"):
        row = pd.DataFrame([user_vals])
        X = ensure_feature_df(row)
        prob = float(cal.predict_proba(X)[0,1])
        action = map_action(prob, row.iloc[0].to_dict(), thr=THRESHOLD)

        st.metric("Probabilitas infeksi (terkalibrasi)", f"{prob:.3f}")
        st.metric("Ambang aksi (sens tinggi)", f"{THRESHOLD:.3f}")
        st.success(f"Rekomendasi: **{action}**")

        # SHAP (optional): tampilkan top-5 kontributor
        try:
            import shap
            explainer = shap.TreeExplainer(lgbm_raw)
            sv = explainer.shap_values(X)  # untuk binary, hasil list; ambil kelas 1
            if isinstance(sv, list): sv = sv[1]
            contrib = pd.Series(sv[0], index=FEATURE_ORDER).sort_values(key=np.abs, ascending=False).head(5)
            st.subheader("Top-5 faktor yang mempengaruhi skor (SHAP)")
            st.bar_chart(contrib)
        except Exception as e:
            st.info(f"SHAP tidak ditampilkan ({e}).")

with tab2:
    st.subheader("Batch CSV")
    st.caption("Upload CSV berisi kolom fitur. Opsi: sertakan kolom identitas (misal `order_proc_id_coded`).")
    file = st.file_uploader("Pilih file CSV", type=["csv"])

    if file:
        raw = pd.read_csv(file)
        Xb = ensure_feature_df(raw.copy())
        probs = cal.predict_proba(Xb)[:,1]
        actions = []
        for i in range(len(Xb)):
            actions.append(map_action(probs[i], raw.iloc[i].to_dict(), thr=THRESHOLD))

        out = raw.copy()
        out["prob_infection"] = np.round(probs, 3)
        out["recommendation"] = actions

        st.dataframe(out.head(20))
        st.download_button("Download hasil (.csv)", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="cds_predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("⚠️ Catatan: Rekomendasi bersifat pendukung keputusan dan harus dikombinasikan dengan penilaian klinis, kultur, dan guideline (SSC, IDSA, WHO AWaRe).")