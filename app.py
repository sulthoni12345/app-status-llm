import time

import pandas as pd
import streamlit as st
from groq import Groq

from classifier_baru import NAMA_FITUR, classify, load_config

st.set_page_config(page_title="Prediksi Kelulusan Siswa (LLM)", page_icon="🤖")

# ---------- 1. Muat konfigurasi hasil evaluasi & siapkan client ----------
cfg = load_config("config.json")
FITUR = list(NAMA_FITUR)
JEDA_CSV = 8                       # detik antar request pada tab CSV (batas 8K token/menit)

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:                  # file secrets belum ada atau key belum diisi
    st.error("GROQ_API_KEY belum diatur. Isi di .streamlit/secrets.toml (lokal) "
             "atau menu Secrets (Streamlit Community Cloud).")
    st.stop()


@st.cache_resource
def get_client():
    return Groq(api_key=API_KEY)


client = get_client()


@st.cache_data(show_spinner=False)
def prediksi(ipk_tuple):
    # hasil disimpan di cache: data yang sama tidak memanggil API dua kali (hemat kuota)
    data = dict(zip(FITUR, ipk_tuple))
    return classify(client, cfg, data)


# ---------- 2. Sidebar: informasi konfigurasi yang dikunci ----------
ev = cfg["evaluasi"]
with st.sidebar:
    st.header("Tentang Model")
    st.write(f"LLM: `{cfg['model']}` (Groq)")
    st.write(f"Few-shot: {len(cfg['few_shot'])} contoh")
    st.metric("F1 macro LLM (data uji)", f"{ev['f1_macro']:.3f}")
    if ev.get("f1_dnn_pembanding") is not None:
        st.metric("F1 macro DNN (pembanding)", f"{ev['f1_dnn_pembanding']:.3f}")
    st.caption(f"Dievaluasi pada {ev['n']} data uji, tanggal {ev['tanggal']}")
    if ev.get("lolos", False):
        st.success("Lolos gerbang evaluasi.")
    else:
        st.warning("Belum lolos gerbang evaluasi.")
    with st.expander("Lihat system prompt"):
        st.code(cfg["system_prompt"], language=None)

st.title("🤖 Prediksi Kelulusan Siswa dengan LLM")
st.write("Masukkan data siswa untuk memprediksi status: **Lulus** atau **Tidak Lulus**.")

tab1, tab2 = st.tabs(["Prediksi Satu Siswa", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            jam = st.selectbox("Jam belajar per minggu", [1, 2, 3, 4],
                               format_func=lambda k: {1: "< 2 jam", 2: "2–5 jam",
                                                      3: "5–10 jam", 4: "> 10 jam"}[k], index=1)
            gagal = st.slider("Jumlah kelas yang pernah gagal", 0, 4, 0)
            absen = st.slider("Jumlah ketidakhadiran", 0, 75, 4)
            pend_ibu = st.selectbox("Pendidikan ibu", [0, 1, 2, 3, 4], index=2,
                                    format_func=lambda k: ["Tidak sekolah", "SD", "SMP", "SMA", "Perguruan tinggi"][k])
        with col2:
            nilai1 = st.slider("Nilai periode 1", 0.0, 20.0, 10.0, step=0.5)
            nilai2 = st.slider("Nilai periode 2", 0.0, 20.0, 10.0, step=0.5)
            pend_ayah = st.selectbox("Pendidikan ayah", [0, 1, 2, 3, 4], index=2,
                                     format_func=lambda k: ["Tidak sekolah", "SD", "SMP", "SMA", "Perguruan tinggi"][k])
            lanjut = st.radio("Ingin melanjutkan kuliah?", ["Ya", "Tidak"], horizontal=True)
            internet = st.radio("Memiliki akses internet di rumah?", ["Ya", "Tidak"], horizontal=True)
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        ya_tidak = lambda pilihan: 1 if pilihan == "Ya" else 0
        vals = (float(jam), float(gagal), float(absen), float(nilai1), float(nilai2),
                float(pend_ibu), float(pend_ayah), float(ya_tidak(lanjut)), float(ya_tidak(internet)))
        with st.spinner("Menghubungi Groq..."):
            hasil = prediksi(vals)
        tampil = {"Lulus": st.success, "Tidak Lulus": st.error}
        tampil.get(hasil, st.info)(f"Prediksi status: **{hasil}**")

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR)
             + " (maksimal 30 baris agar hemat kuota).")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30)
            if st.button("Prediksi semua"):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    vals = tuple(float(b[c]) for c in FITUR)
                    hasil.append(prediksi(vals))
                    bar.progress(i / len(df))
                    time.sleep(JEDA_CSV)
                df["prediksi_status"] = hasil
                st.dataframe(df)
                st.write(df["prediksi_status"].value_counts())
                st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                                   file_name="hasil_prediksi_llm.csv", mime="text/csv")
                