"""Klasifikasi status kelulusan siswa (dataset student-mat, UCI 320) menggunakan LLM via Groq API.
Dipakai saat EVALUASI dan saat DEPLOY (Modul 2B) agar perilakunya sama persis."""
import json
import re
import time

import groq

# nama fitur yang ditampilkan ke LLM (urutan & nama kolom sama dengan dataset_nilai_siswa.csv)
NAMA_FITUR = {
    "jam_belajar_kategori": "Jam belajar per minggu (1: <2 jam, 2: 2-5 jam, 3: 5-10 jam, 4: >10 jam)",
    "jumlah_gagal": "Jumlah kelas yang pernah gagal",
    "jumlah_absen": "Jumlah ketidakhadiran",
    "nilai_periode_1": "Nilai periode 1 (0-20)",
    "nilai_periode_2": "Nilai periode 2 (0-20)",
    "pendidikan_ibu": "Pendidikan ibu (0: tidak ada, 4: tinggi)",
    "pendidikan_ayah": "Pendidikan ayah (0: tidak ada, 4: tinggi)",
    "ingin_lanjut_kuliah": "Ingin melanjutkan kuliah (1=ya, 0=tidak)",
    "akses_internet": "Memiliki akses internet di rumah (1=ya, 0=tidak)",
}


def load_config(path="config_baru.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(data):
    """dict data siswa -> teks satu baris untuk LLM."""
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):          # contoh (few-shot), boleh kosong
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def parse_label(output, labels):
    """Rapikan jawaban model. Hanya label yang persis cocok (tanpa huruf kecil/besar, tanda baca,
    dan spasi di ujung) yang diterima. Di luar daftar -> 'invalid'."""
    jawab = re.sub(r"[^a-z ]", "", output.lower()).strip()
    peta = {label.lower(): label for label in labels}
    return peta.get(jawab, "invalid")


class KuotaHabis(Exception):
    """Batas kuota API tercapai. Bukan jawaban 'invalid' dari model, jadi tidak boleh disimpan."""


def classify(client, cfg, data, max_retry=3):
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
            )
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except groq.RateLimitError:
            tunggu = 15 * (percobaan + 1)
            print(f"  batas kuota tercapai, menunggu {tunggu} detik...", flush=True)
            time.sleep(tunggu)
    raise KuotaHabis("batas kuota masih tercapai setelah beberapa kali mencoba")