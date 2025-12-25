# ☕ Presentasi Kedai Kopi — Machine Learning Project

Proyek ini merupakan implementasi machine learning sederhana untuk studi kasus kedai kopi, mencakup proses pengolahan data, pemodelan, dan pengujian dasar.  
Repository ini juga telah dilengkapi dengan **Continuous Integration (CI)** menggunakan **GitHub Actions** untuk memastikan kode dapat dijalankan secara konsisten di environment bersih.

---
Python == V3.10
How to use? install depedencies with "pip instal -r requirements.txt"

## 📌 Fitur Utama
- Script machine learning berbasis Python 3.10
- Dependency management menggunakan `pip`
- CI otomatis untuk:
  - setup environment
  - instalasi dependency
  - menjalankan script utama
- Struktur proyek sederhana dan mudah direproduksi

---

## 📂 Struktur Proyek
```text
.
├── Presentasi_Kedai_Kopi.py   # Script utama ML
├── requirements.txt          # Daftar dependency Python
├── .github/
│   └── workflows/
│       └── ci.yml            # Konfigurasi GitHub Actions
└── README.md

