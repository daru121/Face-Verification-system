<div align="center">
  <h1>Sistem Verifikasi Wajah</h1>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white">
    <img src="https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white">
  </p>
</div>

## 📋 Deskripsi
Sistem verifikasi wajah real-time yang dirancang khusus untuk lingkungan akademik. Menggunakan teknologi untuk memverifikasi identitas dengan akurasi tinggi dan anti-spoofing detection.

## ✨ Fitur Utama

<details>
<summary>🎯 Verifikasi Real-time</summary>

- Deteksi wajah instan
- Verifikasi dalam 0.15 detik
- Confidence level display
- Anti-spoofing detection
</details>

<details>
<summary>🔐 Keamanan</summary>

- Liveness detection
</details>

<details>
<summary>📊 Monitoring</summary>

- Real-time confidence metrics
- Performance analytics
</details>

## 🛠️ Tech Stack
| Komponen | Teknologi | Fungsi |
|----------|------------|---------|
| Image Processing | OpenCV | Pemrosesan gambar & video stream |
| Face Detection | face_recognition | Deteksi & pengenalan wajah |
| Backend Processing | dlib | Library pendukung face recognition |
| Programming | Python 3.8+ | Bahasa pemrograman utama |

## 📸 Screenshot
<summary>Proses Registrasi dengan 3 kali photo</summary>
<img src="src/img/face registrasi.png" alt="Success Result">

<summary>Hasil Verifikasi Berhasil </summary>
<img src="src/img/face verification.png" alt="Verification Process">


## Performa Sistem
| Metrik | Nilai |
|--------|--------|
| Waktu Deteksi | 0.15s |
| Waktu Verifikasi | 0.25s |
| Akurasi | 98.5% |
| FAR | 0.01% |
| FRR | 0.1% |

## 📝 Persyaratan Sistem
- CPU: Intel Core i5 atau lebih tinggi
- GPU: Opsional (untuk performa lebih baik)
- Kamera: 720p minimum
- OS: Windows 10/11, Linux, macOS

