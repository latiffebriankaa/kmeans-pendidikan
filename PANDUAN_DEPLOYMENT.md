# 🚀 Panduan Deployment Streamlit
## Analisis Pemerataan Pendidikan Dasar Indonesia

---

## 📁 File yang Diperlukan
Pastikan kamu punya 3 file ini dalam satu folder:
- `app.py` → kode utama aplikasi
- `requirements.txt` → daftar library
- (opsional) file CSV dataset jika ingin pakai data asli

---

## 🖥️ Cara 1: Jalankan di Lokal (Laptop/PC)

### Langkah-langkah:
1. **Install Python** (versi 3.9 ke atas) jika belum ada
2. **Buka terminal / command prompt**, masuk ke folder project:
   ```
   cd nama_folder_project
   ```
3. **Install semua library:**
   ```
   pip install -r requirements.txt
   ```
4. **Jalankan aplikasi:**
   ```
   streamlit run app.py
   ```
5. Browser akan otomatis terbuka di `http://localhost:8501`

---

## 🌐 Cara 2: Deploy ke Streamlit Cloud (GRATIS & Online)

### Langkah-langkah:

### A. Persiapan GitHub
1. Buat akun di **github.com** jika belum punya
2. Buat repository baru (contoh nama: `kmeans-pendidikan`)
3. Upload ketiga file (`app.py`, `requirements.txt`, dan CSV jika ada) ke repository tersebut

### B. Deploy ke Streamlit Cloud
1. Buka **share.streamlit.io**
2. Login menggunakan akun GitHub
3. Klik tombol **"New app"**
4. Isi form:
   - **Repository:** pilih repo yang sudah dibuat tadi
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Klik **"Deploy!"**
6. Tunggu sekitar 2–5 menit hingga proses selesai
7. Aplikasi akan online di URL seperti:
   `https://nama-aplikasi.streamlit.app`

---

## 📋 Cara 3: Jalankan di Google Colab

Jika ingin coba tanpa install apapun, bisa pakai Google Colab:

```python
# Jalankan cell ini di Google Colab
!pip install streamlit pyngrok -q

# Tulis app.py
%%writefile app.py
# (paste isi app.py di sini)

# Jalankan Streamlit via ngrok
from pyngrok import ngrok
import subprocess

public_url = ngrok.connect(8501)
print("URL Aplikasi:", public_url)
subprocess.Popen(["streamlit", "run", "app.py"])
```

---

## ⚠️ Tips Penting

- Jika pakai **data asli dari Kaggle**, upload file CSV ke repository GitHub dan sesuaikan baris berikut di `app.py`:
  ```python
  df_raw = pd.read_csv('nama_file.csv')
  ```
- Pastikan nama kolom CSV sesuai dengan yang dipakai di kode
- Jika ada error saat deploy, cek tab **"Logs"** di Streamlit Cloud untuk melihat pesan errornya

---

## 🔗 Link Berguna
- Streamlit Cloud: https://share.streamlit.io
- Kaggle Dataset: https://www.kaggle.com/datasets/puanbeningpastika/dataset-pendidikan-sd-indonesia-2023-2024
- Dokumentasi Streamlit: https://docs.streamlit.io
