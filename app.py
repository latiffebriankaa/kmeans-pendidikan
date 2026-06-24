import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Konfigurasi dasar halaman
st.set_page_config(
    page_title="Analisis Pemerataan Pendidikan Dasar Indonesia - Kelompok 2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS kustom untuk mempercantik tampilan dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Font global halaman */
    html, body, [class*="css"], .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Desain sidebar kiri */
    section[data-testid="stSidebar"] {
        background-color: #E2E8F0 !important;
        border-right: 1px solid #CBD5E1 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #1E293B !important;
    }
    
    /* Header menu di sidebar */
    .sidebar-header {
        text-align: center;
        padding: 25px 10px;
        margin-bottom: 25px;
        border-bottom: 1px solid #CBD5E1;
    }
    
    .group-badge-sidebar {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.15);
        letter-spacing: 0.5px;
    }
    
    /* Desain menu navigasi tombol */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px !important;
        display: flex;
        flex-direction: column;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        margin-bottom: 2px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        transform: translateX(3px);
    }
    
    /* Style saat menu aktif/dipilih */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Sembunyikan bulatan radio default */
    div[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stRadioCircle"],
    div[data-testid="stSidebar"] div[role="radiogroup"] label span[data-testid="stRadioCircle"] {
        display: none !important;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        margin-left: 0px !important;
        padding-left: 0px !important;
    }
    
    /* Header judul utama */
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        text-align: center;
        padding-top: 15px;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 0.95rem;
        font-weight: 500;
        color: #64748B;
        text-align: center;
        margin-bottom: 35px;
    }
    
    /* Kartu metrik ringkasan */
    .metric-card-wrapper {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card-wrapper:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px 0 rgba(0, 0, 0, 0.04);
        border-color: #CBD5E1;
    }
    
    .card-blue { border-top: 4px solid #0284C7 !important; }
    .card-teal { border-top: 4px solid #0D9488 !important; }
    .card-indigo { border-top: 4px solid #6366F1 !important; }
    .card-amber { border-top: 4px solid #D97706 !important; }
    
    .metric-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    /* Kartu informasi kelompok cluster */
    .cluster-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease;
    }
    
    .cluster-card:hover {
        transform: translateY(-2px);
    }
    
    .cluster-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
    }
    
    .cluster-card-count {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 10px;
    }
    
    .cluster-card-members {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
        background: #F8FAFC;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #F1F5F9;
    }
    
    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.8rem;
        padding: 25px 0 10px 0;
        border-top: 1px solid #E2E8F0;
        margin-top: 50px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Koordinat lokasi provinsi untuk menggambar peta sebaran geografis
def get_province_coordinates():
    return {
        'Aceh': (95.32, 4.69),
        'Sumatera Utara': (99.07, 2.11),
        'Sumatera Barat': (100.12, -0.74),
        'Riau': (101.97, 0.54),
        'Jambi': (102.78, -1.61),
        'Sumatera Selatan': (104.17, -3.31),
        'Bengkulu': (102.24, -3.58),
        'Lampung': (105.15, -4.85),
        'Kep. Bangka Belitung': (106.14, -2.74),
        'Kepulauan Bangka Belitung': (106.14, -2.74),
        'Kepulauan Riau': (104.44, 1.08),
        'DKI Jakarta': (106.82, -6.17),
        'Jawa Barat': (107.61, -6.91),
        'Jawa Tengah': (110.02, -7.15),
        'DI Yogyakarta': (110.37, -7.80),
        'Jawa Timur': (112.73, -7.54),
        'Banten': (106.12, -6.44),
        'Bali': (115.18, -8.40),
        'Nusa Tenggara Barat': (117.41, -8.57),
        'Nusa Tenggara Timur': (121.57, -8.65),
        'Kalimantan Barat': (111.10, -0.28),
        'Kalimantan Tengah': (113.92, -1.60),
        'Kalimantan Selatan': (115.12, -3.00),
        'Kalimantan Timur': (116.47, 0.53),
        'Kalimantan Utara': (116.22, 3.07),
        'Sulawesi Utara': (124.70, 1.25),
        'Sulawesi Tengah': (120.98, -1.43),
        'Sulawesi Selatan': (119.97, -3.66),
        'Sulawesi Tenggara': (122.07, -4.14),
        'Gorontalo': (122.52, 0.69),
        'Sulawesi Barat': (119.33, -2.84),
        'Maluku': (130.14, -3.24),
        'Maluku Utara': (127.37, 0.64),
        'Papua Barat': (132.90, -1.33),
        'Papua': (138.08, -2.50),
        'Papua Selatan': (139.08, -6.50),
        'Papua Tengah': (136.00, -3.80),
        'Papua Pegunungan': (138.80, -4.10),
        'Papua Barat Daya': (131.25, -1.10)
    }

# Menghilangkan spasi dan karakter khusus nama provinsi agar mudah dicocokkan
def normalize_province_name(name):
    name = str(name).lower()
    name = name.replace('prov. ', '').replace('provinsi ', '').strip()
    name = name.replace('.', '').replace('kepulauan', 'kep').replace(' ', '')
    return name


# Fungsi untuk merapikan kolom dan menghitung rasio indikator clustering
def preprocess_dataframe(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, df.columns != '']
    
    if 'Provinsi' in df.columns and 'Siswa' in df.columns:
        df['provinsi'] = df['Provinsi'].str.replace('Prov. ', '', regex=False).str.strip()
        df = df[df['provinsi'] != 'Luar Negeri']
        
        # Reset indeks agar pemetaan array PCA tidak bergeser/out of bounds
        df = df.reset_index(drop=True)
        
        numeric_cols = [c for c in df.columns if c not in ['Provinsi', 'provinsi']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        k1 = "Kepala Sekolah dan Guru(<S1)"
        k2 = "Kepala Sekolah dan Guru(≥ S1)"
        if k1 in df.columns and k2 in df.columns:
            df['jumlah_guru'] = df[k1] + df[k2]
        else:
            df['jumlah_guru'] = df['Siswa'] // 20
            
        df['jumlah_siswa'] = df['Siswa']
        df['rasio_siswa_guru'] = np.where(df['jumlah_guru'] > 0, df['jumlah_siswa'] / df['jumlah_guru'], 0).round(1)
        
        # Mengubah nilai absolut putus sekolah/mengulang ke persentase (%)
        df['angka_mengulang'] = np.where(df['jumlah_siswa'] > 0, (df['Mengulang'] / df['jumlah_siswa']) * 100, 0).round(2)
        df['angka_putus_sekolah'] = np.where(df['jumlah_siswa'] > 0, (df['Putus Sekolah'] / df['jumlah_siswa']) * 100, 0).round(2)
        
        df['jumlah_rombel'] = df['Rombongan Belajar']
        
        c_baik = "Ruang kelas(baik)"
        c_r1 = "Ruang kelas(rusak ringan)"
        c_r2 = "Ruang kelas(rusak sedang)"
        c_r3 = "Ruang kelas(rusak berat)"
        
        total_kelas = df[c_baik] + df[c_r1] + df[c_r2] + df[c_r3]
        df['kondisi_ruang_kelas_baik_pct'] = np.where(total_kelas > 0, (df[c_baik] / total_kelas) * 100, 0).round(1)
        
        features = [
            'provinsi', 'jumlah_siswa', 'jumlah_guru', 'rasio_siswa_guru',
            'angka_mengulang', 'angka_putus_sekolah', 'jumlah_rombel',
            'kondisi_ruang_kelas_baik_pct'
        ]
        return df[features]
        
    return df.reset_index(drop=True)


# Membuat data simulasi jika berkas CSV tidak ditemukan
def generate_simulated_data():
    np.random.seed(42)
    provinsi = [
        'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi',
        'Sumatera Selatan', 'Bengkulu', 'Lampung', 'Kep. Bangka Belitung',
        'Kepulauan Riau', 'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah',
        'DI Yogyakarta', 'Jawa Timur', 'Banten', 'Bali',
        'Nusa Tenggara Barat', 'Nusa Tenggara Timur', 'Kalimantan Barat',
        'Kalimantan Tengah', 'Kalimantan Selatan', 'Kalimantan Timur',
        'Kalimantan Utara', 'Sulawesi Utara', 'Sulawesi Tengah',
        'Sulawesi Selatan', 'Sulawesi Tenggara', 'Gorontalo',
        'Sulawesi Barat', 'Maluku', 'Maluku Utara', 'Papua Barat',
        'Papua', 'Papua Selatan', 'Papua Tengah',
        'Papua Pegunungan', 'Papua Barat Daya'
    ]

    def gen(n, s_range, r_range, p_range, k_range):
        return {
            'jumlah_siswa': np.random.randint(*s_range, n),
            'jumlah_guru': np.random.randint(s_range[0]//20, s_range[1]//18, n),
            'rasio_siswa_guru': np.random.uniform(*r_range, n).round(1),
            'angka_mengulang': np.random.uniform(0.5, 3.5, n).round(2),
            'angka_putus_sekolah': np.random.uniform(*p_range, n).round(2),
            'jumlah_rombel': np.random.randint(s_range[0]//35, s_range[1]//25, n),
            'kondisi_ruang_kelas_baik_pct': np.random.uniform(*k_range, n).round(1),
        }

    d1 = gen(9,  (250000, 3500000), (18, 25), (0.05, 0.3),  (70, 95))
    d2 = gen(18, (50000,  600000),  (20, 30), (0.1,  0.8),  (55, 80))
    d3 = gen(11, (10000,  150000),  (25, 40), (0.5,  2.5),  (30, 65))

    all_data = {k: np.concatenate([d1[k], d2[k], d3[k]]) for k in d1}
    df = pd.DataFrame(all_data)
    df.insert(0, 'provinsi', provinsi)
    return df.reset_index(drop=True)


# Fungsi pemuatan data utama
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            return preprocess_dataframe(df_raw)
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
            
    local_csv = "kelayakan-pendidikan-indonesia.csv"
    if os.path.exists(local_csv):
        try:
            df_raw = pd.read_csv(local_csv)
            return preprocess_dataframe(df_raw)
        except Exception:
            pass
            
    return generate_simulated_data()


# Fungsi komputasi clustering K-Means dan analisis reduksi dimensi PCA
@st.cache_data
def run_clustering(df, k):
    fitur = [
        'jumlah_siswa', 'jumlah_guru', 'rasio_siswa_guru',
        'angka_mengulang', 'angka_putus_sekolah',
        'jumlah_rombel', 'kondisi_ruang_kelas_baik_pct'
    ]
    X = df[fitur].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)

    df = df.copy()
    df['cluster'] = kmeans.labels_

    sil = silhouette_score(X_scaled, kmeans.labels_)
    wcss_list = []
    sil_list  = []
    for ki in range(2, 11):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss_list.append(km.inertia_)
        sil_list.append(silhouette_score(X_scaled, km.labels_))

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    return df, X_scaled, X_pca, kmeans, sil, wcss_list, sil_list, explained, fitur


# Menu navigasi dan parameter di sidebar kiri
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <div class="group-badge-sidebar">KELOMPOK 2</div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #475569; margin-top: 8px; letter-spacing: 0.5px;">PROJECT AKHIR ML</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Menu Navigasi")
    menu = st.radio(
        "Pilih Halaman Analisis:",
        options=[
            "Dashboard Ringkasan",
            "Analisis Elbow & K",
            "Visualisasi Spasial & PCA",
            "Detail & Pencarian",
            "Ekspor Data"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Konfigurasi Model")
    k_value = st.slider("Jumlah Cluster (K)", min_value=2, max_value=8, value=3, step=1, help="Tentukan jumlah cluster untuk analisis K-Means.")

    st.markdown("### Input Data")
    uploaded = st.file_uploader("Upload file CSV (opsional)", type=["csv"])
    if not uploaded:
        if os.path.exists("kelayakan-pendidikan-indonesia.csv"):
            st.success("Aktif: Data Riil (Local CSV)")
        else:
            st.info("Aktif: Data Simulasi Default")
    else:
        st.success("Data Sukses Terupload")

    st.markdown("---")
    st.markdown("### Pengaturan Visual")
    color_theme = st.selectbox(
        "Palet Warna Grafik:",
        options=["Modern Tech", "Ocean Breeze", "Sunset Glow", "Classic Slate"]
    )
    layout_style = st.radio(
        "Tata Letak Kartu Cluster:",
        options=["Grid (Sejajar)", "List (Vertikal)"]
    )

    st.markdown("---")
    st.markdown("### Informasi")
    st.info("**Model:** K-Means Clustering\n\n**Tingkat:** Pendidikan Dasar SD/MI\n\n**Sumber:** Kemendikbudristek")
    
    st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.75rem; margin-top: 25px; border-top: 1px solid #E2E8F0; padding-top: 15px;">
            © 2026 — Teknik Informatika<br>Kelompok 2
        </div>
    """, unsafe_allow_html=True)


# Load data dan jalankan clustering
df_raw = load_data(uploaded)
df, X_scaled, X_pca, kmeans, sil_score_val, wcss_list, sil_list, explained, fitur = run_clustering(df_raw, k_value)

# Pemetaan geografis koordinat provinsi
coords_dict = get_province_coordinates()
normalized_coords = {normalize_province_name(k): v for k, v in coords_dict.items()}

df['norm_prov'] = df['provinsi'].apply(normalize_province_name)
df['lon'] = df['norm_prov'].map(lambda x: normalized_coords.get(x, (np.nan, np.nan))[0])
df['lat'] = df['norm_prov'].map(lambda x: normalized_coords.get(x, (np.nan, np.nan))[1])

# Definisi palet warna grafis berdasarkan pilihan tema
themes_dict = {
    "Modern Tech": ['#4F46E5', '#0D9488', '#0284C7', '#D97706', '#7C3AED', '#EC4899', '#2563EB', '#475569'],
    "Ocean Breeze": ['#0284C7', '#0D9488', '#10B981', '#06B6D4', '#3B82F6', '#64748B', '#0F172A', '#94A3B8'],
    "Sunset Glow": ['#7C3AED', '#EC4899', '#EF4444', '#F59E0B', '#10B981', '#6366F1', '#3B82F6', '#64748B'],
    "Classic Slate": ['#475569', '#0284C7', '#10B981', '#D97706', '#EF4444', '#6366F1', '#8B5CF6', '#EC4899']
}
colors_list = themes_dict.get(color_theme, themes_dict["Modern Tech"])

# Injeksi CSS dinamis untuk mewarnai cluster cards (transparansi 5%)
dynamic_css = ""
for i in range(k_value):
    color = colors_list[i]
    bg_color = color + "0D"      
    border_color = color + "26"  
    dynamic_css += f"""
    .card-cluster-{i} {{
        background-color: {bg_color} !important;
        border-left: 6px solid {color} !important;
        border-color: {border_color} !important;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }}
    """
st.markdown(f"<style>{dynamic_css}</style>", unsafe_allow_html=True)

# Konfigurasi style global diagram Matplotlib (Tema Terang)
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#1E293B'
plt.rcParams['axes.labelcolor'] = '#475569'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['grid.color'] = '#CBD5E1'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#94A3B8'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Mapping nama label cluster dan keterangannya
label_map = {i: f"Cluster {chr(65+i)}" for i in range(k_value)}
label_desc = {
    0: "Kondisi Baik",
    1: "Kondisi Sedang",
    2: "Perlu Perhatian",
    3: "Grup D", 4: "Grup E", 5: "Grup F", 6: "Grup G", 7: "Grup H"
}

df['label_cluster'] = df['cluster'].apply(
    lambda x: f"{label_map[x]} – {label_desc.get(x, 'Grup ' + chr(65+x))}"
)

# Label fitur global untuk kolom dataframe tabel
fitur_label = ['Jml Siswa', 'Jml Guru', 'Rasio Siswa/Guru', 'Angka Mengulang (%)',
               'Angka Putus Sekolah (%)', 'Jml Rombel', 'Kondisi Kelas Baik (%)']
fitur_label_short = ['Jml Siswa', 'Jml Guru', 'Rasio S/G', 'Mengulang',
                     'Putus Sekolah', 'Rombel', 'Kelas Baik (%)']

# Judul halaman utama aplikasi
st.markdown('<div class="main-title">Analisis Pemerataan Pendidikan Dasar Antar Provinsi di Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Menggunakan Algoritma K-Means Clustering | Data Kemendikbudristek | Dikembangkan oleh Kelompok 2</div>', unsafe_allow_html=True)
st.markdown("---")


# Tampilan utama berdasarkan pilihan navigasi menu
if menu == "Dashboard Ringkasan":
    st.subheader("Ringkasan Hasil Analisis Clustering")

    # Kartu indikator metrik utama
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card-wrapper card-blue">
            <div class="metric-title">Total Provinsi</div>
            <div class="metric-value">{len(df)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card-wrapper card-teal">
            <div class="metric-title">Jumlah Cluster (K)</div>
            <div class="metric-value">{k_value}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card-wrapper card-indigo">
            <div class="metric-title">Silhouette Score</div>
            <div class="metric-value">{sil_score_val:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card-wrapper card-amber">
            <div class="metric-title">PCA Explained Variance</div>
            <div class="metric-value">{sum(explained):.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Pembagian Provinsi Per Cluster")

    # Kartu pembagian kelompok cluster
    if layout_style == "Grid (Sejajar)":
        cols = st.columns(min(k_value, 3))
        for i in range(k_value):
            subset = df[df['cluster'] == i]
            card_class = f'card-cluster-{i}' if i < 8 else 'card-cluster-generic'
            with cols[i % 3]:
                prov_list = "  •  ".join(subset['provinsi'].values)
                st.markdown(f"""
                <div class="cluster-card {card_class}">
                    <div class="cluster-card-title" style="color: {colors_list[i]};">{label_map[i]} — {label_desc.get(i, 'Grup')}</div>
                    <div class="cluster-card-count">{len(subset)} Provinsi</div>
                    <div class="cluster-card-members"><strong>Anggota:</strong><br>{prov_list}</div>
                </div>""", unsafe_allow_html=True)
    else:
        for i in range(k_value):
            subset = df[df['cluster'] == i]
            card_class = f'card-cluster-{i}' if i < 8 else 'card-cluster-generic'
            prov_list = "  •  ".join(subset['provinsi'].values)
            st.markdown(f"""
            <div class="cluster-card {card_class}">
                <div class="cluster-card-title" style="color: {colors_list[i]};">{label_map[i]} — {label_desc.get(i, 'Grup')}</div>
                <div class="cluster-card-count">{len(subset)} Provinsi</div>
                <div class="cluster-card-members"><strong>Anggota:</strong><br>{prov_list}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Perbandingan Profil Indikator Kunci")
    
    col_vis1, col_vis2 = st.columns([3, 2])
    
    # Hitung profil rata-rata masing-masing cluster
    cluster_profile = df.groupby('cluster')[fitur].mean().round(2)
    cluster_profile.index = [f"{label_map[i]} – {label_desc.get(i,'')}" for i in cluster_profile.index]
    cluster_profile.columns = fitur_label
    
    with col_vis1:
        fig_profile, ax_profile = plt.subplots(figsize=(8, 4.5), facecolor='none')
        vis_df = pd.DataFrame({
            'Rasio Siswa/Guru': cluster_profile['Rasio Siswa/Guru'],
            'Angka Mengulang (%)': cluster_profile['Angka Mengulang (%)'],
            'Angka Putus Sekolah (%)': cluster_profile['Angka Putus Sekolah (%)'],
            'Kondisi Kelas Baik (%)': cluster_profile['Kondisi Kelas Baik (%)']
        })
        
        vis_df.plot(kind='bar', ax=ax_profile, color=colors_list[:4], width=0.75)
        ax_profile.set_xlabel('Cluster', fontsize=9, fontweight='bold')
        ax_profile.set_ylabel('Nilai Rata-rata', fontsize=9, fontweight='bold')
        ax_profile.grid(True, axis='y', linestyle='--', alpha=0.7, color='#CBD5E1', linewidth=0.8)
        plt.xticks(rotation=0, fontsize=8)
        plt.yticks(fontsize=8)
        ax_profile.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#E2E8F0')
        plt.tight_layout()
        st.pyplot(fig_profile)
        plt.close()
        
    with col_vis2:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4.5), facecolor='none')
        sizes = df['cluster'].value_counts().sort_index()
        labels = [f"{label_map[i]}\n({v} prov)" for i, v in sizes.items()]
        
        ax_pie.pie(
            sizes, labels=labels, colors=colors_list[:len(sizes)],
            autopct='%1.1f%%', startangle=140, pctdistance=0.7,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
        )
        ax_pie.set_title('Proporsi Provinsi per Cluster', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close()
        
    st.subheader("Detail Profil Rata-rata per Indikator")
    st.dataframe(cluster_profile.style.background_gradient(cmap='GnBu', axis=0), width='stretch')

elif menu == "Analisis Elbow & K":
    st.subheader("Evaluasi K-Means: Penentuan Nilai Cluster (K) Optimal")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
        ax.plot(range(2, 11), wcss_list, color='#0284C7', marker='o', linewidth=2, markersize=8, markerfacecolor='#0284C7', markeredgecolor='white', markeredgewidth=1.5)
        ax.axvline(x=k_value, color='#DC2626', linestyle='--', linewidth=1.5, label=f'K={k_value} (dipilih)')
        ax.set_title('Elbow Method (WCSS)', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Jumlah Cluster (K)', fontsize=10, labelpad=8)
        ax.set_ylabel('WCSS (Inersia)', fontsize=10, labelpad=8)
        ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')
        ax.grid(True, which='both', linestyle='--', alpha=0.7, color='#CBD5E1', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
        ax.plot(range(2, 11), _sil_list := sil_list, color='#0D9488', marker='s', linewidth=2, markersize=8, markerfacecolor='#0D9488', markeredgecolor='white', markeredgewidth=1.5)
        ax.axvline(x=k_value, color='#DC2626', linestyle='--', linewidth=1.5, label=f'K={k_value} (dipilih)')
        ax.set_title('Silhouette Score vs K', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Jumlah Cluster (K)', fontsize=10, labelpad=8)
        ax.set_ylabel('Silhouette Score', fontsize=10, labelpad=8)
        ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')
        ax.grid(True, which='both', linestyle='--', alpha=0.7, color='#CBD5E1', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Tabel Evaluasi Nilai K (Clustering Score)")
    eval_df = pd.DataFrame({
        'Jumlah Cluster (K)': list(range(2, 11)),
        'WCSS (Inertia)': [round(w, 2) for w in wcss_list],
        'Silhouette Score': [round(s, 4) for s in sil_list]
    })
    eval_df['Status'] = eval_df['Jumlah Cluster (K)'].apply(lambda x: 'Dipilih (Aktif)' if x == k_value else '')
    st.dataframe(eval_df.style.highlight_max(subset=['Silhouette Score'], color='#D1FAE5'), width='stretch')

    st.info(f"""
    **Interpretasi Silhouette Score:**
    - **>= 0.70** -> Struktur cluster sangat kuat (Sangat Baik)
    - **0.50 - 0.69** -> Struktur cluster cukup baik (Baik)
    - **0.25 - 0.49** -> Struktur cluster lemah (Cukup)
    - **< 0.25** -> Tidak ada struktur yang nyata (Kurang)

    Skor Silhouette untuk K={k_value} adalah **{sil_score_val:.4f}** (Kategori: **{'Baik/Sangat Baik' if sil_score_val >= 0.5 else 'Cukup'}**).
    """)

elif menu == "Visualisasi Spasial & PCA":
    st.subheader("Peta Spasial Distribusi Cluster Pemerataan Pendidikan Indonesia")
    st.markdown("Sebaran geografis berdasarkan bujur (longitude) dan lintang (latitude) asli provinsi di Indonesia.")
    
    map_df = df.dropna(subset=['lon', 'lat']).copy()
    if len(map_df) > 0:
        fig_map, ax_map = plt.subplots(figsize=(12, 6.2), facecolor='none')
        
        for c in range(k_value):
            mask = map_df['cluster'] == c
            if mask.sum() > 0:
                ax_map.scatter(
                    map_df.loc[mask, 'lon'], map_df.loc[mask, 'lat'],
                    c=colors_list[c], label=f"{label_map[c]} – {label_desc.get(c, '')}",
                    s=240, alpha=0.9, edgecolors='white', linewidth=1.2, zorder=3
                )
                
                # Menampilkan label teks nama provinsi di atas peta
                for idx, row in map_df[mask].iterrows():
                    ax_map.annotate(
                        row['provinsi'], (row['lon'], row['lat']),
                        fontsize=7, weight='bold', color='#1E293B',
                        xytext=(4, 4), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#CBD5E1', linewidth=0.5)
                    )
                        
        ax_map.set_title('Peta Geografis Cluster Pemerataan Pendidikan Indonesia', fontsize=13, fontweight='bold', pad=15)
        ax_map.set_xlabel('Garis Bujur (Longitude)', fontsize=9)
        ax_map.set_ylabel('Garis Lintang (Latitude)', fontsize=9)
        
        # Mengatur batas visual agar pas melingkupi kepulauan Indonesia
        ax_map.set_xlim(94, 142)
        ax_map.set_ylim(-11.5, 6.5)
        
        ax_map.legend(loc='lower left', fontsize=9, frameon=True, facecolor='white', edgecolor='#E2E8F0', shadow=False)
        ax_map.grid(True, linestyle='--', alpha=0.7, color='#CBD5E1', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_map)
        plt.close()
    else:
        st.warning("Data koordinat provinsi tidak ditemukan.")
        
    st.markdown("---")

    # Scatter plot pemisahan cluster secara matematis (PCA 2D)
    st.subheader("Scatter Plot Hasil Reduksi Dimensi (PCA 2D)")
    st.markdown("Memetakan data multi-variabel ke grafik 2 dimensi koordinat utama.")
    fig_pca, ax_pca = plt.subplots(figsize=(12, 7.5), facecolor='none')
    
    for c in range(k_value):
        mask = df['cluster'] == c
        ax_pca.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors_list[c], label=f"{label_map[c]} – {label_desc.get(c, '')}",
            s=200, alpha=0.9, edgecolors='white', linewidth=1.2
        )
        
        for i, row in df[mask].iterrows():
            ax_pca.annotate(
                row['provinsi'], (X_pca[i, 0], X_pca[i, 1]),
                fontsize=8, weight='bold', color='#1E293B',
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='#E2E8F0', linewidth=0.5)
            )

    pca_temp = PCA(n_components=2, random_state=42)
    pca_temp.fit(X_scaled)
    centroids_pca = pca_temp.transform(kmeans.cluster_centers_)
    # Outline/glow effect untuk centroid model
    ax_pca.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c='white', marker='o', s=450, zorder=4, alpha=1.0
    )
    ax_pca.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c='#0F172A', marker='X', s=250, zorder=5, label='Centroid Model',
        edgecolors='white', linewidth=1.0
    )

    ax_pca.set_title('Grafik Sebaran Hasil Reduksi Dimensi PCA', fontsize=13, fontweight='bold', pad=15)
    ax_pca.set_xlabel(f'PC1 ({explained[0]:.1f}% Variance)', fontsize=10, labelpad=8)
    ax_pca.set_ylabel(f'PC2 ({explained[1]:.1f}% Variance)', fontsize=10, labelpad=8)
    ax_pca.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='#E2E8F0', shadow=False)
    ax_pca.grid(True, linestyle='--', alpha=0.7, color='#CBD5E1', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig_pca)
    plt.close()

    st.markdown("---")
    
    # Heatmap visualisasi perbedaan rata-rata fitur masing-masing cluster
    st.subheader("Heatmap Perbedaan Karakteristik Antar Cluster")
    cluster_profile_raw = df.groupby('cluster')[fitur].mean()
    cluster_profile_norm = (cluster_profile_raw - cluster_profile_raw.min()) / \
                           (cluster_profile_raw.max() - cluster_profile_raw.min())
    cluster_profile_norm.index = [f"{label_map[i]}" for i in cluster_profile_norm.index]
    # Menggunakan label fitur global
    pass

    fig_heat, ax_heat = plt.subplots(figsize=(11, 4.5), facecolor='none')
    sns.heatmap(
        cluster_profile_norm, annot=cluster_profile_raw.round(1), fmt='.1f',
        cmap='GnBu', linewidths=1.5, linecolor='white', ax=ax_heat,
        xticklabels=fitur_label_short,
        cbar_kws={'label': 'Tingkat Relatif Indikator'},
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    ax_heat.set_title('Profil Karakteristik Setiap Cluster (Rata-rata Nilai Aktual)', fontsize=12, fontweight='bold', pad=15)
    ax_heat.set_ylabel('Cluster', fontsize=10, labelpad=10)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close()

elif menu == "Detail & Pencarian":
    st.subheader("Pencarian Provinsi dan Detail Data Cluster")

    filter_cluster = st.selectbox(
        "Pilih Filter Cluster:",
        options=['Semua'] + [f"{label_map[i]} – {label_desc.get(i,'')}" for i in range(k_value)]
    )

    search = st.text_input("Ketik Nama Provinsi:", placeholder="Contoh: Papua, Jawa, Bali...")

    df_show = df[['provinsi', 'label_cluster'] + fitur].copy()
    df_show.columns = ['Provinsi', 'Cluster'] + fitur_label_short

    if filter_cluster != 'Semua':
        df_show = df_show[df_show['Cluster'] == filter_cluster]
    if search:
        df_show = df_show[df_show['Provinsi'].str.contains(search, case=False)]

    st.dataframe(df_show, width='stretch')
    st.caption(f"Menampilkan {len(df_show)} dari {len(df)} provinsi")

    st.markdown("---")
    st.subheader("Statistik Ringkas Per Cluster")
    for c in range(k_value):
        subset = df[df['cluster'] == c]
        with st.expander(f"{label_map[c]} – {label_desc.get(c, '')} ({len(subset)} provinsi)"):
            stat = subset[fitur].describe().round(2)
            stat.columns = fitur_label_short
            st.dataframe(stat, width='stretch')
            st.write("**Daftar Provinsi Anggota:**", ", ".join(subset['provinsi'].values))

elif menu == "Ekspor Data":
    st.subheader("Ekspor Hasil Analisis K-Means")
    st.markdown("Gunakan menu ini untuk mengunduh hasil pembagian cluster provinsi ke dalam format berkas CSV.")

    df_download = df[['provinsi', 'cluster', 'label_cluster'] + fitur].copy()
    df_download.columns = ['Provinsi', 'No Cluster', 'Label Cluster'] + fitur_label_short

    csv = df_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Hasil Analisis (CSV)",
        data=csv,
        file_name="hasil_clustering_pendidikan_indonesia.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("Pratinjau Data (Preview)")
    st.dataframe(df_download, width='stretch')

    st.markdown("---")
    st.subheader("Evaluasi Akhir Model Model K-Means")
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score (Ketepatan)", f"{sil_score_val:.4f}")
    col2.metric("Jumlah Cluster Terpilih (K)", k_value)
    col3.metric("Total Provinsi Teranalisis", len(df))


# Footer halaman utama
st.markdown("""
<div class="footer">
    Analisis Pemerataan Pendidikan Dasar Antar Provinsi di Indonesia Menggunakan K-Means Clustering<br>
    Teknik Informatika 2026 — Dikembangkan oleh Kelompok 2
</div>
""", unsafe_allow_html=True)
