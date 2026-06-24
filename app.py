import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# Configure Matplotlib styling globally for a clean, modern look
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#1E293B'
plt.rcParams['axes.labelcolor'] = '#475569'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.edgecolor'] = '#E2E8F0'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Pemerataan Pendidikan Dasar Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS KUSTOM
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #4F46E5 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 15px 0 5px 0;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #64748B;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* Metric Cards */
    .metric-card-wrapper {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    
    .metric-card-wrapper:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        line-height: 1.1;
    }
    
    /* Cluster Cards */
    .cluster-card {
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid transparent;
        transition: transform 0.2s ease;
    }
    
    .cluster-card:hover {
        transform: translateY(-2px);
    }
    
    .card-cluster-0 {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        border-color: #D1FAE5;
    }
    .card-cluster-1 {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        border-color: #FEF3C7;
    }
    .card-cluster-2 {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        border-color: #FEE2E2;
    }
    
    .card-cluster-generic {
        background-color: #F8FAFC;
        border-left: 6px solid #6366F1;
        border-color: #E2E8F0;
    }
    
    .cluster-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
    }
    
    .cluster-card-count {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 12px;
    }
    
    .cluster-card-members {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.5;
        background: rgba(255, 255, 255, 0.5);
        padding: 10px;
        border-radius: 8px;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
        padding: 25px 0 10px 0;
        border-top: 1px solid #F1F5F9;
        margin-top: 50px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNGSI: LOAD / GENERATE DATASET
# ============================================================
@st.cache_data
def load_data():
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
    return df

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

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 25px; padding: 12px; background: rgba(255, 255, 255, 0.7); border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/NUSAPUTRA_LOGO.png/320px-NUSAPUTRA_LOGO.png" width="140" style="display: block; margin: 0 auto;"/>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Pengaturan Model")
    k_value = st.slider("Jumlah Cluster (K)", min_value=2, max_value=8, value=3, step=1, help="Tentukan jumlah cluster untuk analisis K-Means.")

    st.markdown("### Upload Dataset")
    uploaded = st.file_uploader("Upload file CSV (opsional)", type=["csv"])
    if not uploaded:
        st.info("Info: Data simulasi default 38 Provinsi Indonesia 2023-2024 aktif.")
    else:
        st.success("Data Sukses Terupload")

    st.markdown("### Informasi")
    st.info("**Algoritma:** K-Means Clustering\n\n**Data:** Pendidikan SD Indonesia 2023–2024\n\n**Sumber:** Kemendikbudristek")
    
    st.markdown("""
        <div style="text-align: center; color: #94A3B8; font-size: 0.75rem; margin-top: 30px; border-top: 1px solid #E2E8F0; padding-top: 15px;">
            © 2026 — Universitas Nusaputra<br>Teknik Informatika
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = load_data()

df, X_scaled, X_pca, kmeans, sil_score_val, wcss_list, sil_list, explained, fitur = run_clustering(df_raw, k_value)

# Label cluster
label_map   = {i: f"Cluster {chr(65+i)}" for i in range(k_value)}
label_desc  = {
    0: "Kondisi Baik",
    1: "Kondisi Sedang",
    2: "Perlu Perhatian",
    3: "Grup D", 4: "Grup E", 5: "Grup F", 6: "Grup G", 7: "Grup H"
}
colors_list = ['#10B981', '#F59E0B', '#EF4444', '#6366F1', '#EC4899', '#06B6D4', '#8B5CF6', '#64748B']

df['label_cluster'] = df['cluster'].apply(
    lambda x: f"{label_map[x]} – {label_desc.get(x, 'Grup ' + chr(65+x))}"
)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">Analisis Pemerataan Pendidikan Dasar Antar Provinsi di Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Menggunakan Algoritma K-Means Clustering | Data Kemendikbudristek 2023–2024</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# TAB NAVIGASI
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard Ringkasan",
    "Analisis Elbow",
    "Visualisasi Spasial 2D",
    "Detail dan Pencarian",
    "Ekspor Data"
])

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
with tab1:
    st.subheader("Ringkasan Hasil Clustering")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card-wrapper">
            <div class="metric-title">Total Provinsi</div>
            <div class="metric-value">{len(df)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card-wrapper">
            <div class="metric-title">Jumlah Cluster</div>
            <div class="metric-value">{k_value}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card-wrapper">
            <div class="metric-title">Silhouette Score</div>
            <div class="metric-value">{sil_score_val:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card-wrapper">
            <div class="metric-title">PCA Explained Var</div>
            <div class="metric-value">{sum(explained):.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Distribusi Provinsi per Cluster")

    cols = st.columns(min(k_value, 3))
    for i in range(k_value):
        subset = df[df['cluster'] == i]
        card_class = f'card-cluster-{i}' if i < 3 else 'card-cluster-generic'
        with cols[i % 3]:
            prov_list = "  •  ".join(subset['provinsi'].values)
            st.markdown(f"""
            <div class="cluster-card {card_class}">
                <div class="cluster-card-title">{label_map[i]} — {label_desc.get(i, 'Grup')}</div>
                <div class="cluster-card-count">{len(subset)} Provinsi</div>
                <div class="cluster-card-members"><strong>Anggota:</strong><br>{prov_list}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Profil Rata-rata Indikator per Cluster")
    cluster_profile = df.groupby('cluster')[fitur].mean().round(2)
    cluster_profile.index = [f"{label_map[i]} – {label_desc.get(i,'')}" for i in cluster_profile.index]
    fitur_label = ['Jml Siswa', 'Jml Guru', 'Rasio Siswa/Guru', 'Angka Mengulang',
                   'Angka Putus Sekolah', 'Jml Rombel', 'Kondisi Kelas (%)']
    cluster_profile.columns = fitur_label
    st.dataframe(cluster_profile.style.background_gradient(cmap='RdYlGn', axis=0), use_container_width=True)

# ============================================================
# TAB 2 — ELBOW & EVALUASI
# ============================================================
with tab2:
    st.subheader("Penentuan Jumlah Cluster Optimal")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
        ax.plot(range(2, 11), wcss_list, color='#4F46E5', marker='o', linewidth=2, markersize=8, markerfacecolor='#4F46E5', markeredgecolor='white', markeredgewidth=1.5)
        ax.axvline(x=k_value, color='#EF4444', linestyle='--', linewidth=1.5, label=f'K={k_value} (dipilih)')
        ax.set_title('Elbow Method', fontsize=12, fontweight='bold', pad=15, color='#1E293B')
        ax.set_xlabel('Jumlah Cluster (K)', fontsize=10, labelpad=8)
        ax.set_ylabel('WCSS', fontsize=10, labelpad=8)
        ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')
        ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#CBD5E1')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
        ax.plot(range(2, 11), sil_list, color='#10B981', marker='s', linewidth=2, markersize=8, markerfacecolor='#10B981', markeredgecolor='white', markeredgewidth=1.5)
        ax.axvline(x=k_value, color='#EF4444', linestyle='--', linewidth=1.5, label=f'K={k_value} (dipilih)')
        ax.set_title('Silhouette Score vs K', fontsize=12, fontweight='bold', pad=15, color='#1E293B')
        ax.set_xlabel('Jumlah Cluster (K)', fontsize=10, labelpad=8)
        ax.set_ylabel('Silhouette Score', fontsize=10, labelpad=8)
        ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')
        ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#CBD5E1')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Tabel Evaluasi Semua Nilai K")
    eval_df = pd.DataFrame({
        'K': list(range(2, 11)),
        'WCSS': [round(w, 2) for w in wcss_list],
        'Silhouette Score': [round(s, 4) for s in sil_list]
    })
    eval_df['Status'] = eval_df['K'].apply(lambda x: 'Dipilih' if x == k_value else '')
    st.dataframe(eval_df.style.highlight_max(subset=['Silhouette Score'], color='#c8e6c9'), use_container_width=True)

    st.info(f"""
    **Interpretasi Silhouette Score:**
    - **>= 0.70** -> Sangat Baik
    - **0.50 - 0.69** -> Baik
    - **0.25 - 0.49** -> Cukup
    - **< 0.25** -> Kurang

    Hasil untuk K={k_value}: **{sil_score_val:.4f}** — {'Baik' if sil_score_val >= 0.5 else 'Cukup'}
    """)

# ============================================================
# TAB 3 — VISUALISASI CLUSTER
# ============================================================
with tab3:
    st.subheader("Scatter Plot PCA 2D")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='none')
    for c in range(k_value):
        mask = df['cluster'] == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors_list[c], label=f"{label_map[c]} – {label_desc.get(c, '')}",
                   s=180, alpha=0.9, edgecolors='white', linewidth=1.2)
        for i, row in df[mask].iterrows():
            ax.annotate(row['provinsi'], (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=9, weight='bold', alpha=0.85, 
                        xytext=(5, 5), textcoords='offset points')

    pca_temp = PCA(n_components=2, random_state=42)
    pca_temp.fit(X_scaled)
    centroids_pca = pca_temp.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='#0F172A', marker='X', s=350, zorder=5, label='Centroid Model',
               edgecolors='white', linewidth=1.5)

    ax.set_title('Visualisasi Cluster Provinsi (Analisis Komponen Utama - PCA 2D)',
                 fontsize=15, fontweight='bold', pad=20, color='#1E293B')
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% variance)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% variance)', fontsize=12, labelpad=10)
    ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='#E2E8F0', shadow=True)
    ax.grid(True, linestyle=':', alpha=0.4, color='#CBD5E1')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Heatmap Profil Cluster")

    cluster_profile_raw = df.groupby('cluster')[fitur].mean()
    cluster_profile_norm = (cluster_profile_raw - cluster_profile_raw.min()) / \
                           (cluster_profile_raw.max() - cluster_profile_raw.min())
    cluster_profile_norm.index = [f"{label_map[i]}" for i in cluster_profile_norm.index]
    fitur_label_short = ['Jml Siswa', 'Jml Guru', 'Rasio S/G', 'Mengulang',
                         'Putus Sekolah', 'Rombel', 'Kelas Baik (%)']

    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='none')
    sns.heatmap(cluster_profile_norm, annot=cluster_profile_raw.round(1), fmt='.1f',
                cmap='YlGnBu', linewidths=1.5, linecolor='white', ax=ax,
                xticklabels=fitur_label_short,
                cbar_kws={'label': 'Tingkat Relatif Indikator'},
                annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_title('Profil Karakteristik Setiap Cluster (Rata-rata Nilai Aktual)',
                 fontsize=13, fontweight='bold', pad=20, color='#1E293B')
    ax.set_ylabel('Cluster', fontsize=11, labelpad=10)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Jumlah Provinsi per Cluster")
        counts = df.groupby('label_cluster').size().reset_index(name='Jumlah')
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
        bars = ax.bar(range(len(counts)), counts['Jumlah'],
                      color=colors_list[:len(counts)], edgecolor='none', width=0.45)
        for bar, val in zip(bars, counts['Jumlah']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontweight='bold', color='#1E293B', fontsize=10)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([c.split('–')[0].strip() for c in counts['label_cluster']], rotation=0, fontsize=10)
        ax.set_ylabel('Jumlah Provinsi', fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Proporsi Distribusi Cluster")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
        sizes = df['cluster'].value_counts().sort_index()
        labels = [f"{label_map[i]}\n({v} prov.)" for i, v in sizes.items()]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list[:len(sizes)],
                                          autopct='%1.1f%%', startangle=140, pctdistance=0.75,
                                          wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
        plt.setp(texts, size=9, weight="bold", color="#475569")
        plt.setp(autotexts, size=9, weight="bold", color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# TAB 4 — DETAIL PROVINSI
# ============================================================
with tab4:
    st.subheader("Detail Anggota Setiap Cluster")

    filter_cluster = st.selectbox(
        "Pilih Cluster untuk ditampilkan:",
        options=['Semua'] + [f"{label_map[i]} – {label_desc.get(i,'')}" for i in range(k_value)]
    )

    search = st.text_input("Cari provinsi:", placeholder="Contoh: Papua, Jawa, Bali...")

    df_show = df[['provinsi', 'label_cluster'] + fitur].copy()
    df_show.columns = ['Provinsi', 'Cluster'] + fitur_label_short

    if filter_cluster != 'Semua':
        df_show = df_show[df_show['Cluster'] == filter_cluster]
    if search:
        df_show = df_show[df_show['Provinsi'].str.contains(search, case=False)]

    st.dataframe(df_show.style.background_gradient(subset=fitur_label_short, cmap='RdYlGn'),
                 use_container_width=True)
    st.caption(f"Menampilkan {len(df_show)} dari {len(df)} provinsi")

    st.markdown("---")
    st.subheader("Statistik per Cluster")
    for c in range(k_value):
        subset = df[df['cluster'] == c]
        with st.expander(f"{label_map[c]} – {label_desc.get(c, '')} ({len(subset)} provinsi)"):
            stat = subset[fitur].describe().round(2)
            stat.columns = fitur_label_short
            st.dataframe(stat, use_container_width=True)
            st.write("**Anggota:**", ", ".join(subset['provinsi'].values))

# ============================================================
# TAB 5 — UNDUH HASIL
# ============================================================
with tab5:
    st.subheader("Unduh Hasil Clustering")
    st.markdown("Unduh hasil analisis dalam format CSV untuk keperluan lebih lanjut.")

    df_download = df[['provinsi', 'cluster', 'label_cluster'] + fitur].copy()
    df_download.columns = ['Provinsi', 'No Cluster', 'Label Cluster'] + fitur_label_short

    csv = df_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Hasil Clustering (CSV)",
        data=csv,
        file_name="hasil_clustering_pendidikan_indonesia.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("Preview Data yang Akan Diunduh")
    st.dataframe(df_download, use_container_width=True)

    st.markdown("---")
    st.subheader("Ringkasan Evaluasi Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{sil_score_val:.4f}")
    col2.metric("Jumlah Cluster", k_value)
    col3.metric("Total Provinsi", len(df))

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    Analisis Pemerataan Pendidikan Dasar Antar Provinsi di Indonesia Menggunakan K-Means Clustering<br>
    Program Studi Teknik Informatika — Fakultas Teknik, Komputer, dan Desain — Universitas Nusaputra 2026
</div>
""", unsafe_allow_html=True)
