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

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Pemerataan Pendidikan Dasar Indonesia",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS KUSTOM
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        padding: 10px 0 5px 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #546e7a;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 18px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h3 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.85rem; margin: 4px 0 0 0; opacity: 0.9; }
    .cluster-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .card-blue   { background-color: #e3f2fd; border-left: 5px solid #2196F3; }
    .card-green  { background-color: #e8f5e9; border-left: 5px solid #4CAF50; }
    .card-orange { background-color: #fff3e0; border-left: 5px solid #FF5722; }
    .footer {
        text-align: center;
        color: #90a4ae;
        font-size: 0.8rem;
        padding: 20px 0 5px 0;
        border-top: 1px solid #eceff1;
        margin-top: 30px;
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/NUSAPUTRA_LOGO.png/320px-NUSAPUTRA_LOGO.png",
             width=160)
    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan")

    k_value = st.slider("Jumlah Cluster (K)", min_value=2, max_value=8, value=3, step=1)

    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload file CSV (opsional)", type=["csv"])
    st.caption("Jika tidak diupload, data simulasi akan digunakan.")

    st.markdown("---")
    st.markdown("### 📌 Informasi")
    st.info("**Algoritma:** K-Means Clustering\n\n**Data:** Pendidikan SD Indonesia 2023–2024\n\n**Sumber:** Kemendikbudristek")
    st.markdown("---")
    st.caption("© 2026 — Universitas Nusaputra\nTeknik Informatika")

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
colors_list = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800', '#00BCD4', '#795548', '#607D8B']

df['label_cluster'] = df['cluster'].apply(
    lambda x: f"{label_map[x]} – {label_desc.get(x, 'Grup ' + chr(65+x))}"
)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">📚 Analisis Pemerataan Pendidikan Dasar Antar Provinsi di Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Menggunakan Algoritma K-Means Clustering | Data Kemendikbudristek 2023–2024</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# TAB NAVIGASI
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard",
    "📊 Elbow & Evaluasi",
    "🗺️ Visualisasi Cluster",
    "📋 Detail Provinsi",
    "📥 Unduh Hasil"
])

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
with tab1:
    st.subheader("Ringkasan Hasil Clustering")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>{len(df)}</h3><p>Total Provinsi</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h3>{k_value}</h3><p>Jumlah Cluster</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h3>{sil_score_val:.3f}</h3><p>Silhouette Score</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <h3>{sum(explained):.1f}%</h3><p>PCA Variance Explained</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Distribusi Provinsi per Cluster")

    card_class = ['card-blue', 'card-green', 'card-orange']
    icons = ['🔵', '🟢', '🔴', '🟣', '🟠', '🔵', '🟤', '⚫']

    cols = st.columns(min(k_value, 3))
    for i in range(k_value):
        subset = df[df['cluster'] == i]
        card_c = card_class[i] if i < 3 else 'card-blue'
        with cols[i % 3]:
            prov_list = "  •  ".join(subset['provinsi'].values)
            st.markdown(f"""
            <div class="cluster-card {card_c}">
                <strong>{icons[i]} {label_map[i]} — {label_desc.get(i, 'Grup')}</strong><br>
                <span style="font-size:1.4rem; font-weight:700">{len(subset)} Provinsi</span><br><br>
                <span style="font-size:0.78rem; color:#555">{prov_list}</span>
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
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(2, 11), wcss_list, 'bo-', linewidth=2, markersize=8)
        ax.axvline(x=k_value, color='red', linestyle='--', linewidth=1.5, label=f'K={k_value} (dipilih)')
        ax.set_title('Elbow Method', fontsize=12, fontweight='bold')
        ax.set_xlabel('Jumlah Cluster (K)')
        ax.set_ylabel('WCSS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(2, 11), sil_list, 'gs-', linewidth=2, markersize=8)
        ax.axvline(x=k_value, color='red', linestyle='--', linewidth=1.5, label=f'K={k_value} (dipilih)')
        ax.set_title('Silhouette Score vs K', fontsize=12, fontweight='bold')
        ax.set_xlabel('Jumlah Cluster (K)')
        ax.set_ylabel('Silhouette Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📊 Tabel Evaluasi Semua Nilai K")
    eval_df = pd.DataFrame({
        'K': list(range(2, 11)),
        'WCSS': [round(w, 2) for w in wcss_list],
        'Silhouette Score': [round(s, 4) for s in sil_list]
    })
    eval_df['Status'] = eval_df['K'].apply(lambda x: '✅ Dipilih' if x == k_value else '')
    st.dataframe(eval_df.style.highlight_max(subset=['Silhouette Score'], color='#c8e6c9'), use_container_width=True)

    st.info(f"""
    **Interpretasi Silhouette Score:**
    - **≥ 0.70** → Sangat Baik
    - **0.50 – 0.69** → Baik
    - **0.25 – 0.49** → Cukup
    - **< 0.25** → Kurang

    Hasil untuk K={k_value}: **{sil_score_val:.4f}** — {'Baik ✅' if sil_score_val >= 0.5 else 'Cukup ⚠️'}
    """)

# ============================================================
# TAB 3 — VISUALISASI CLUSTER
# ============================================================
with tab3:
    st.subheader("Scatter Plot PCA 2D")

    fig, ax = plt.subplots(figsize=(12, 8))
    for c in range(k_value):
        mask = df['cluster'] == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors_list[c], label=f"{label_map[c]} – {label_desc.get(c, '')}",
                   s=130, alpha=0.85, edgecolors='white', linewidth=0.8)
        for i, row in df[mask].iterrows():
            ax.annotate(row['provinsi'], (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=7.5, alpha=0.8, xytext=(4, 4), textcoords='offset points')

    pca_temp = PCA(n_components=2, random_state=42)
    pca_temp.fit(X_scaled)
    centroids_pca = pca_temp.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='black', marker='X', s=250, zorder=5, label='Centroid')

    ax.set_title('Hasil K-Means Clustering — 38 Provinsi Indonesia\nBerdasarkan Indikator Pendidikan Dasar',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% variance)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
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

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(cluster_profile_norm, annot=cluster_profile_raw.round(1), fmt='g',
                cmap='RdYlGn', linewidths=0.5, ax=ax,
                xticklabels=fitur_label_short,
                cbar_kws={'label': 'Nilai Relatif'})
    ax.set_title('Profil Karakteristik Setiap Cluster\n(Nilai dalam kotak = rata-rata aktual)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Jumlah Provinsi per Cluster")
        counts = df.groupby('label_cluster').size().reset_index(name='Jumlah')
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(range(len(counts)), counts['Jumlah'],
                      color=colors_list[:len(counts)], edgecolor='white', width=0.5)
        for bar, val in zip(bars, counts['Jumlah']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', fontweight='bold')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([c.split('–')[0].strip() for c in counts['label_cluster']], rotation=15)
        ax.set_ylabel('Jumlah Provinsi')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Proporsi Cluster (Pie Chart)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sizes = df['cluster'].value_counts().sort_index()
        labels = [f"{label_map[i]}\n({v} prov.)" for i, v in sizes.items()]
        ax.pie(sizes, labels=labels, colors=colors_list[:len(sizes)],
               autopct='%1.1f%%', startangle=140, pctdistance=0.75,
               wedgeprops=dict(edgecolor='white', linewidth=1.5))
        ax.set_title('Proporsi Cluster', fontweight='bold')
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

    search = st.text_input("🔍 Cari provinsi:", placeholder="Contoh: Papua, Jawa, Bali...")

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
        with st.expander(f"{icons[c]} {label_map[c]} – {label_desc.get(c, '')} ({len(subset)} provinsi)"):
            stat = subset[fitur].describe().round(2)
            stat.columns = fitur_label_short
            st.dataframe(stat, use_container_width=True)
            st.write("**Anggota:**", ", ".join(subset['provinsi'].values))

# ============================================================
# TAB 5 — UNDUH HASIL
# ============================================================
with tab5:
    st.subheader("📥 Unduh Hasil Clustering")
    st.markdown("Unduh hasil analisis dalam format CSV untuk keperluan lebih lanjut.")

    df_download = df[['provinsi', 'cluster', 'label_cluster'] + fitur].copy()
    df_download.columns = ['Provinsi', 'No Cluster', 'Label Cluster'] + fitur_label_short

    csv = df_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Hasil Clustering (CSV)",
        data=csv,
        file_name="hasil_clustering_pendidikan_indonesia.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("📋 Preview Data yang Akan Diunduh")
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
