"""
Clustering Wholesale Customers (UCI ML Repository, id=292)
Algoritma : K-Means, Agglomerative Hierarchical (Ward), DBSCAN
Dataset   : https://archive.ics.uci.edu/dataset/292/wholesale+customers
            440 pelanggan distributor grosir, 6 fitur pengeluaran tahunan.

Cara memuat data:
  - Online   : from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=292)
  - Offline  : letakkan 'wholesale.csv' (Wholesale customers data.csv) di folder yang sama.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

RS = 42
OUT = "output"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25})

# ------------------------------------------------------------------ 1. DATA
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=292)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        if "Channel" not in df.columns:           # jaga-jaga struktur kolom berbeda
            raise ValueError
        return df
    except Exception:
        return pd.read_csv("wholesale.csv")

df = load_data()
FEATURES = [c for c in df.columns if c not in ("Channel", "Region")]   # 6 fitur pengeluaran
print("Shape:", df.shape, "| missing:", int(df.isna().sum().sum()))
print("\nSkewness fitur (asli):\n", df[FEATURES].skew().round(2))

# ------------------------------------------------------------ 2. PRA-PROSES
# Data sangat miring ke kanan (skew 2.5 - 11) -> log1p, lalu standarisasi.
# Channel & Region TIDAK dipakai untuk clustering (dipakai hanya sebagai
# pembanding eksternal setelah cluster terbentuk).
X_log = np.log1p(df[FEATURES])
X = StandardScaler().fit_transform(X_log)
print("\nSkewness setelah log1p:\n", X_log.skew().round(2))

pca = PCA(n_components=2, random_state=RS)
X2 = pca.fit_transform(X)                         # hanya untuk VISUALISASI
ev = pca.explained_variance_ratio_
print(f"\nPCA 2D menjelaskan {ev.sum()*100:.1f}% varians (PC1 {ev[0]*100:.1f}%, PC2 {ev[1]*100:.1f}%)")
lab_x, lab_y = f"PC1 ({ev[0]*100:.1f}%)", f"PC2 ({ev[1]*100:.1f}%)"

def metrics(X, labels):
    """Metrik internal; noise DBSCAN (-1) dikeluarkan saat menghitung."""
    m = labels != -1
    if len(set(labels[m])) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(silhouette=silhouette_score(X[m], labels[m]),
                davies_bouldin=davies_bouldin_score(X[m], labels[m]),
                calinski_harabasz=calinski_harabasz_score(X[m], labels[m]))

# ------------------------------------------------------------- 3. K-MEANS
ks = range(2, 11)
inertia, sil_km, db_km = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=RS).fit(X)
    inertia.append(km.inertia_)
    sil_km.append(silhouette_score(X, km.labels_))
    db_km.append(davies_bouldin_score(X, km.labels_))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="K-Means: Elbow (inertia)", xlabel="k", ylabel="WCSS")
ax[1].plot(ks, sil_km, "o-", c="tab:green"); ax[1].set(title="K-Means: Silhouette", xlabel="k")
ax[2].plot(ks, db_km, "o-", c="tab:red"); ax[2].set(title="K-Means: Davies-Bouldin (kecil=bagus)", xlabel="k")
plt.tight_layout(); plt.savefig(f"{OUT}/00_kmeans_pemilihan_k.png"); plt.close()
print("\nK-Means  k:", list(ks)); print("silhouette:", np.round(sil_km, 3))

K_BEST = list(ks)[int(np.argmax(sil_km))]
km = KMeans(n_clusters=K_BEST, n_init=20, random_state=RS).fit(X)
lab_km = km.labels_
cent2 = pca.transform(km.cluster_centers_)        # centroid diproyeksikan ke PCA

# --------------------------------------------------------- 4. HIERARCHICAL
Z = linkage(X, method="ward")
coph = cophenet(Z, pdist(X))[0]
print(f"\nCophenetic correlation (Ward): {coph:.3f}")
link_cmp = {}
for meth in ["ward", "complete", "average", "single"]:
    Zm = linkage(X, method=meth)
    lab = fcluster(Zm, K_BEST, criterion="maxclust")
    link_cmp[meth] = (cophenet(Zm, pdist(X))[0], silhouette_score(X, lab) if len(set(lab)) > 1 else np.nan)
print("Perbandingan linkage (coph, silhouette) @k=%d:" % K_BEST)
for k_, v in link_cmp.items(): print(f"  {k_:9s} coph={v[0]:.3f}  sil={v[1]:.3f}")

lab_hc = fcluster(Z, K_BEST, criterion="maxclust") - 1
cut_height = (Z[-K_BEST, 2] + Z[-K_BEST + 1, 2]) / 2

fig, ax = plt.subplots(figsize=(13, 5.5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8,
           show_contracted=True, color_threshold=cut_height, ax=ax)
ax.axhline(cut_height, c="k", ls="--", lw=1, label=f"potong pada k={K_BEST}")
ax.set(title="Dendrogram Hierarchical Clustering (Ward, 30 daun terakhir)",
       xlabel="Cluster (angka dalam kurung = jumlah pelanggan)", ylabel="Jarak Ward")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/02_hierarchical_dendrogram.png"); plt.close()

# ------------------------------------------------------------------ 5. DBSCAN
min_pts = 2 * X.shape[1]                          # aturan umum: min_samples = 2 x dimensi
dist, _ = NearestNeighbors(n_neighbors=min_pts).fit(X).kneighbors(X)
kdist = np.sort(dist[:, -1])

# Titik "knee" = titik kurva yang paling jauh dari garis lurus ujung-ke-ujung
n = len(kdist)
xn = np.linspace(0, 1, n)                                   # sumbu dinormalisasi ke [0,1]
yn = (kdist - kdist[0]) / (kdist[-1] - kdist[0])
d_line = (xn - yn) / np.sqrt(2)                             # jarak ke garis diagonal (0,0)-(1,1)
knee_idx = int(np.argmax(d_line)); EPS = round(float(kdist[knee_idx]), 2); MINS = min_pts

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(kdist); ax.axhline(EPS, c="r", ls="--", label=f"knee: eps = {EPS}")
ax.set(title=f"k-distance plot (k={min_pts}) untuk memilih eps", xlabel="titik (diurutkan)",
       ylabel=f"jarak ke tetangga ke-{min_pts}"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/03_dbscan_kdistance.png"); plt.close()

# Sensitivitas parameter: jumlah cluster & persentase noise terhadap eps
rows = []
for mp in [5, MINS]:
    for eps in np.arange(0.5, 3.01, 0.1):
        lb = DBSCAN(eps=eps, min_samples=mp).fit_predict(X)
        ncl = len(set(lb)) - (1 if -1 in lb else 0)
        sil = silhouette_score(X[lb != -1], lb[lb != -1]) if ncl >= 2 else np.nan
        rows.append(dict(eps=round(eps, 2), min_samples=mp, n_cluster=ncl,
                         noise_pct=(lb == -1).mean() * 100, silhouette=sil))
grid = pd.DataFrame(rows); grid.to_csv(f"{OUT}/dbscan_grid.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for mp in [5, MINS]:
    g = grid[grid.min_samples == mp]
    ax[0].plot(g.eps, g.n_cluster, "o-", label=f"min_samples={mp}")
    ax[1].plot(g.eps, g.noise_pct, "o-", label=f"min_samples={mp}")
ax[0].set(title="DBSCAN: jumlah cluster vs eps", xlabel="eps", ylabel="jumlah cluster (tanpa noise)")
ax[1].set(title="DBSCAN: persentase noise vs eps", xlabel="eps", ylabel="% titik noise")
ax[0].legend(); ax[1].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/08_dbscan_sensitivitas.png"); plt.close()

multi = grid[grid.n_cluster >= 2].sort_values("noise_pct")
print("\nKonfigurasi DBSCAN dengan >=2 cluster (noise terkecil):\n", multi.head(4).round(2).to_string(index=False))
print("\nSensitivitas eps (min_samples=%d):" % MINS)
print(grid[(grid.min_samples == MINS) & grid.eps.isin([0.8, 1.0, 1.5, 2.0, 2.5])].round(1).to_string(index=False))

db = DBSCAN(eps=EPS, min_samples=MINS).fit(X)
lab_db = db.labels_
print(f"\nDBSCAN terpilih (knee): eps={EPS}, min_samples={MINS} -> cluster={len(set(lab_db))-(1 if -1 in lab_db else 0)}, noise={(lab_db==-1).sum()} ({(lab_db==-1).mean()*100:.1f}%)")

# Konfigurasi DBSCAN "dipaksa" menghasilkan >=2 cluster (untuk perbandingan)
mrow = multi.iloc[0]; EPS2, MINS2 = float(mrow.eps), int(mrow.min_samples)
db2 = DBSCAN(eps=EPS2, min_samples=MINS2).fit(X); lab_db2 = db2.labels_

# --------------------------------------------------------- 6. PERBANDINGAN
named = [("K-Means", lab_km), ("Hierarchical (Ward)", lab_hc),
         (f"DBSCAN (knee, eps={EPS})", lab_db), (f"DBSCAN (dipaksa >=2 cluster, eps={EPS2}, ms={MINS2})", lab_db2)]
res = {}
for name, lb in named:
    r = metrics(X, lb)
    r["n_cluster"] = len(set(lb)) - (1 if -1 in lb else 0)
    r["noise"] = int((lb == -1).sum())
    r["ukuran"] = dict(zip(*np.unique(lb, return_counts=True)))
    r["ARI_vs_Channel"] = adjusted_rand_score(df["Channel"], lb)   # noise dianggap label sendiri
    res[name] = r
tab = pd.DataFrame(res).T
print("\n=== RINGKASAN METRIK (metrik internal DBSCAN dihitung tanpa noise) ===\n", tab.drop(columns="ukuran").round(3).to_string())
print("\nUkuran cluster:"); [print(" ", n, {int(k): int(v) for k, v in r["ukuran"].items()}) for n, r in res.items()]

# Kemiripan antar-algoritma (ARI) pada SEMUA 440 titik; noise DBSCAN = satu label tersendiri
print("\n=== ARI antar algoritma (semua titik; 1 = identik) ===")
for a_, la, b_, lb in [("K-Means", lab_km, "Hierarchical", lab_hc), ("K-Means", lab_km, "DBSCAN(knee)", lab_db),
                       ("Hierarchical", lab_hc, "DBSCAN(knee)", lab_db), ("K-Means", lab_km, "DBSCAN(dipaksa)", lab_db2)]:
    print(f"  {a_:12s} vs {b_:15s}: ARI = {adjusted_rand_score(la, lb):.3f}")
agree = (lab_km == lab_hc) | (lab_km == 1 - lab_hc)
print(f"  Persentase titik yang diberi cluster sama oleh K-Means & Hierarchical: {max((lab_km==lab_hc).mean(), (lab_km!=lab_hc).mean())*100:.1f}%")

# Uji k=3 (tidak ada patahan elbow yang jelas, jadi dicek juga)
print("\n=== Perbandingan k=2 vs k=3 (K-Means & Ward) ===")
for k in (2, 3):
    kmk = KMeans(n_clusters=k, n_init=20, random_state=RS).fit(X).labels_
    hck = fcluster(Z, k, criterion="maxclust") - 1
    for nm, lb in (("K-Means", kmk), ("Ward", hck)):
        print(f"  k={k} {nm:8s} sil={silhouette_score(X, lb):.3f} DB={davies_bouldin_score(X, lb):.3f} "
              f"CH={calinski_harabasz_score(X, lb):.1f} ARI(Channel)={adjusted_rand_score(df['Channel'], lb):.3f} "
              f"ukuran={np.bincount(lb).tolist()}")

# Mengapa silhouette average/single linkage tampak tinggi? -> cluster singleton/outlier
print("\n=== Ukuran cluster per linkage @k=2 ===")
for meth in ["ward", "complete", "average", "single"]:
    print(f"  {meth:9s}", np.bincount(fcluster(linkage(X, meth), 2, criterion='maxclust'))[1:].tolist())

# Profil cluster (rata-rata pengeluaran asli) per algoritma
for name, lb in [("K-Means", lab_km), ("Hierarchical", lab_hc), ("DBSCAN", lab_db)]:
    prof = df[FEATURES].groupby(lb).mean().round(0)
    prof["n"] = pd.Series(lb).value_counts().sort_index()
    prof["%Retail"] = df.groupby(lb)["Channel"].apply(lambda s: (s == 2).mean() * 100).round(0)
    print(f"\nProfil {name} (rata-rata m.u.):\n", prof.to_string())
    prof.to_csv(f"{OUT}/profil_{name.split()[0].lower()}.csv")

# ------------------------------------------------------------- 7. VISUALISASI
PAL = plt.cm.tab10.colors

# 7a. K-Means scatter + centroid
fig, ax = plt.subplots(figsize=(8, 6))
for c in np.unique(lab_km):
    ax.scatter(*X2[lab_km == c].T, s=28, alpha=.75, color=PAL[c], label=f"Cluster {c} (n={np.sum(lab_km==c)})")
ax.scatter(*cent2.T, marker="X", s=330, c="k", edgecolor="w", linewidth=1.6, label="Centroid", zorder=5)
for i, (x, y) in enumerate(cent2): ax.annotate(f"C{i}", (x, y), xytext=(8, 8), textcoords="offset points", fontweight="bold")
ax.set(title=f"K-Means (k={K_BEST}) - Scatter Plot dengan Centroid (proyeksi PCA)", xlabel=lab_x, ylabel=lab_y)
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/01_kmeans_scatter_centroid.png"); plt.close()

# 7b. DBSCAN scatter: core, border, noise
core = np.zeros(len(X), bool); core[db.core_sample_indices_] = True
fig, ax = plt.subplots(figsize=(8, 6))
for c in sorted(set(lab_db) - {-1}):
    m = lab_db == c
    ax.scatter(*X2[m & core].T, s=40, color=PAL[c % 10], alpha=.85, label=f"Cluster {c} - core (n={np.sum(m)})")
    ax.scatter(*X2[m & ~core].T, s=16, color=PAL[c % 10], alpha=.5, edgecolor="k", linewidth=.3)
nz = lab_db == -1
ax.scatter(*X2[nz].T, marker="x", s=45, c="red", label=f"Noise / outlier (n={nz.sum()})")
ax.set(title=f"DBSCAN (eps={EPS}, min_samples={MINS}) - Density-based Scatter Plot", xlabel=lab_x, ylabel=lab_y)
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/04_dbscan_scatter.png"); plt.close()

# 7b'. DBSCAN dipaksa >=2 cluster (pembanding)
fig, ax = plt.subplots(figsize=(8, 6))
for c in sorted(set(lab_db2) - {-1}):
    ax.scatter(*X2[lab_db2 == c].T, s=34, color=PAL[c % 10], alpha=.9, label=f"Cluster {c} (n={np.sum(lab_db2==c)})")
nz2 = lab_db2 == -1
ax.scatter(*X2[nz2].T, marker="x", s=30, c="red", alpha=.6, label=f"Noise (n={nz2.sum()}, {nz2.mean()*100:.0f}%)")
ax.set(title=f"DBSCAN dipaksa >=2 cluster (eps={EPS2}, min_samples={MINS2})", xlabel=lab_x, ylabel=lab_y)
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/04b_dbscan_scatter_dipaksa.png"); plt.close()

# 7c. Hierarchical scatter
fig, ax = plt.subplots(figsize=(8, 6))
for c in np.unique(lab_hc):
    ax.scatter(*X2[lab_hc == c].T, s=28, alpha=.75, color=PAL[c], label=f"Cluster {c} (n={np.sum(lab_hc==c)})")
ax.set(title=f"Hierarchical Ward (k={K_BEST}) - Scatter Plot (proyeksi PCA)", xlabel=lab_x, ylabel=lab_y)
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/05_hierarchical_scatter.png"); plt.close()

# 7d. Panel perbandingan + referensi Channel
fig, ax = plt.subplots(1, 4, figsize=(22, 5), sharex=True, sharey=True)
ref = df["Channel"].map({1: "Horeca", 2: "Retail"})
for i, (n, c) in enumerate([("Horeca", PAL[0]), ("Retail", PAL[1])]):
    ax[0].scatter(*X2[ref == n].T, s=18, color=c, alpha=.7, label=n)
ax[0].set_title("Referensi: label Channel (tidak dipakai saat clustering)"); ax[0].legend()
for a, (t, lb) in zip(ax[1:], [("K-Means", lab_km), (f"Hierarchical", lab_hc), ("DBSCAN", lab_db)]):
    for c in sorted(set(lb)):
        col = "red" if c == -1 else PAL[c % 10]
        a.scatter(*X2[lb == c].T, s=18, color=col, marker="x" if c == -1 else "o", alpha=.75)
    a.set_title(t)
for a in ax: a.set_xlabel(lab_x)
ax[0].set_ylabel(lab_y)
plt.tight_layout(); plt.savefig(f"{OUT}/06_perbandingan_panel.png"); plt.close()

# 7e. Heatmap profil K-Means (z-score)
prof_z = pd.DataFrame(X, columns=FEATURES).groupby(lab_km).mean()
fig, ax = plt.subplots(figsize=(8, 3.2))
im = ax.imshow(prof_z.values, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=30, ha="right")
ax.set_yticks(range(len(prof_z))); ax.set_yticklabels([f"C{i}" for i in prof_z.index]); ax.grid(False)
for i in range(prof_z.shape[0]):
    for j in range(prof_z.shape[1]): ax.text(j, i, f"{prof_z.values[i,j]:.1f}", ha="center", va="center", fontsize=9)
ax.set_title("Profil centroid K-Means (z-score fitur log)"); plt.colorbar(im)
plt.tight_layout(); plt.savefig(f"{OUT}/07_kmeans_profil_heatmap.png"); plt.close()

tab.drop(columns="ukuran").round(4).to_csv(f"{OUT}/ringkasan_metrik.csv")
print("\nSelesai. Gambar & tabel tersimpan di folder:", OUT)
