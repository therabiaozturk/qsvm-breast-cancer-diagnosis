import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Qiskit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC as Quantum_SVC

# ---------------------------------------------------
# SAYFA AYARLARI
# ---------------------------------------------------

st.set_page_config(
    page_title="QSVM Kanser Teşhisi",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

gizleme_stili = """
<style>

.stDeployButton {
    display:none;
}

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

.block-container{
    padding-top:1rem !important;
    padding-bottom:0rem !important;
}

[data-testid="stMetric"]{
    background-color:#f8f9fa;
    border:1px solid #e9ecef;
    padding:15px;
    border-radius:12px;
    box-shadow:0 2px 4px rgba(0,0,0,0.05);
}

</style>
"""

st.markdown(gizleme_stili, unsafe_allow_html=True)

#başlık

st.title(" Kuantum Makine Öğrenmesi (QSVM) ile Kanser Teşhisi")

st.markdown("""

Klasik Destek Vektör Makineleri (SVM) ile
Kuantum Destekli Destek Vektör Makinelerinin (QSVM)
meme kanseri verisi üzerindeki performanslarının karşılaştırılması.
""")

# ---------------------------------------------------
# MODEL EĞİTİMİ
# ---------------------------------------------------

@st.cache_resource(show_spinner="Kuantum simülasyonu çalıştırılıyor...")
def egit_ve_test_et():

    data = load_breast_cancer()
    X = data.data[:200]
    y = data.target[:200]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    def ozellikleri_hazirla(X_train_raw, X_test_raw, X_all_raw=None):
        standart_olcekleyici = StandardScaler()
        # PCA=3 için ana model eğitimi
        pca = PCA(
            n_components=3,
            random_state=42
        )
        # PCA=2 görselleştirme için (karar sınırları)
        pca_2d = PCA(
            n_components=2,
            random_state=42
        )
        # Separate scalers for 3D and 2D features
        aci_olcekleyici_3d = MinMaxScaler(
            feature_range=(0, np.pi),
            clip=True
        )
        aci_olcekleyici_2d = MinMaxScaler(
            feature_range=(0, np.pi),
            clip=True
        )

        X_train_scaled = standart_olcekleyici.fit_transform(X_train_raw)
        # PCA=3 işlemek
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_train = aci_olcekleyici_3d.fit_transform(X_train_pca)
        # PCA=2 işlemek (görselleştirme için)
        X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
        X_train_2d = aci_olcekleyici_2d.fit_transform(X_train_pca_2d)

        X_test_scaled = standart_olcekleyici.transform(X_test_raw)
        X_test_pca = pca.transform(X_test_scaled)
        X_test = aci_olcekleyici_3d.transform(X_test_pca)
        X_test_pca_2d = pca_2d.transform(X_test_scaled)
        X_test_2d = aci_olcekleyici_2d.transform(X_test_pca_2d)

        if X_all_raw is None:
            return X_train, X_test, X_train_2d, X_test_2d

        X_all_scaled = standart_olcekleyici.transform(X_all_raw)
        X_all_pca = pca.transform(X_all_scaled)
        X_all = aci_olcekleyici_3d.transform(X_all_pca)
        X_all_pca_2d = pca_2d.transform(X_all_scaled)
        X_all_2d = aci_olcekleyici_2d.transform(X_all_pca_2d)

        return X_train, X_test, X_train_2d, X_test_2d, X_all, X_all_2d

    X_train, X_test, X_train_2d, X_test_2d, X_pca, X_pca_2d = ozellikleri_hazirla(
        X_train_raw,
        X_test_raw,
        X
    )

    klasik_svm = SVC(kernel="linear")
    klasik_svm.fit(X_train, y_train)

    # PCA=3 kullanıldığında karar sınırı görselleştirmeleri bozuluyorsa, 
    # model eğitimi için PCA(3), görselleştirme için PCA(2) kullan
    ozellik_haritasi = ZZFeatureMap(
        feature_dimension=3,
        reps=2,
        entanglement="linear"
    )

    q_kernel = FidelityQuantumKernel(
        feature_map=ozellik_haritasi
    )

    kuantum_svm = Quantum_SVC(
        kernel=q_kernel.evaluate
    )

    kuantum_svm.fit(X_train, y_train)

    # 2D görselleştirme için ayrı modeller
    klasik_svm_2d = SVC(kernel="linear")
    klasik_svm_2d.fit(X_train_2d, y_train)

    ozellik_haritasi_2d = ZZFeatureMap(
        feature_dimension=2,
        reps=2,
        entanglement="linear"
    )
    q_kernel_2d = FidelityQuantumKernel(
        feature_map=ozellik_haritasi_2d
    )
    kuantum_svm_2d = Quantum_SVC(
        kernel=q_kernel_2d.evaluate
    )
    kuantum_svm_2d.fit(X_train_2d, y_train)

    def kuantum_svm_olustur():
        fold_ozellik_haritasi = ZZFeatureMap(
            feature_dimension=3,
            reps=2,
            entanglement="linear"
        )
        fold_q_kernel = FidelityQuantumKernel(
            feature_map=fold_ozellik_haritasi
        )
        return Quantum_SVC(
            kernel=fold_q_kernel.evaluate
        )

    def capraz_dogrulama_skorlari(model_olustur, X_raw, y):
        skorlar = []

        for train_index, test_index in cv.split(X_raw, y):
            X_fold_train, X_fold_test, _, _ = ozellikleri_hazirla(
                X_raw[train_index],
                X_raw[test_index]
            )
            model = model_olustur()
            model.fit(X_fold_train, y[train_index])
            skorlar.append(model.score(X_fold_test, y[test_index]))

        return np.array(skorlar)

    cv_klasik = capraz_dogrulama_skorlari(
        lambda: SVC(kernel="linear"),
        X,
        y
    )

    cv_kuantum = capraz_dogrulama_skorlari(
        kuantum_svm_olustur,
        X,
        y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        klasik_svm,
        kuantum_svm,
        klasik_svm_2d,
        kuantum_svm_2d,
        data.target_names,
        X_pca,
        X_pca_2d,
        X_train_2d,
        X_test_2d,
        y,
        cv_klasik,
        cv_kuantum
    )

# ---------------------------------------------------
# VERİLER
# ---------------------------------------------------

(
    X_train,
    X_test,
    y_train,
    y_test,
    klasik_svm,
    kuantum_svm,
    klasik_svm_2d,
    kuantum_svm_2d,
    target_names,
    X_pca,
    X_pca_2d,
    X_train_2d,
    X_test_2d,
    y,
    cv_klasik,
    cv_kuantum
) = egit_ve_test_et()

klasik_tahmin = klasik_svm.predict(X_test)
kuantum_tahmin = kuantum_svm.predict(X_test)

klasik_acc = accuracy_score(y_test, klasik_tahmin) * 100
kuantum_acc = accuracy_score(y_test, kuantum_tahmin) * 100

# ---------------------------------------------------
# KPI KARTLARI
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Klasik SVM Doğruluk",
        f"%{klasik_acc:.2f}"
    )

with col2:
    st.metric(
        "Kuantum SVM Doğruluk",
        f"%{kuantum_acc:.2f}"
    )

with col3:
    st.metric(
        "Analiz Edilen Boyut",
        "3 (PCA)"
    )

# ---------------------------------------------------
# TABLAR
# ---------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Veri Seti Analizi",
    "Başarı Skorları",
    "Karar Sınırları",
    "Model İstikrarı",
    "Sonuçlar"
])

# ---------------------------------------------------
# TAB 1
# ---------------------------------------------------

with tab1:

    kötü_huylu = np.count_nonzero(y == 0)
    iyi_huylu = np.count_nonzero(y == 1)

    col1, col2 = st.columns([2, 1])

    with col1:

        fig0, ax0 = plt.subplots(figsize=(3.2, 2.2))

        ax0.pie(
            [kötü_huylu, iyi_huylu],
            labels=["Kötü Huylu", "İyi Huylu"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4a148c", "#e8f5e9"],
            pctdistance=0.65,
            labeldistance=1.02,
            textprops={"fontsize": 7},
            wedgeprops={
                "edgecolor": "white",
                "linewidth": 1.5
            }
        )

        ax0.set_title(
            "Veri Seti Sınıf Dağılımı",
            fontsize=12,
            fontweight="bold"
        )

        ax0.axis("equal")
        fig0.tight_layout(pad=0.5)
        st.pyplot(fig0)

    with col2:
        st.metric("Toplam Örnek", len(y))
        st.metric("Kötü Huylu", kötü_huylu)
        st.metric("İyi Huylu", iyi_huylu)

# ---------------------------------------------------
# TAB 2
# ---------------------------------------------------

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Klasik SVM")
        st.metric(
            "Confusion Matrix Doğruluk",
            f"%{klasik_acc:.2f}"
        )

        fig1, ax1 = plt.subplots(figsize=(5, 4))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            klasik_tahmin,
            display_labels=target_names,
            cmap="Purples",
            ax=ax1,
            colorbar=False
        )

        st.pyplot(fig1)

    with col2:

        st.subheader("Kuantum SVM (QSVM)")
        st.metric(
            "Confusion Matrix Doğruluk",
            f"%{kuantum_acc:.2f}"
        )

        fig2, ax2 = plt.subplots(figsize=(5, 4))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            kuantum_tahmin,
            display_labels=target_names,
            cmap="GnBu",
            ax=ax2,
            colorbar=False
        )

        st.pyplot(fig2)

# ---------------------------------------------------
# TAB 3
# ---------------------------------------------------

with tab3:

    # Karar sınırları için PCA=2 ile eğitilmiş ayrı görselleştirme modellerini kullan
    # Ana performans modelleri PCA=3 ile eğitildiği için bu grafikler açıklayıcı 2D temsildir
    x_min, x_max = X_pca_2d[:, 0].min() - 1, X_pca_2d[:, 0].max() + 1
    y_min, y_max = X_pca_2d[:, 1].min() - 1, X_pca_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 10),
        np.linspace(y_min, y_max, 10)
    )

    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

    with st.spinner("Karar sınırları hesaplanıyor (PCA=2 ile eğitilmiş görselleştirme modelleri)..."):
        # 2D görselleştirme modelleri ile karar sınırlarını hesaplıyoruz
        Z_klasik = klasik_svm_2d.predict(grid_points_2d).reshape(xx.shape)
        Z_kuantum = kuantum_svm_2d.predict(grid_points_2d).reshape(xx.shape)

    fig3, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    sinif_etiketleri = ["Malignant", "Benign"]
    sinif_renkleri = ["#3b4cc0", "#b40426"]

    axes[0].contourf(
        xx,
        yy,
        Z_klasik,
        alpha=0.3,
        cmap=plt.cm.Purples
    )

    for sinif_degeri, sinif_etiketi in enumerate(sinif_etiketleri):
        sinif_maskesi = y_train == sinif_degeri
        axes[0].scatter(
            X_train_2d[sinif_maskesi, 0],
            X_train_2d[sinif_maskesi, 1],
            color=sinif_renkleri[sinif_degeri],
            edgecolors="k",
            label=sinif_etiketi
        )

    axes[0].set_title(
        "Klasik SVM",
        fontweight="bold"
    )

    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2 (Görselleştirme)")
    axes[0].legend()

    axes[1].contourf(
        xx,
        yy,
        Z_kuantum,
        alpha=0.3,
        cmap=plt.cm.GnBu
    )

    for sinif_degeri, sinif_etiketi in enumerate(sinif_etiketleri):
        sinif_maskesi = y_train == sinif_degeri
        axes[1].scatter(
            X_train_2d[sinif_maskesi, 0],
            X_train_2d[sinif_maskesi, 1],
            color=sinif_renkleri[sinif_degeri],
            edgecolors="k",
            label=sinif_etiketi
        )

    axes[1].set_title(
        "Kuantum SVM (QSVM)",
        fontweight="bold"
    )

    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2 (Görselleştirme)")
    axes[1].legend()

    plt.tight_layout()

    st.pyplot(fig3)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:16px; line-height:1.7; color:#1f2937; padding:14px 16px; border:1px solid #d1d5db; border-radius:12px; background:#f8fafc;'>"
        "<strong>Karar sınırları yorumu:</strong> Ana performans modelleri PCA=3 ile eğitilmiştir. Karar sınırı grafikleri ise 2 boyutlu düzlemde gösterilebilmesi için PCA=2 verisiyle ayrıca eğitilen görselleştirme modellerinden elde edilmektedir. Klasik SVM daha düzgün ve net karar bölgeleri oluştururken, QSVM daha karmaşık karar yüzeyleri üretmektedir."
        "</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# TAB 4
# ---------------------------------------------------

with tab4:

    col_plot, col_summary = st.columns([7, 3], gap="small")

    fig4, ax4 = plt.subplots(figsize=(8, 4))

    ax4.boxplot(
        [cv_klasik * 100, cv_kuantum * 100],
        tick_labels=[
            "Klasik SVM",
            "QSVM"
        ],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.2),
        medianprops=dict(color="#1d4ed8", linewidth=2),
        whiskerprops=dict(color="#2563eb", linewidth=1),
        capprops=dict(color="#2563eb", linewidth=1)
    )

    ax4.set_ylabel(
        "Doğruluk (%)",
        fontweight="bold",
        fontsize=11
    )

    ax4.tick_params(axis="x", labelsize=11)
    ax4.tick_params(axis="y", labelsize=11)

    ax4.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    fig4.tight_layout(pad=0.8)

    with col_plot:
        st.pyplot(fig4)

    with col_summary:
        klasik_fold_skorlari = ", ".join(
            f"Fold {i + 1}: %{skor * 100:.1f}"
            for i, skor in enumerate(cv_klasik)
        )
        kuantum_fold_skorlari = ", ".join(
            f"Fold {i + 1}: %{skor * 100:.1f}"
            for i, skor in enumerate(cv_kuantum)
        )

        st.markdown(
            f"""
<div style='padding:18px; border-radius:14px; border:1px solid #d1d5db; background:#f8fafc;'>
  <h4 style='margin:0 0 12px; font-size:18px; color:#111827;'>Model İstikrarı Özeti</h4>
  <div style='font-size:14px; color:#1f2937; line-height:1.6;'>
    <strong>Ort. Klasik SVM Doğruluğu:</strong> %{cv_klasik.mean() * 100:.1f}<br>
    <strong>Ort. QSVM Doğruluğu:</strong> %{cv_kuantum.mean() * 100:.1f}<br>
    <strong>Klasik SVM Fold Skorları:</strong> {klasik_fold_skorlari}<br>
    <strong>QSVM Fold Skorları:</strong> {kuantum_fold_skorlari}<br>
  </div>
  <div style='margin-top:14px; font-size:14px; color:#111827; line-height:1.6;'>
    Boxplot, her iki modelin çapraz doğrulama doğruluk dağılımını gösterir. Klasik SVM daha dar bir dağılım ve daha yüksek tutarlılık sergilerken, QSVM daha geniş bir doğruluk aralığına sahiptir.
  </div>
</div>
""",
            unsafe_allow_html=True
        )

# ---------------------------------------------------
# TAB 5
# ---------------------------------------------------

with tab5:

    st.subheader("Sonuçlar")

    summary_style = "border:1px solid #d1d5db; border-radius:14px; background:#f8fafc; padding:18px;"
    summary_title = "font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;"
    summary_item = "font-size:14px; color:#1f2937; line-height:1.8; margin:0 0 8px 0;"
    comparison_style = "border:1px solid #d1d5db; border-radius:14px; background:#ffffff; padding:16px;"
    comparison_title = "font-size:16px; font-weight:700; color:#111827; margin-bottom:10px;"
    comparison_line = "font-size:14px; color:#1f2937; line-height:1.7; margin:0 0 8px 0;"

    left_col, right_col = st.columns([3, 1.4], gap="medium")

    with left_col:
        st.markdown(
            f"""
<div style='{summary_style}'>
  <div style='{summary_title}'>Genel Sonuç</div>
  <div style='{summary_item}'>Bu çalışmada meme kanseri veri seti üzerinde Klasik SVM ve Kuantum Destekli SVM (QSVM) modelleri karşılaştırılmıştır. PCA ile özellik boyutu 3'e indirgenmiştir. Deney sonuçlarına göre Klasik SVM %{klasik_acc:.1f} doğruluk elde ederken, QSVM %{kuantum_acc:.1f} doğruluk göstermiştir. Her iki model de başarılı sonuçlar üretmiş olsa da bu veri setinde Klasik SVM daha yüksek performans ve kararlılık sergilemiştir. QSVM ise daha karmaşık karar sınırları oluşturarak alternatif bir sınıflandırma yaklaşımı sunmuştur.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            f"""
<div style='{comparison_style}'>
  <div style='{comparison_title}'>Karşılaştırma Özeti</div>
  <div style='{comparison_line}'>Klasik SVM → %{klasik_acc:.1f}</div>
  <div style='{comparison_line}'>QSVM → %{kuantum_acc:.1f}</div>
</div>
""",
            unsafe_allow_html=True,
        )
