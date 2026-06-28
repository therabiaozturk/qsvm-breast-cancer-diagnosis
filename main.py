import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC as Quantum_SVC

print("Veri hazırlanıyor...")

data = load_breast_cancer()

# 250 örnek
X_pca = PCA(n_components=2).fit_transform(data.data[:250])
y = data.target[:250]

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

print("Modeller eğitiliyor...")

klasik_svm = SVC(kernel="linear")
klasik_svm.fit(X_train, y_train)

ozellik_haritasi = ZZFeatureMap(
    feature_dimension=2,
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

print("Karar sınırları hesaplanıyor...")

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

grid_res = 25

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, grid_res),
    np.linspace(y_min, y_max, grid_res)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_klasik = klasik_svm.predict(grid_points).reshape(xx.shape)

print("Kuantum karar sınırı haritalanıyor...")
Z_kuantum = kuantum_svm.predict(grid_points).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].contourf(
    xx,
    yy,
    Z_klasik,
    alpha=0.3,
    cmap=plt.cm.Purples
)

scatter = axes[0].scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap=plt.cm.coolwarm,
    edgecolors="k",
    s=40
)

axes[0].set_title(
    "Klasik SVM Karar Sınırı",
    fontweight="bold"
)

axes[0].set_xlabel("PCA 1")
axes[0].set_ylabel("PCA 2")

axes[1].contourf(
    xx,
    yy,
    Z_kuantum,
    alpha=0.3,
    cmap=plt.cm.GnBu
)

axes[1].scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap=plt.cm.coolwarm,
    edgecolors="k",
    s=40
)

axes[1].set_title(
    "Kuantum SVM (QSVM) Karar Sınırı",
    fontweight="bold"
)

axes[1].set_xlabel("PCA 1")
axes[1].set_ylabel("PCA 2")

axes[0].legend(
    handles=scatter.legend_elements()[0],
    labels=["Malignant", "Benign"],
    loc="lower left"
)

axes[1].legend(
    handles=scatter.legend_elements()[0],
    labels=["Malignant", "Benign"],
    loc="lower left"
)

plt.suptitle(
    "SVM ve QSVM Karar Sınırları",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()