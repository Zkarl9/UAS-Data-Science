
# 📘 Judul Proyek
*(Klasifikasi Iklan Internet Menggunakan Model Machine Learning dan Deep Learning)*

## 👤 Informasi
- **Nama:** [Haikal Azzrial Akbar]  
- **Repo:** [https://github.com/Zkarl9/UAS-Data-Science.git]  
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk membangun dan mengevaluasi model klasifikasi untuk mendeteksi iklan dalam sebuah dataset yang kompleks. Langkah-langkah utama meliputi:
- Mengidentifikasi dan menyelesaikan permasalahan deteksi iklan pada dataset dengan tantangan *class imbalance*, *missing values*, dan *outliers*.
- Melakukan *data preparation* ekstensif (pembersihan, transformasi, dan pemisahan data).
- Membangun tiga jenis model: **Baseline** (Logistic Regression), **Advanced** (XGBoost Classifier), dan **Deep Learning** (Multilayer Perceptron - MLP).
- Melakukan evaluasi komprehensif menggunakan metrik yang relevan untuk data tidak seimbang dan menentukan model terbaik.

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Dataset `ad.data` memiliki *missing values*, *duplicate rows*, dan *outliers* yang signifikan, yang dapat memengaruhi kinerja model.
- Dataset mengalami *class imbalance* yang parah (rasio nonad/ad sekitar 6.14:1), sehingga membutuhkan strategi penanganan khusus untuk menghindari bias.
- Tingginya dimensi fitur (1558 fitur) berpotensi menyebabkan masalah *overfitting* dan meningkatkan biaya komputasi, memerlukan seleksi fitur.

**Goals:**  
- Mengidentifikasi dan mengatasi masalah kualitas data (seperti *missing values*, duplikat, *outliers*) melalui teknik *data cleaning* dan *preprocessing*.
- Membangun model klasifikasi biner yang efektif untuk membedakan antara konten iklan dan non-iklan.
- Melakukan evaluasi kinerja model secara komprehensif menggunakan metrik yang relevan untuk data tidak seimbang.
- Menentukan model terbaik dan memberikan rekomendasi yang kuat untuk implementasi di dunia nyata.

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb    # Notebook utama proyek
│
├── src/                    # Source code (jika ada script terpisah)
│   
├── models/                 # Saved trained models
│   ├── model_baseline.pkl  # Logistic Regression model
│   ├── model_xgb.pkl       # XGBoost Classifier model
│   └── model_mlp.h5        # Multilayer Perceptron model
│
├── images/                 # Visualizations (plots, confusion matrices, etc.)
│   ├── eda_class_distribution.png
│   ├── eda_continuous_boxplot.png
│   ├── eda_top10_binary_features.png
│   ├── mlp_training_history.png
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_xgb.png
│   ├── feature_importance_xgb.png
│   ├── confusion_matrix_mlp.png
│   └── model_comparison_metrics.png
│
├── requirements.txt        # Python package dependencies
├── .gitignore              # Files/directories to ignore in Git
└── README.md               # Project documentation (this file)
```
---

# 3. 📊 Dataset
- **Sumber:** `ad.data` (Dataset ini berasal dari UCI Machine Learning Repository)
- **Jumlah Data:** 3279 baris (awal) / 2419 baris (setelah *cleaning* dan penghapusan duplikat)
- **Tipe:** Data tabular campuran dengan 3 fitur kontinu dan 1555 fitur biner.

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
| `F0` | Tinggi objek yang diukur (piksel). |
| `F1` | Lebar objek yang diukur (piksel). |
| `F2` | Rasio aspek (`F1`/`F0`). |
| `F3-F1557` | Fitur biner yang merepresentasikan keberadaan kata kunci, URL, atau karakteristik piksel tertentu. |
| `target` | Variabel target yang menunjukkan apakah objek adalah iklan (`ad.`) atau bukan (`nonad.`). |

---

# 4. 🔧 Data Preparation
- **Cleaning (Missing Values, Duplicates, Outliers):**
    - *Missing values* pada fitur kontinu (`F0`, `F1`, `F2`) diimputasi menggunakan nilai median.
    - *Missing values* pada fitur biner (`F3-F1557`) diisi dengan nilai 0.
    - Sebanyak 860 baris duplikat dihapus dari dataset.
    - *Outliers* pada fitur kontinu (`F0`, `F1`, `F2`) ditangani menggunakan metode *winsorisasi* (capping pada persentil 1% dan 99%).
- **Transformasi (Scaling):**
    - Fitur-fitur diskalakan menggunakan `StandardScaler` untuk memastikan kontribusi yang seimbang dari setiap fitur terhadap model.
- **Splitting (Train/Test Split & Feature Selection):**
    - Data dibagi menjadi *training set* (80%) dan *testing set* (20%) menggunakan *stratified sampling* untuk menjaga distribusi kelas target.
    - Dilakukan *feature selection* menggunakan `SelectKBest` dengan `f_classif` untuk memilih 200 fitur terbaik, mengurangi dimensi dan meningkatkan efisiensi model.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** **Logistic Regression**
    - Model linier sederhana sebagai titik referensi.
    - Menggunakan `class_weight='balanced'` untuk menangani *class imbalance*.
- **Model 2 – Advanced ML:** **XGBoost Classifier**
    - Model *ensemble* berbasis *gradient boosting* yang dikenal performa tinggi.
    - Menggunakan `scale_pos_weight` untuk menangani *class imbalance*.
- **Model 3 – Deep Learning:** **Multilayer Perceptron (MLP)**
    - Jaringan saraf dasar dengan arsitektur sequential.
    - Terdiri dari *dense layers* dan *dropout* untuk menangkap pola non-linier dan mencegah *overfitting*.

---

# 6. 🧪 Evaluation
**Metrik:** Model dievaluasi menggunakan Accuracy, Precision, Recall, F1-Score, dan ROC-AUC, yang relevan untuk data klasifikasi biner dengan *class imbalance*.

### Hasil Singkat
| Model                  | Accuracy | Precision | Recall | F1-Score | AUC    |
| :--------------------- | :------- | :-------- | :----- | :------- | :----- |
| Baseline (Logistic Regression) | 0.9401   | 0.8158    | 0.8052 | 0.8105   | 0.9237 |
| Advanced (XGBoost Classifier) | 0.9525   | 0.8750    | 0.8182 | 0.8456   | 0.9608 |
| Deep Learning (MLP Model) | 0.9525   | 0.9091    | 0.7792 | 0.8392   | 0.9458 |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** **XGBoost Classifier**
- **Alasan:** XGBoost Classifier menunjukkan kinerja terbaik secara keseluruhan, memimpin pada F1-Score (0.8456) dan AUC (0.9608), serta memiliki *recall* yang kuat (0.8182). Ini mengindikasikan keseimbangan yang optimal antara mengidentifikasi iklan secara akurat dan meminimalkan iklan yang terlewat.
- **Insight penting:** Penanganan *class imbalance* dan kualitas data yang buruk (melalui *data cleaning* dan *feature engineering*) sangat krusial. Model *ensemble* seperti XGBoost terbukti sangat efektif untuk data tabular yang kompleks, mengungguli model *baseline* dan sedikit lebih baik dari model *deep learning* dalam konteks ini.

---

# 8. 🔮 Future Work
Berikut adalah beberapa saran pengembangan untuk proyek selanjutnya:

### Data:
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [x] Feature engineering lebih lanjut (misalnya, membuat fitur interaksi, fitur berbasis domain)

### Model:
- [ ] Mencoba arsitektur DL yang lebih kompleks (misalnya, Transformer, Conv1D jika relevan)
- [x] Hyperparameter tuning lebih ekstensif (menggunakan GridSearch/RandomSearch dengan cross-validation untuk XGBoost dan MLP)
- [x] Ensemble methods (misalnya, stacking, voting, atau blending model-model terbaik)
- [ ] Transfer learning dengan model yang lebih besar (jika ada data yang memungkinkan pre-training)

### Deployment:
- [ ] Membuat API (Flask/FastAPI) untuk melayani prediksi model
- [ ] Membuat web application (Streamlit/Gradio) sebagai antarmuka pengguna
- [ ] Containerization dengan Docker untuk portabilitas dan konsistensi lingkungan
- [ ] Deploy ke cloud (Heroku, GCP, AWS) agar dapat diakses publik

### Optimization:
- [ ] Model compression (pruning, quantization) untuk model DL
- [ ] Improving inference speed untuk penggunaan real-time
- [ ] Reducing model size untuk deployment pada perangkat dengan sumber daya terbatas

---

# 9. 🔁 Reproducibility
### 9.1 GitHub Repository
Link Repository: [URL GitHub Anda]

Repository harus berisi:
*   ✅ Notebook Jupyter/Colab dengan hasil running
*   ✅ Script Python (jika ada)
*   ✅ requirements.txt atau environment.yml
*   ✅ README.md yang informatif
*   ✅ Folder structure yang terorganisir
*   ✅ .gitignore (jangan upload dataset besar)

### 9.2 Environment & Dependencies
Python Version: 3.10

Main Libraries & Versions:
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tensorflow==2.14.0
xgboost==1.7.6
```
