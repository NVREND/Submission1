# Laporan Proyek Machine Learning - Endritha Pramudya

## Project Domain

Penyakit jantung juga dikenal sebagai penyakit kardiovaskular yang merupakan salah satu penyebab kematian utama secara global. Menurut WHO, penyakit kardiovaskular merenggut sekitar 17,9 juta jiwa setiap tahunnya. Penyakit kardiovaskular adalah sekelompok kelainan jantung dan pembuluh darah yang meliputi penyakit jantung koroner, penyakit serebrovaskular, penyakit jantung rematik, dan kondisi lainnya. Mengidentifikasi penyakit jantung merupakan suatu tantangan karena berbagai faktor risiko yang berkontribusi, antara lain diabetes, tekanan darah tinggi, kolesterol tinggi, denyut nadi tidak normal, dan beberapa faktor lainnya [2].   Seringkali, tidak ada gejala penyakit yang mendasari pembuluh darah. Banyak orang yang tidak menyadari risikonya hingga kondisinya menjadi serius.   Seiring dengan berkembangnya kemampuan komputasi dan pemrosesan data, teknologi dapat digunakan untuk menganalisis data dalam jumlah besar yang sebelumnya sulit diproses secara manual. Dengan menggunakan algoritma pembelajaran mesin, data medis yang besar dan kompleks dapat dianalisis untuk mengidentifikasi pola dan hubungan yang tidak mudah terlihat, sehingga memungkinkan prediksi risiko penyakit jantung yang lebih akurat. 

Penelitian sebelumnya yang dilakukan oleh (Haganta Depari dkk., 2022) menggunakan kumpulan data pasien penyakit jantung yang dikenal dengan 'Personal Key Indicators of Heart Disease' dan menerapkan algoritma klasifikasi Decision Tree, Naive Bayes, dan Random Forest. Penelitian ini bertujuan untuk mengolah dan menganalisis data, serta menerapkan metode tersebut pada klasifikasi penyakit jantung. Hasil evaluasi kinerja menunjukkan akurasi metode Decision Tree sebesar 71%, Naive Bayes sebesar 72%, dan Random Forest sebesar 75%, dengan Random Forest menjadi metode terbaik untuk mengklasifikasikan penyakit jantung berdasarkan dataset yang digunakan [1]. Penelitian yang dilakukan (Putri et al., 2024) menunjukkan keberhasilan algoritma Random Forest dalam menangani kompleksitas data dan mencegah overfitting, dengan akurasi hingga 87.7% di data pengujian menggunakan dataset Heart Failure Prediction Dataset[3]. 

Model pembelajaran mesin, seperti Random Forest, dapat dilatih menggunakan data historis pasien untuk mengenali faktor risiko yang terkait dengan penyakit jantung, seperti pola tekanan darah, kadar kolesterol, dan riwayat kesehatan. Dengan teknik klasifikasi, model ini mampu memberikan prediksi apakah seseorang berisiko tinggi mengalami penyakit jantung, serta mengidentifikasi faktor risiko yang mungkin tidak langsung terlihat. Hasil yang diperoleh dari model pembelajaran mesin ini sangat berguna bagi para profesional medis untuk melakukan intervensi lebih awal dan memberikan rekomendasi pengobatan yang lebih tepat. Dengan demikian, penerapan pembelajaran mesin dapat meningkatkan akurasi deteksi dini penyakit jantung dan membantu mengurangi kematian terkait penyakit ini melalui pencegahan yang lebih efektif.

## Business Understanding

### Problem Statements

- Bagaimana penerapan algoritma machine learning Random Forest, dapat meningkatkan akurasi dalam mendeteksi dan memprediksi risiko penyakit jantung pada pasien?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Menganalisis performa algoritma machine learning Random Forest, dalam klasifikasi penyakit jantung

## Data Understanding
Project ini menggunakan dataset yang tersedia secara publik di Kaggle: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. Dataset ini berisi informasi tentang pasien gagal jantung dan dimaksudkan untuk digunakan dalam tugas pemodelan prediktif. Dataset terdiri dari 918 data.

Dataset ini dibuat dengan menggabungkan berbagai dataset yang sudah tersedia secara mandiri namun belum digabungkan sebelumnya. Dalam kumpulan data ini, 5 kumpulan data jantung digabungkan dalam 11 fitur umum yang menjadikannya kumpulan data penyakit jantung terbesar yang tersedia sejauh ini untuk tujuan penelitian.  Lima kumpulan data yang digunakan untuk kurasinya adalah:
- Cleveland: 303 observations
- Hungarian: 294 observations
- Switzerland: 123 observations
- Long Beach VA: 200 observations
- Stalog (Heart) Data Set: 270 observations
  
Total: 1190 observations

Duplicated: 272 observations

Final dataset: 918 observations

### Variabel-variabel pada heart failur prediction dataset adalah sebagai berikut:
- Age: usia pasien [tahun]
- Sex: jenis kelamin pasien [M: Pria, F: Wanita]
- ChestPainType: jenis nyeri dada [TA: Typical Angina, ATA: Atypical Angina, NAP: -Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: tekanan darah istirahat [mm Hg]
- Cholesterol: kolesterol serum [mm/dl]
- FastingBS: gula darah puasa [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: hasil elektrokardiogram istirahat [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: detak jantung maksimum tercapai [Numeric value between 60 and 202]
- ExerciseAngina: angina akibat olahraga [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: kemiringan puncak latihan segmen ST [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: kelas keluaran [1: heart disease, 0: Normal]
--------------------------------------------------------------------------------------------------------------------------------------
- Terdapat 5 kolom dengan object types, namely: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope 
- Terdapat 6 kolom numeric dengan type int64, namely: Age, RestigBP, Cholesterol, FastingBS, MaxHR, HeartDisease
- Terdapat 1 kolom numeric dengan type float64, namely Oldpeak

  
## Data Preparation

Dalam data preparation dilakukan pembagian data atau _spliting data, oversampling, dan Standarization_. 
- Encoding
  - Encoding adalah proses mengubah data kategorikal (data yang memiliki kategori atau label seperti "warna", "jenis kelamin", atau "status pernikahan") menjadi format numerik yang dapat dimengerti oleh algoritma machine learning. Encoding membantu mengonversi data tersebut ke dalam bentuk numerik tanpa mengurangi informasi yang terkandung di dalamnya.
  - 
- Pembagian data
  - dilakukan dengan membagi dataset menjadi 75% data pelatihan dan 25% data uji dengan masing masing 516 data pelatihan dan 230 data uji.
- Standarization
   - Standardization adalah teknik untuk mengubah data agar memiliki distribusi dengan rata-rata 0 (mean = 0) dan simpangan baku 1 (standard deviation = 1). Teknik ini digunakan untuk memastikan bahwa semua fitur memiliki skala yang sama, yang penting untuk algoritma machine learning sensitif terhadap skala fitur, seperti algoritma berbasis gradien. 
   - Standarization dapat meningkatkan performa dan stabilitas numerik pada model. Standarisasi diterapkan pada data latih (X_train) dan data uji (X_test).

## Modeling
Dalam project ini menggunakan model Random Forest. 
- Random forest merupakan salah satu algoritma _machine learning_ yang efektif, cara kerjanya yaitu dengan menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi prediksi. 
  - dalam project ini menggunakan n_estimators=70 untuk menentukan jumlah pohon keputusan yang akan dibangun dalam model random forest. Semakin banyak pohon, maka akurasi model akan meningkat, tetapi membutuhkan waktu pelatihan yang lebih lama.
  - random_state=42 seeding generator bilangan random untuk memastikan reprodusibilitas hasil.
__________________________________________________________________________________________________________________________________________________________
Tahapan

1. Pembentukan banyaknya pohon keputusan
  - Setiap pohon keputusan dalam Random Forest dibangun menggunakan subset data yang berbeda, sehingga tidak ada pohon yang menggunakan seluruh data yang sama.
2. Pembuatan Pohon Keputusan
  - Pada setiap titik percabangan (node), model memilih fitur terbaik yang digunakan untuk membagi data.
________________________________________________________________________________________________________________________________________________________
Penjelasan Parameter yang Digunakan dalam Model

- n_estimators=70: Parameter ini menentukan jumlah pohon keputusan yang akan dibangun dalam model Random Forest. Dalam konteks ini, kami memilih 70 pohon keputusan, yang berarti model akan menggunakan 70 pohon untuk membuat keputusan akhir. Semakin banyak pohon yang digunakan, semakin besar kemampuan model untuk menggeneralisasi dan memberikan prediksi yang akurat, tetapi juga membutuhkan waktu pelatihan yang lebih lama. Terlalu banyak pohon tidak selalu meningkatkan kinerja model secara signifikan setelah mencapai jumlah tertentu, tetapi ada trade-off antara akurasi dan waktu komputasi.
- random_state=42: Parameter ini digunakan untuk menentukan seeding generator bilangan random. Dengan menggunakan nilai random_state yang tetap (misalnya 42), kita memastikan bahwa proses acak dalam pembagian data, pemilihan subset fitur, dan pembuatan pohon keputusan tetap konsisten setiap kali model dijalankan. Ini penting untuk reproducibility, yang berarti jika eksperimen dilakukan berulang kali, hasil yang sama dapat diperoleh. Hal ini sangat berguna dalam penelitian dan pengembangan untuk memastikan keandalan dan validitas hasil eksperimen._________________________________________________________________________________________________________________________________

## Evaluation
Kinerja model Random Forest dievaluasi dengan menggunakan data pengujian yang belum pernah dilihat oleh model selama pelatihan. Beberapa metrik evaluasi yang digunakan adalah:

- Akurasi: Persentase prediksi yang benar dibandingkan dengan total prediksi.
- Precision: Seberapa akurat model dalam memprediksi kelas positif.
- Recall: Seberapa banyak kelas positif yang berhasil diprediksi oleh model.
- F1-Score: Kombinasi dari precision dan recall untuk memberikan ukuran kinerja yang lebih seimbang.
__________________________________________________________________________________________________________________________________________________
Hasil dari model random forest:

1. Akurasi Model
- Train Accuracy: 1.0 (100%)
- Model memiliki performa sempurna di data latih, tetapi ini mungkin menunjukkan potensi overfitting.
- Test Accuracy: 0.883 (88.26%)
- Akurasi cukup tinggi di data uji, menunjukkan model dapat memprediksi dengan baik, meskipun sedikit lebih rendah dibandingkan data latih.
2. Classification Report
- Precision, Recall, dan F1-Score:
- Kelas 0 (Negatif):
- Precision: 0.83 → 83% dari prediksi negatif benar.
- Recall: 0.91 → Model mampu menangkap 91% sampel negatif yang benar.
- F1-Score: 0.87 → Keseimbangan antara precision dan recall cukup baik.
- Kelas 1 (Positif):
- Precision: 0.93 → 93% dari prediksi positif benar.
- Recall: 0.86 → Model mampu menangkap 86% sampel positif yang benar.
- F1-Score: 0.89 → Hasilnya sedikit lebih baik dibandingkan kelas negatif.
3. Rata-rata Skor
- Akurasi Keseluruhan: 88% menunjukkan model cukup baik secara umum.
- Macro Avg dan Weighted Avg:
- Nilai-nilai ini konsisten di sekitar 0.88–0.89, mengindikasikan performa yang seimbang antar kelas.

Penerapan algoritma Random Forest dapat meningkatkan akurasi dalam mendeteksi dan memprediksi risiko penyakit jantung dengan kemampuannya menangani data kompleks dan non-linear, seperti faktor risiko yang saling terkait. Algoritma ini menggunakan pendekatan ensemble learning dan bagging, yang mengurangi overfitting sehingga model dapat menggeneralisasi dengan lebih baik pada data pengujian dan menghasilkan prediksi yang lebih stabil. Hasil evaluasi menunjukkan bahwa Random Forest memberikan akurasi tinggi, dengan penelitian sebelumnya menunjukkan akurasi mencapai 87.7% pada data pengujian dan 92.63% pada data validasi. Selain itu, Random Forest juga mampu mengidentifikasi fitur-fitur penting dalam memprediksi risiko penyakit jantung, memberikan wawasan bagi profesional medis tentang faktor-faktor risiko utama yang berpengaruh pada kesehatan jantung. Algoritma ini juga efektif menangani data yang hilang dan tidak terstruktur, membuatnya cocok untuk digunakan dalam deteksi dini dan pencegahan penyakit jantung pada pasien. Dengan demikian, Random Forest terbukti menjadi algoritma yang efektif untuk meningkatkan akurasi prediksi dan mendukung keputusan medis dalam penanganan penyakit jantung.

## Referensi
1. Haganta Depari, D., Widiastiwi, Y., Mega Santoni, M., Ilmu Komputer, F., Pembangunan Nasional Veteran Jakarta, U., Fatmawati Raya, J. R., & Labu, P. (n.d.). Perbandingan Model Decision Tree, Naive Bayes dan Random Forest untuk Prediksi Klasifikasi Penyakit Jantung. JURNAL INFORMATIK Edisi Ke, 18, 2022.
2. Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid machine learning techniques. IEEE Access, 7, 81542–81554. https://doi.org/10.1109/ACCESS.2019.2923707
3. Putri, S. A., Selayanti, N., Kristanaya, M., Azzahra, M. P., Navsih, M. G., & Hindrayani, K. M. (n.d.). Penerapan Machine Learning Algoritma Random Forest Untuk Prediksi Penyakit Jantung. Seminar Nasional Sains Data, 4(1), 895-906. https://doi.org/https://doi.org/10.33005/senada.v4i1.376
 
