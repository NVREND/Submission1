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
Setiap dataset yang digunakan dapat ditemukan pada Index of heart disease datasets dari UCI Machine Learning Repository pada link berikut : https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

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
- Pembagian data
  - dilakukan dengan membagi dataset menjadi 75% data pelatihan dan 25% data uji dengan masing masing 516 data pelatihan dan 230 data uji.
- Oversampling
- Standarization
   - Standardization adalah teknik untuk mengubah data agar memiliki distribusi dengan rata-rata 0 (mean = 0) dan simpangan baku 1 (standard deviation = 1). Teknik ini digunakan untuk memastikan bahwa semua fitur memiliki skala yang sama, yang penting untuk algoritma machine learning sensitif terhadap skala fitur, seperti algoritma berbasis gradien. 
   - Standarization dapat meningkatkan performa dan stabilitas numerik pada model. Standarisasi diterapkan pada data latih (X_train) dan data uji (X_test).

## Modeling
Dalam project ini menggunakan model Random Forest. 
- Random forest merupakan salah satu algoritma _machine learning_ yang efektif, cara kerjanya yaitu dengan menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi prediksi. 
  - dalam project ini menggunakan n_estimators=70 untuk menentukan jumlah pohon keputusan yang akan dibangun dalam model random forest. Semakin banyak pohon, maka akurasi model akan meningkat, tetapi membutuhkan waktu pelatihan yang lebih lama.
  - random_state=42 seeding generator bilangan random untuk memastikan reprodusibilitas hasil.

## Evaluation
Pada project ini saya menggunakan metrik akurasi, precision, recall, dan F1 score.
- Metrik yang digunakan
  - Akurasi: Merupakan metrik untuk menghitung proporsi prediksi yang benar dari semua prediksi yang dilakukan model. Nilai akurasi yang tinggi menunjukkan bahwa model mampu memprediksi sebagian besar kasus dengan benar.
    - Hasil: Random Forest memiliki akurasi testing yang sedikit lebih tinggi dibandingkan KNN. Rndom forest: akurasi training 100% dan akurasi testing 88.3%. KNN: akurasi training 90.4% dan akurasi testing 85.7%.
      
  - Precision:metrik ini berguna untuk mengukur proporsi prediksi positif yang benar-benar positif. Dihitung dengan membagi jumlah positif yang diprediksi dengan benar dengan jumlah total prediksi positif.
     - Hasil: Random Forest memiliki precision yang lebih tinggi untuk kelas positif (penyakit jantung) dibandingkan KNN. KNN memiliki precision yang lebih tinggi untuk kelas negatif (tidak memiliki penyakit jantung) dibandingkan Random Forest.
         - Random Forest: Kelas negatif (0) 0.83, Kelas positif (1) 0.93
         - KNN: Kelas negatif (0) 0.80, Kelas positif (1) 0.90

  - Recall: untuk proporsi kasus positif yang benar-benar diidentifikasi oleh model. Dihitung dengan membagi jumlah positif yang diprediksi dengan benar dengan jumlah total kasus positif yang sebenarnya. Recall yang tinggi menunjukkan bahwa model mampu mengidentifikasi sebagian besar kasus positif.
      - Hasil: Random Forest memiliki recall yang lebih tinggi untuk kelas negatif dibandingkan KNN. KNN memiliki recall yang sedikit lebih tinggi untuk kelas positif dibandingkan Random Forest.
          - Random Forest: Kelas negatif (0) 0.91. Kelas positif (1) 0.86
          - KNN: Kelas negatif (0) 0.88, Kelas positif (1) 0.84
     
  -  F1-Score: Merupakan metrik yang menggabungkan precision dan recall menjadi satu skor. Dihitung dengan rata-rata harmonis precision dan recall.
F1-Score yang tinggi menunjukkan keseimbangan yang baik antara precision dan recall, artinya model dapat memprediksi kedua kelas dengan cukup baik.
      - Hasil: F1-Score untuk kedua model berada di kisaran yang serupa.
          - Random Forest: Kelas negatif (0) 0.87, Kelas positif 0.89
          - KNN: Kelas negatif (0) 0.84, Kelas (positif (1) 0.87

- Kesimpulan: Random Forest memiliki akurasi testing yang sedikit lebih tinggi dan unggul dalam mendeteksi penyakit jantung (precision dan recall untuk kelas positif). Namun, KNN mungkin lebih baik dalam mengidentifikasi pasien yang sehat (precision untuk kelas negatif). 



## Referensi
1. Haganta Depari, D., Widiastiwi, Y., Mega Santoni, M., Ilmu Komputer, F., Pembangunan Nasional Veteran Jakarta, U., Fatmawati Raya, J. R., & Labu, P. (n.d.). Perbandingan Model Decision Tree, Naive Bayes dan Random Forest untuk Prediksi Klasifikasi Penyakit Jantung. JURNAL INFORMATIK Edisi Ke, 18, 2022.
2. Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid machine learning techniques. IEEE Access, 7, 81542–81554. https://doi.org/10.1109/ACCESS.2019.2923707
3. Putri, S. A., Selayanti, N., Kristanaya, M., Azzahra, M. P., Navsih, M. G., & Hindrayani, K. M. (n.d.). Penerapan Machine Learning Algoritma Random Forest Untuk Prediksi Penyakit Jantung. Seminar Nasional Sains Data, 4(1), 895-906. https://doi.org/https://doi.org/10.33005/senada.v4i1.376
 
