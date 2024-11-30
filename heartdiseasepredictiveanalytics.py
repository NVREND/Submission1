# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# # Import Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV



#evaluasi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, classification_report

# %% [markdown]
# Import semua library yang digunakan untuk mengerjakan project

# %% [markdown]
# # Data Loading

# %%
!mkdir ~/.kaggle

# %% [markdown]
# membuat folder kaggle

# %%
!cp '/content/kaggle.json' ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets list

# %%
!kaggle datasets download -d fedesoriano/heart-failure-prediction

# %% [markdown]
# download dataset dari kaggle

# %%
!unzip /content/heart-failure-prediction.zip

# %%
df= pd.read_csv('heart.csv')
df

# %% [markdown]
# Dari output diatas dapat dilihat bahwa:
# - Terdapat 918 baris dalam dataset
# - terdapat 12 kolom, yaitu: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease

# %%
df.info()

# %% [markdown]
# Dapat dilihat bahwa terdapat beberapa informasi:
#   - Terdapat 5 kolom dengan tipe data object, yaitu: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
#   - There are 6 kolom numeric dengan tipe data int64, yaitu: Age, RestigBP, Cholesterol, FastingBS, MaxHR, HeartDisease
#   - There is 1 kolom numeric dengan tipe data float64, yaitu Oldpeak

# %% [markdown]
# # Data Understanding

# %% [markdown]
# Project ini menggunakan dataset yang tersedia secara publik di Kaggle: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. Dataset ini berisi informasi tentang pasien gagal jantung dan dimaksudkan untuk digunakan dalam tugas pemodelan prediktif. Dataset terdiri dari 918 data.
# 
# Dataset ini dibuat dengan menggabungkan berbagai dataset yang sudah tersedia secara mandiri namun belum digabungkan sebelumnya. Dalam kumpulan data ini, 5 kumpulan data jantung digabungkan dalam 11 fitur umum yang menjadikannya kumpulan data penyakit jantung terbesar yang tersedia sejauh ini untuk tujuan penelitian.  Lima kumpulan data yang digunakan untuk kurasinya adalah:
# - Cleveland: 303 observations
# - Hungarian: 294 observations
# - Switzerland: 123 observations
# - Long Beach VA: 200 observations
# - Stalog (Heart) Data Set: 270 observations
#   
# Total: 1190 observations
# 
# Duplicated: 272 observations
# 
# Final dataset: 918 observations
# 
# ### Variabel-variabel pada heart failur prediction dataset adalah sebagai berikut:
# - Age: usia pasien [tahun]
# - Sex: jenis kelamin pasien [M: Pria, F: Wanita]
# - ChestPainType: jenis nyeri dada [TA: Typical Angina, ATA: Atypical Angina, NAP: -Non-Anginal Pain, ASY: Asymptomatic]
# - RestingBP: tekanan darah istirahat [mm Hg]
# - Cholesterol: kolesterol serum [mm/dl]
# - FastingBS: gula darah puasa [1: if FastingBS > 120 mg/dl, 0: otherwise]
# - RestingECG: hasil elektrokardiogram istirahat [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# - MaxHR: detak jantung maksimum tercapai [Numeric value between 60 and 202]
# - ExerciseAngina: angina akibat olahraga [Y: Yes, N: No]
# - Oldpeak: oldpeak = ST [Numeric value measured in depression]
# - ST_Slope: kemiringan puncak latihan segmen ST [Up: upsloping, Flat: flat, Down: downsloping]
# - HeartDisease: kelas keluaran [1: heart disease, 0: Normal]
# --------------------------------------------------------------------------------------------------------------------------------------
# - Terdapat 5 kolom dengan object types, yaitu: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope 
# - Terdapat 6 kolom numeric dengan type int64, yaitu: Age, RestigBP, Cholesterol, FastingBS, MaxHR, HeartDisease
# - Terdapat 1 kolom numeric dengan type float64, yaitu Oldpeak

# %% [markdown]
# # EDA

# %% [markdown]
# Mengecek missing values

# %%
df.isna().sum()

# %% [markdown]
# Dapat dilihat bahwa tidak ada missing value untuk semua kolom

# %% [markdown]
# check statistics

# %%
df.describe(include="all")

# %% [markdown]
# Dataset ini berisi informasi tentang pasien dengan potensi penyakit jantung, dengan beberapa temuan utama:
# 
# - Demografi: Rata-rata usia pasien adalah 53.5 tahun, mayoritas adalah pria (725 pasien).
# - Klinis: 
#     - Rata-rata tekanan darah istirahat 132.4 mmHg, kolesterol 198.8 mg/dL, dan denyut jantung maksimum 136.8 bpm.
#     - Nilai tidak realistis, seperti tekanan darah dan kolesterol 0, menunjukkan adanya data yang perlu dibersihkan.
# - Target: Data target biner (HeartDisease) menjadi fokus analisis prediksi.

# %%
# Calculate the percentage of heart disease
hd_count = df['HeartDisease'].value_counts().reset_index(name='count')
hd_count.columns = ['HeartDisease', 'count']
hd_count['persentage'] = (hd_count['count'] / hd_count['count'].sum()) * 100
hd_count

# %% [markdown]
# Dari data ini, kita dapat menyimpulkan bahwa dalam dataset ini, lebih banyak individu yang terindikasi memiliki penyakit jantung (55,34%) dibandingkan mereka yang tidak memiliki penyakit jantung (44,66%).

# %% [markdown]
# ## Univariatve Analysis

# %% [markdown]
# - Ctegorical Features = Sex, ChestPainType, FastingBs, RestingECG, ExerciseAngina:, ST_Slope, HeartDisease
# - numerical features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
# 
# 

# %%
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# %%
df.info()

# %% [markdown]
# ### Categorical Features

# %% [markdown]
# #### Feature Sex

# %%
feature = categorical_features[0]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_percent = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_percent)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Dalam dataset, mayoritas sampel dalam dataset adalah Pria dengan persentase 79.0%, sedangkan Wanita memiliki persentase 21.0%.

# %% [markdown]
# #### Feature ChestPainType

# %%
feature = categorical_features[1]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_percent = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_percent)
count.plot(kind='bar', title=feature);

# %% [markdown]
# - Terdapat 4 kategori pada fitur ChestPainType , yaitu TA: Typical Angina, ATA: Atypical Angina, NAP: -Non-Anginal Pain, ASY: Asymptomatic.  
# - Mayoritas pasien dalam kumpulan data mengalami jenis nyeri dada tanpa gejala (ASY), yang mencapai 54,0% dari total sampel.  Diikuti oleh nyeri non-angina (NAP) dengan 22,1%, angina tipikal (ATA) dengan 18,8%, dan angina atipikal (TA) hanya 5,0% dari total sampel.

# %% [markdown]
# #### Feature FastingBS

# %%
feature = categorical_features[2]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_percent = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_percent)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Ada 2 kategori dalam fitur FastingBS, yaitu gula darah puasa 1: jika FastingBS > 120 mg/dl, 0: sebaliknya.
#  Grafik batang juga menunjukkan distribusi kategori secara visual, dengan mayoritas pasien (76,7%) tidak berpuasa, sementara 23,3% dari total sampel berpuasa.  Analisis fitur “FastingBS” dapat menjadi faktor yang relevan dalam memahami prediksi gagal jantung.

# %% [markdown]
# #### RestingECG

# %%
feature = categorical_features[3]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_percent = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_percent)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Dari distribusi data RestingECG (Elektrokardiogram saat istirahat):
# 
# Kategori Normal (Normal):
# 
# 552 sampel (60.1%) menunjukkan elektrokardiogram normal.
# Ini adalah kategori terbesar, menunjukkan mayoritas pasien tidak memiliki kelainan elektrokardiogram saat istirahat.
# Kategori LVH (Left Ventricular Hypertrophy):
# 
# 188 sampel (20.5%) menunjukkan hipertrofi ventrikel kiri, yang bisa menjadi indikasi tekanan darah tinggi atau penyakit jantung.
# Kategori ST:
# 
# 178 sampel (19.4%) menunjukkan kelainan segmen ST, yang sering terkait dengan iskemia atau serangan jantung.
# Kesimpulan: Mayoritas pasien memiliki elektrokardiogram normal, tetapi sekitar 40% menunjukkan kelainan (LVH atau ST), yang menjadi indikasi risiko penyakit jantung yang perlu ditangani lebih lanjut.

# %% [markdown]
# #### ExerciseAngina

# %%
feature = categorical_features[4]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_percent = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_percent)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Dari distribusi data ExerciseAngina (Angina saat berolahraga):
# 
# Kategori N (Tidak Ada Angina):
# 
# 547 sampel (59.6%) tidak mengalami angina saat berolahraga.
# Ini menunjukkan mayoritas pasien tidak menunjukkan gejala angina terkait aktivitas fisik.
# Kategori Y (Ada Angina):
# 
# 371 sampel (40.4%) mengalami angina saat berolahraga.
# Angka ini cukup signifikan, mengindikasikan bahwa hampir setengah dari pasien menunjukkan gejala angina saat aktivitas fisik, yang dapat menjadi indikator penting risiko penyakit jantung.
# Kesimpulan: Mayoritas pasien tidak mengalami angina saat berolahraga, tetapi persentase yang mengalami angina cukup tinggi (40.4%), menunjukkan perlunya perhatian khusus terhadap kelompok ini dalam analisis prediksi penyakit jantung.

# %% [markdown]
# #### Feature ST_Slope

# %%
feature = categorical_features[5]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_percent = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_percent)
count.plot(kind='bar', title=feature);

# %% [markdown]
# Mayoritas data memiliki kemiringan ST yang “Datar” (50,1%), diikuti oleh “Naik” (43,0%) dan “Turun” (6,9%).

# %% [markdown]
# ### Numerical Features

# %%
df.hist(bins=50, figsize=(15,10))
plt.show()

# %% [markdown]
# - Distribusi usia menunjukkan bahwa sebagian besar individu dalam data berusia antara 40 dan 60 tahun.   Puncak tertinggi menunjukkan bahwa terdapat paling banyak orang berusia 54 tahun dalam set data.
# - Distribusi tekanan darah istirahat Histogram pada kolom RestingBP menunjukkan bahwa sebagian besar orang dalam data memiliki tekanan darah istirahat yang normal (di bawah 120 mmHg).
# - Distribusi detak jantung maksimum menunjukkan bahwa mayoritas orang dalam data memiliki detak jantung maksimum yang normal (antara 75 dan 150 bpm).
# - Distribusi depresi ST (Oldpeak) menunjukkan bahwa mayoritas orang dalam data tidak memiliki depresi ST (0 mm).

# %%
df_enc = df

# %%
# encode all the categorical features
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

#Encode Sex Feature
label_encoder.fit(df_enc["Sex"])
df_enc["Sex_en"] = label_encoder.fit_transform(df_enc["Sex"])
df_enc.drop("Sex",axis=1,inplace=True)

# %% [markdown]
# melakukan encoding fitur kategorikal
# 
# Encoding ini membantu algoritma machine learning memahami data kategorikal dengan mengubahnya menjadi representasi numerik. Teknik ini cocok untuk fitur kategorikal dengan jumlah kategori yang terbatas.

# %%
#Encode ChestPainType feature
label_encoder.fit(df_enc["ChestPainType"])
df_enc["ChestPainType_en"] = label_encoder.fit_transform(df_enc["ChestPainType"])
df_enc.drop("ChestPainType",axis=1,inplace=True)
df_enc

# %% [markdown]
# Dapat dilihat bahwa fitur ChestPainType  menjadi representasi numerik

# %%
#Encode RestingECG feature
label_encoder.fit(df_enc["RestingECG"])
df_enc["RestingECG_en"] = label_encoder.fit_transform(df_enc["RestingECG"])
df_enc.drop("RestingECG",axis=1,inplace=True)
df_enc

# %% [markdown]
# Dapat dilihat bahwa fitur RestingECG menjadi representasi numerik

# %%
label_encoder.fit(df_enc["ExerciseAngina"])
df_enc["ExerciseAngina_en"] = label_encoder.fit_transform(df_enc["ExerciseAngina"])
df_enc.drop("ExerciseAngina",axis=1,inplace=True)
df_enc

# %%
label_encoder.fit(df_enc["ST_Slope"])
df_enc["ST_Slope_en"] = label_encoder.fit_transform(df_enc["ST_Slope"])
df_enc.drop("ST_Slope",axis=1,inplace=True)
df_enc

# %% [markdown]
# Dapat dilihat bahwa fitur ST_Slope  menjadi representasi numerik

# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# ## Spliting Data

# %%
from sklearn.model_selection import train_test_split

# feature & target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %% [markdown]
# import train_test_split
# membagi dataset menjadi 25% data uji dan 75% data latih

# %%
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# %% [markdown]
# Dapat dilihat bahwa terdapat 688 data latih dan 230 data uji

# %% [markdown]
# ### Standarization

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train

# %%
#X_train[numerical_features].describe().round(4)

# %% [markdown]
# # Modeling

# %%
knn = KNeighborsClassifier(n_neighbors=3)
# Train the model
knn.fit(X_train, y_train)

# %% [markdown]
# inisialisasi n_neighbors dengan nilai 3

# %%
print("Train Accuracy:",knn.score(X_train, y_train))
prediction = knn.predict(X_test)
print("Test Accuracy:",accuracy_score(prediction, y_test))
print("Classification Report",classification_report(y_test, prediction))

# %%
rf =  RandomForestClassifier(n_estimators=70, random_state=42)
# Train the model
rf.fit(X_train, y_train)

# %% [markdown]
# inisialisasi n_estimator atau pohon berjumlah 70
# dan random state 42

# %%
print("Train Accuracy:",rf.score(X_train, y_train))
prediction = rf.predict(X_test)
print("Test Accuracy:",accuracy_score(prediction, y_test))
print("Classification Report",classification_report(y_test, prediction))

# %% [markdown]
# 1. Akurasi Model
# - Train Accuracy: 1.0 (100%)
# - Model memiliki performa sempurna di data latih, tetapi ini mungkin menunjukkan potensi overfitting.
# - Test Accuracy: 0.883 (88.26%)
# - Akurasi cukup tinggi di data uji, menunjukkan model dapat memprediksi dengan baik, meskipun sedikit lebih rendah dibandingkan data latih.
# 2. Classification Report
# - Precision, Recall, dan F1-Score:
# - Kelas 0 (Negatif):
# - Precision: 0.83 → 83% dari prediksi negatif benar.
# - Recall: 0.91 → Model mampu menangkap 91% sampel negatif yang benar.
# - F1-Score: 0.87 → Keseimbangan antara precision dan recall cukup baik.
# - Kelas 1 (Positif):
# - Precision: 0.93 → 93% dari prediksi positif benar.
# - Recall: 0.86 → Model mampu menangkap 86% sampel positif yang benar.
# - F1-Score: 0.89 → Hasilnya sedikit lebih baik dibandingkan kelas negatif.
# 3. Rata-rata Skor
# - Akurasi Keseluruhan: 88% menunjukkan model cukup baik secara umum.
# - Macro Avg dan Weighted Avg:
# - Nilai-nilai ini konsisten di sekitar 0.88–0.89, mengindikasikan performa yang seimbang antar kelas.

# %% [markdown]
# 


