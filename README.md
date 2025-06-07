## Laporan Proyek Machine Learning - Dwi NurCahyo Purbonegoro

![image](https://github.com/user-attachments/assets/9629af78-8f79-40f8-85a7-f95b631bafb6)

### Project Overview

Dalam era digital saat ini, konsumen seringkali dihadapkan dengan banyaknya pilihan produk smartphone di pasaran. Banyaknya pilihan ini dapat menimbulkan kebingungan dalam pengambilan keputusan pembelian. Oleh karena itu, dibutuhkan sistem rekomendasi yang mampu membantu pengguna untuk menemukan produk yang sesuai dengan preferensi dan kebutuhan mereka. Proyek ini bertujuan membangun sistem rekomendasi smartphone dengan memanfaatkan pendekatan Content-based Filtering dan Collaborative Filtering.

Sistem rekomendasi semacam ini dapat meningkatkan pengalaman pengguna, meningkatkan penjualan, dan efisiensi pencarian produk pada platform e-commerce. Berdasarkan studi dari Ricci et al. (2011), sistem rekomendasi memiliki kontribusi signifikan dalam e-commerce karena mampu menyajikan informasi relevan secara personal.

### Business Understanding

#### Problem Statements

1. Bagaimana merekomendasikan produk smartphone kepada pengguna berdasarkan fitur dan preferensi pengguna?
2. Bagaimana meningkatkan relevansi hasil rekomendasi agar sesuai dengan minat pengguna?

#### Goals

1. Mengembangkan model rekomendasi berbasis konten untuk menyarankan smartphone dengan fitur serupa.
2. Mengembangkan model rekomendasi berbasis kolaboratif untuk menyarankan smartphone berdasarkan interaksi pengguna lain.

#### Solution Approach

1. **Content-based Filtering**: Menggunakan data pada kolom "corpus" untuk mengekstrak fitur smartphone dan merekomendasikan produk serupa.
2. **Collaborative Filtering**: Menggunakan matrix factorization (contoh: SVD) untuk membangun model berdasarkan rating pengguna.

### Data Understanding

Dataset yang digunakan berasal dari kaggle (https://www.kaggle.com/datasets/gyanprakashkushwaha/mobile-recommendation-system-dataset) terdiri dari 2546 entri dengan 5 kolom, yaitu:

* **name**: Nama produk smartphone.
* **ratings**: Skor rating dari pengguna.
* **price**: Harga produk (dalam format teks, perlu diubah ke numerik).
* **imgURL**: URL gambar produk.
* **corpus**: Deskripsi fitur produk (seperti RAM, storage, OS, dll).

Jumlah nilai null pada kolom `corpus`: 12 nilai.
Jumlah duplikasi data: 1 baris.

### Data Preparation

1. Load dataset dari kaggle dan melakukan EDA

  - **Import Library**

  ![image](https://github.com/user-attachments/assets/e4aabff9-9fbd-4aff-9424-0ef848f158bc)

  Pada tahap ini, dilakukan import library-library Python yang dibutuhkan untuk membangun sistem rekomendasi dan melakukan preprocessing data:

  * **`pandas`**: digunakan untuk membaca dan mengelola data dalam bentuk tabel (DataFrame).
  * **`TfidfVectorizer`** dari `sklearn.feature_extraction.text`: digunakan untuk mengubah teks menjadi representasi numerik menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency).
  * **`linear_kernel`** dari `sklearn.metrics.pairwise`: digunakan untuk menghitung kesamaan antar vektor (TF-IDF) menggunakan rumus kernel linier (dot product).
  * **`NearestNeighbors`** dari `sklearn.neighbors`: digunakan untuk membangun model rekomendasi berbasis Collaborative Filtering dengan pendekatan K-Nearest Neighbors (KNN).
  * **`MinMaxScaler`** dari `sklearn.preprocessing`: digunakan untuk menormalkan fitur numerik ke rentang 0–1 agar setara dalam perhitungan jarak.
  * **`csr_matrix`** dari `scipy.sparse`: digunakan untuk menyimpan data dalam bentuk matriks sparse (hemat memori) yang cocok untuk data besar dan banyak nilai nol.
  * **`re`** (Regular Expression): digunakan untuk melakukan proses pembersihan teks, seperti menghapus karakter khusus, angka, atau simbol yang tidak diperlukan.
  
  - **Mengunggah File kaggle.json** 

  ![image](https://github.com/user-attachments/assets/64a837d4-e8dd-4987-a995-2c180d5d31db)

  Pada tahap ini, dilakukan proses untuk mengunggah file autentikasi agar dapat mengakses dataset dari Kaggle melalui Google Colab:

  files dari google.colab: digunakan untuk mengakses fitur unggah file dari komputer lokal ke Google Colab.

  files.upload(): digunakan untuk membuka dialog unggah file, dalam hal ini bertujuan mengunggah file kaggle.json, yaitu file kredensial yang berisi API token akun Kaggle. File ini diperlukan agar Google Colab bisa terhubung dan mengunduh dataset langsung dari Kaggle.

  - **Mendownload dataset dari kaggle**

  ![image](https://github.com/user-attachments/assets/859e6c3a-e0fd-4360-92f6-4ca4d44c4e20)
  
  Pada tahap ini, dilakukan instalasi dan konfigurasi agar Google Colab bisa mengunduh dataset dari Kaggle menggunakan API, serta mengekstrak dataset ke dalam folder kerja:

  * **`!pip install -q kaggle`**: menginstal library `kaggle` secara diam-diam (`-q` = quiet), yang memungkinkan interaksi dengan Kaggle melalui command line.
  * **`!mkdir -p ~/.kaggle`**: membuat folder `.kaggle` di direktori home, tempat menyimpan file konfigurasi API.
  * **`!cp kaggle.json ~/.kaggle/`**: menyalin file `kaggle.json` (yang telah diunggah sebelumnya) ke dalam folder `.kaggle`.
  * **`!chmod 600 ~/.kaggle/kaggle.json`**: mengatur izin file `kaggle.json` agar hanya bisa dibaca dan ditulis oleh pemilik (keamanan akses API).
  * **`!kaggle datasets download -d gyanprakashkushwaha/mobile-recommendation-system-dataset`**: mengunduh dataset dari Kaggle menggunakan ID dataset `gyanprakashkushwaha/mobile-recommendation-system-dataset`.
  * **`!unzip -o mobile-recommendation-system-dataset.zip -d data`**: mengekstrak file zip hasil unduhan ke dalam folder `data`, dengan opsi `-o` untuk menimpa file jika sudah ada.

  - **Pemuatan Dataset**

  ![image](https://github.com/user-attachments/assets/eb3f4208-52c1-4b91-aa70-f3221d6261a3)

  Pada tahap ini, dilakukan pemuatan (load) dataset ke dalam memori untuk dianalisis lebih lanjut:

  * **`pd.read_csv()`** dari library `pandas`: digunakan untuk membaca file CSV dan mengubahnya menjadi objek DataFrame.
  * **`"/content/data/mobile_recommendation_system_dataset.csv"`**: merupakan path lengkap ke file dataset yang telah diekstrak sebelumnya dari file zip.
  * **Tujuan**: menyimpan isi dataset ke dalam variabel `df` agar dapat digunakan untuk proses eksplorasi data, pembersihan, dan pemodelan sistem rekomendasi.

  - **Pemeriksaan struktur dataset**

  ![image](https://github.com/user-attachments/assets/07e924fe-f6aa-4e5c-89d1-c335f6d90523)

  Pada tahap ini, dilakukan pemeriksaan struktur dataset menggunakan fungsi `info()` untuk memahami tipe data dan kelengkapan tiap kolom:

  * **`df.info()`**: digunakan untuk menampilkan ringkasan informasi DataFrame, seperti jumlah baris, nama kolom, jumlah nilai non-null, dan tipe data setiap kolom.
  * **Hasil yang ditampilkan**:

    * Dataset memiliki **2546 baris** dan **5 kolom**.
    * Kolom `name`, `ratings`, `price`, `imgURL`, dan `corpus` masing-masing berisi data tentang nama produk, rating, harga, URL gambar, dan deskripsi produk.
    * Kolom `corpus` memiliki **12 data yang kosong (null)**.
  * Tipe data:

    * `ratings`: `float64` → nilai numerik (desimal).
    * `name`, `price`, `imgURL`, `corpus`: `object` → umumnya berarti string (teks).
  * **Tujuan**: memahami kondisi awal dataset agar bisa melakukan pembersihan dan preprocessing dengan tepat sebelum digunakan dalam sistem rekomendasi.

  - **Menganalisis statistik deskriptif**

  ![image](https://github.com/user-attachments/assets/6e42cae6-8132-42e5-8124-689377637335)

  Pada tahap ini, dilakukan analisis statistik deskriptif pada kolom numerik dalam dataset untuk memahami distribusi nilai:

  * **`df.describe()`**: digunakan untuk menampilkan ringkasan statistik dari kolom bertipe numerik dalam DataFrame.

  * **Hasil yang ditampilkan untuk kolom `ratings`**:

    * **`count`**: 2546 — jumlah data rating yang tersedia.
    * **`mean`**: 4.30 — rata-rata rating produk.
    * **`std` (standard deviation)**: 0.21 — penyebaran rating relatif kecil, artinya sebagian besar rating saling berdekatan.
    * **`min`**: 2.90 — nilai rating terendah.
    * **`25%`**: 4.20 — 25% data memiliki rating di bawah 4.20.
    * **`50%`** (median): 4.30 — separuh data memiliki rating di bawah atau sama dengan 4.30.
    * **`75%`**: 4.40 — 75% data memiliki rating di bawah 4.40.
    * **`max`**: 5.00 — nilai rating tertinggi.

  * **Tujuan**: memahami karakteristik kolom `ratings` sebagai salah satu fitur penting dalam sistem rekomendasi, terutama untuk model berbasis Collaborative Filtering.

  - **Menampilkan 5 baris pertama**
  
  ![image](https://github.com/user-attachments/assets/414edd1f-1ad4-4d06-b5c6-1a49ebca2c55)

  Pada tahap ini, dilakukan peninjauan data awal menggunakan `head()` untuk melihat contoh isi dari dataset:

  * **`df.head()`**: digunakan untuk menampilkan **5 baris pertama** dari dataset, bertujuan memahami struktur dan isi tiap kolom secara langsung.

  * **Penjelasan tiap kolom yang ditampilkan**:

    * **`name`**: nama lengkap produk smartphone, termasuk merek, model, warna, dan kapasitas penyimpanan.
    * **`ratings`**: skor penilaian produk dari pengguna, berupa nilai desimal (contoh: 4.2, 4.5).
    * **`price`**: harga produk, ditampilkan sebagai string (contoh: `₹20,999`) yang nantinya perlu dibersihkan dan dikonversi ke numerik.
    * **`imgURL`**: link URL gambar produk dari situs sumber (berguna untuk tampilan antarmuka sistem rekomendasi).
    * **`corpus`**: deskripsi fitur-fitur produk (seperti RAM, OS, prosesor) dalam bentuk teks, yang akan digunakan untuk Content-Based Filtering.

  * **Tujuan**: mendapatkan gambaran awal tentang isi dataset agar bisa menentukan langkah preprocessing dan pemodelan sistem rekomendasi yang sesuai.

  - **Pemeriksaan nilai yang hilang (null)**

  ![image](https://github.com/user-attachments/assets/3c8b5460-5764-4544-ac78-5a1618159e9e)

  Pada tahap ini, dilakukan pemeriksaan nilai yang hilang (null) di setiap kolom dataset:

  * **`df.isnull().sum()`**: digunakan untuk menghitung jumlah nilai kosong (null/NaN) di setiap kolom dalam DataFrame.

  * **Hasil yang ditampilkan**:

    * Kolom `name`, `ratings`, `price`, dan `imgURL` tidak memiliki nilai yang hilang (**0**).
    * Kolom **`corpus` memiliki 12 nilai yang hilang**, artinya ada 12 baris data tanpa deskripsi produk.

  * **Tujuan**: mengidentifikasi data yang tidak lengkap agar bisa ditangani (misalnya dihapus atau diisi) sebelum digunakan dalam pemodelan, khususnya pada Content-Based Filtering yang membutuhkan teks deskripsi (`corpus`).

  - **Pemeriksaan terhadap data yang duplikat**

  ![image](https://github.com/user-attachments/assets/c6fce9ec-84d0-4566-834f-669ea82c0fdc)

  Pada tahap ini, dilakukan pemeriksaan terhadap data yang duplikat untuk menjaga kualitas dan keunikan data:

  * **`df.duplicated().sum()`**: digunakan untuk menghitung jumlah baris yang **sama persis** (duplikat) dengan baris lainnya dalam DataFrame.

  * **Hasil**: terdapat **1 baris duplikat** dalam dataset.

  * **`np.int64(1)`**: hanya menunjukkan bahwa hasilnya berupa angka 1 dalam format `numpy int64`, yaitu tipe data bilangan bulat dari library NumPy.

  * **Tujuan**: mendeteksi dan nantinya menghapus data yang duplikat agar tidak memengaruhi hasil analisis atau pemodelan, karena duplikasi bisa menyebabkan bias dalam sistem rekomendasi.

2. Menghapus nilai null pada kolom `corpus`.

  ![image](https://github.com/user-attachments/assets/46348fe1-06b7-4d77-907d-acb9fc3f443a)

  Pada tahap ini, dilakukan pembersihan data dengan menghapus baris yang memiliki nilai kosong (null) pada kolom `corpus`:

  * **`df.dropna(subset=["corpus"], inplace=True)`**:

    * `dropna()`: digunakan untuk menghapus baris yang memiliki nilai kosong (NaN).
    * `subset=["corpus"]`: hanya mengecek kolom `corpus` saat mencari nilai kosong.
    * `inplace=True`: perubahan dilakukan langsung pada DataFrame `df` tanpa perlu membuat salinan baru.

  * **Tujuan**: memastikan bahwa semua baris data memiliki deskripsi produk (`corpus`), yang sangat penting untuk membangun **Content-Based Filtering**. Tanpa deskripsi, model tidak dapat menghitung kemiripan antar produk.

3. Membersihkan dan mengonversi kolom `price` menjadi numerik.

  ![image](https://github.com/user-attachments/assets/7c61d23c-70a4-4b60-bd50-4e9b3cfc2bfb)

  Pada tahap ini, dilakukan pembersihan data harga agar dapat digunakan sebagai nilai numerik dalam analisis:

  * `replace('[₹,]', '', regex=True)`: menghapus simbol `₹` dan tanda koma dari kolom `price` menggunakan regular expression, sehingga hanya menyisakan angka.
  * `replace('', '0')`: mengganti nilai kosong (jika ada) menjadi nol sebagai penanganan error saat konversi tipe data.
  * `astype(float)`: mengubah tipe data dari string menjadi `float` agar bisa dianalisis secara numerik.
  * `df['price_clean']`: kolom baru hasil pembersihan disimpan dengan nama `price_clean`.

  **Tujuan**: mengubah data harga dari format teks menjadi format numerik agar dapat digunakan dalam analisis statistik dan model rekomendasi, seperti untuk pengurutan, normalisasi, atau clustering berdasarkan harga.

  Digunakan untuk menampilkan statistik deskriptif dari kolom harga yang telah dibersihkan (`price_clean`), seperti rata-rata, minimum, maksimum, dan kuartil, untuk memahami sebaran harga produk dalam dataset.

4. Menghilangkan baris duplikat.

  ![image](https://github.com/user-attachments/assets/c9e8d903-6b8a-4fbf-9562-59bbe357071b)

  Pada tahap ini, dilakukan penghapusan baris duplikat dalam dataset untuk menjaga keunikan data:

  * `drop_duplicates()`: digunakan untuk menghapus baris yang **identik sepenuhnya** dengan baris lain dalam DataFrame.
  * `inplace=True`: perubahan dilakukan langsung pada DataFrame `df` tanpa membuat salinan baru.
  
  **Tujuan**: memastikan bahwa setiap baris dalam dataset mewakili entitas produk yang unik. Duplikasi data dapat menyebabkan bias dalam sistem rekomendasi karena satu produk bisa direkomendasikan berulang kali secara tidak adil.

5. Mengubah semua teks pada `corpus` menjadi huruf kecil dan membersihkannya.

  ![image](https://github.com/user-attachments/assets/8823a958-1548-46d8-af38-2ac8d6f5a33f)

  Pada tahap ini, dilakukan pembersihan teks deskripsi produk (`corpus`) agar siap digunakan dalam pemodelan berbasis teks:

  * **`def clean_text(text)`**: mendefinisikan fungsi pembersih teks.

    * `text.lower()`: mengubah semua huruf menjadi huruf kecil untuk konsistensi.
    * `re.sub(r'[^a-z0-9\s]', '', text)`: menghapus semua karakter yang bukan huruf, angka, atau spasi.
    * `re.sub(r'\s+', ' ', text).strip()`: menghilangkan spasi berlebih dan spasi di awal/akhir teks.
  * **`df['corpus_clean']`**: menyimpan hasil pembersihan dalam kolom baru bernama `corpus_clean`.
  * **`.apply(clean_text)`**: menerapkan fungsi `clean_text` ke setiap baris di kolom `corpus`.

  **Tujuan**: membersihkan teks dari simbol, kapitalisasi, dan spasi yang tidak perlu agar model berbasis teks seperti **TF-IDF Vectorizer** dapat mengenali dan membandingkan fitur produk dengan lebih akurat dalam sistem rekomendasi.

6. Menggunakan TF-IDF Vectorizer untuk mentransformasikan `corpus` menjadi vektor fitur.

   ![image](https://github.com/user-attachments/assets/e9921355-b15c-44c1-a537-082b61a02943)

  Pada tahap ini, dilakukan ekstraksi fitur teks menggunakan metode TF-IDF untuk mengubah deskripsi produk menjadi representasi numerik:

  * **`TfidfVectorizer()`**: digunakan untuk mengubah teks menjadi vektor angka berdasarkan nilai **TF-IDF** (Term Frequency–Inverse Document Frequency), yaitu skor yang menunjukkan seberapa penting kata dalam suatu dokumen relatif terhadap semua dokumen.
  * **`stop_words='english'`**: menghapus kata-kata umum dalam bahasa Inggris (seperti "the", "is", "and") agar tidak memengaruhi pemodelan.
  * **`fit_transform(df['corpus_clean'])`**: mempelajari dan mengubah seluruh kolom `corpus_clean` menjadi **matriks TF-IDF**.
  * **`tfidf_matrix.shape`**: menampilkan ukuran matriks hasil, dalam format `(jumlah dokumen, jumlah kata unik)`.

  **Hasil**:

  * `(2533, 2381)` artinya ada **2533 produk** dan **2381 kata unik** yang mewakili fitur dalam deskripsi produk.

  **Tujuan**: mengubah teks `corpus_clean` menjadi bentuk numerik yang bisa dihitung kemiripannya antar produk, sebagai dasar dalam sistem **Content-Based Recommendation**.

7. Melakukan scaling data jika diperlukan untuk model kolaboratif.

  ![image](https://github.com/user-attachments/assets/fb260fb2-ab00-46e2-9dd5-d35247e6fe59)

  Pada tahap ini, dilakukan normalisasi fitur numerik agar memiliki skala yang seragam sebelum digunakan dalam model:

  * **`MinMaxScaler()`**: digunakan untuk mengubah nilai numerik ke rentang **0 hingga 1** berdasarkan rumus:
  $(x - min) / (max - min)$
  * **`scaler.fit_transform()`**: mempelajari nilai minimum dan maksimum dari kolom, lalu menerapkan transformasi.
  * **`[['ratings', 'price_clean']]`**: dua fitur numerik yang akan dinormalisasi, yaitu `ratings` dan harga (`price_clean`).
  * **`df[['ratings_scaled', 'price_scaled']]`**: hasil transformasi disimpan dalam dua kolom baru.

  **Tujuan**: menstandarkan skala fitur agar **nilai rating dan harga tidak mendominasi** dalam proses perhitungan kemiripan atau algoritma pembelajaran mesin lainnya. Ini penting dalam model **Content-Based Filtering** atau kombinasi fitur numerik dan teks.

![image](https://github.com/user-attachments/assets/725234df-7b8d-43d5-a6c7-201e2031a4fb)

Pada tahap ini, dilakukan instalasi library **NumPy** dengan versi 1.24.4:

### Penjelasan:

* **`!pip install`**: perintah untuk menginstal package Python menggunakan `pip` (Python Package Installer), biasanya digunakan di lingkungan seperti Jupyter Notebook atau Google Colab.
* **`numpy==1.24.4`**:

  * `numpy`: adalah library Python untuk komputasi numerik yang efisien, seperti operasi matriks, array, dan aljabar linear.
  * `==1.24.4`: menginstruksikan `pip` untuk **menginstal versi 1.24.4 secara spesifik**, bukan versi terbaru.


### Tujuan:

* **Menjamin kompatibilitas** antara NumPy dan library lain dalam proyek (misalnya, `scikit-learn`, `pandas`, atau `surprise` untuk collaborative filtering).
* Beberapa library mungkin **tidak kompatibel dengan versi NumPy terbaru**, sehingga versi 1.24.4 dipilih untuk **menghindari error atau bug** saat menjalankan model rekomendasi atau matrix factorization.

![image](https://github.com/user-attachments/assets/2acea00b-2be5-4a3f-9888-37ce084ec47b)

Pada tahap ini, dilakukan instalasi library **`scikit-surprise`**:

Penjelasan:

* **`!pip install`**: Perintah untuk menginstal library di Python menggunakan package manager `pip`.
* **`scikit-surprise`**:

  * Merupakan library Python untuk membangun dan menganalisis **sistem rekomendasi berbasis Collaborative Filtering**.
  * Nama lengkapnya adalah **`Surprise` (Simple Python RecommendatIon System Engine)**.
  * Mendukung berbagai algoritma seperti **SVD, KNNBasic, BaselineOnly**, dan lain-lain.

Tujuan:

* Menggunakan algoritma **Collaborative Filtering** berbasis **matrix factorization** (seperti **SVD**) untuk merekomendasikan produk kepada pengguna.
* `scikit-surprise` menyediakan **API sederhana** untuk:

  * Membuat model rekomendasi dari data user-item (misalnya rating).
  * Melakukan evaluasi dengan cross-validation.
  * Memberikan **rekomendasi personalisasi** berdasarkan data pengguna yang tersedia.

![image](https://github.com/user-attachments/assets/f0a0cd6c-a177-4061-b4b8-fccd20d34e37)

Pada tahap ini, dilakukan import library yang dibutuhkan untuk membangun dan mengevaluasi model **Collaborative Filtering** menggunakan pendekatan SVD:

* **`from surprise import SVD, Dataset, Reader`**:

  * `SVD`: algoritma **matrix factorization** yang digunakan untuk membangun sistem rekomendasi berdasarkan pola interaksi pengguna.
  * `Dataset`: modul untuk memuat dan memproses data rating pengguna.
  * `Reader`: digunakan untuk mendefinisikan format data rating (misalnya skala minimum dan maksimum).

* **`from surprise.model_selection import train_test_split`**:

  * `train_test_split`: digunakan untuk **membagi data menjadi data latih dan data uji**, guna mengevaluasi performa model.

* **`from surprise import accuracy`**:

  * `accuracy`: modul untuk menghitung metrik evaluasi seperti **RMSE** atau **MAE**, guna menilai seberapa baik prediksi model dibandingkan rating aktual.

* **`import pandas as pd`**:

  * `pandas`: library utama untuk **manipulasi dan analisis data**, digunakan untuk memproses dataset sebelum dimasukkan ke dalam pipeline `surprise`.


* **Tujuan**: menyiapkan semua library penting yang dibutuhkan untuk membangun, melatih, dan mengevaluasi sistem rekomendasi berbasis **Collaborative Filtering (SVD)** menggunakan data rating dari pengguna.

![image](https://github.com/user-attachments/assets/c0cc3405-6f41-4180-9e3b-f20d26ace420)

Pada tahap ini, dilakukan simulasi data pengguna dan persiapan dataset untuk membangun model **Collaborative Filtering** menggunakan library `surprise`:


* **`df['user_id'] = ['user_' + str(i % 50) for i in range(len(df))]`**:

  * Membuat kolom baru bernama `user_id` yang berisi **50 pengguna fiktif** (`user_0` sampai `user_49`) dengan mendistribusikan secara acak berdasarkan indeks.
  * Simulasi ini dilakukan karena dataset asli **tidak memiliki data pengguna**.


* **`reader = Reader(rating_scale=(df['ratings'].min(), df['ratings'].max()))`**:

  * Membuat objek `Reader` dari library `surprise` untuk mendefinisikan **skala rating**.
  * `rating_scale=(min, max)` digunakan agar model tahu nilai minimum dan maksimum dari kolom `ratings` (misalnya dari 2.9 sampai 5.0).


* **`data = Dataset.load_from_df(df[['user_id', 'name', 'ratings']], reader)`**:

  * Mengubah DataFrame `df` menjadi format yang kompatibel dengan library `surprise`.
  * Hanya menggunakan kolom `user_id`, `name` (sebagai item), dan `ratings`.


* **`trainset, testset = train_test_split(data, test_size=0.2, random_state=4)`**:

  * Membagi data menjadi **data latih (80%)** dan **data uji (20%)**.
  * `random_state=4` digunakan untuk menjamin hasil pembagian yang konsisten (reproducible).


**Tujuan**:

Menyiapkan data dalam format yang dibutuhkan oleh library `surprise`, dengan menyimulasikan pengguna agar model **Collaborative Filtering (SVD)** dapat dibangun dan diuji secara realistis meskipun dataset asli tidak memiliki informasi user.

### Modeling

#### Content-based Filtering

* Menggunakan TF-IDF pada kolom `corpus`.
* Menghitung similarity antar produk menggunakan cosine similarity.
* Menghasilkan top-N rekomendasi berdasarkan item yang sedang dilihat pengguna.

### Content-Based Filtering – Penjelasan Cara Kerja dan Tujuan

#### **Cara Kerja:**

1. **Menggunakan TF-IDF pada kolom `corpus`:**

   * Setiap deskripsi produk (`corpus`) diubah menjadi vektor numerik menggunakan **TF-IDF (Term Frequency–Inverse Document Frequency)**.
   * TF-IDF menekankan kata-kata penting yang unik di setiap produk, dan mengurangi pengaruh kata-kata umum.

2. **Menghitung similarity antar produk menggunakan cosine similarity:**

   * Setelah semua deskripsi dikonversi menjadi vektor, sistem menghitung **kemiripan antar produk** menggunakan **cosine similarity**, yang mengukur sudut antara dua vektor (nilai antara 0 dan 1).
   * Produk dengan nilai cosine similarity tinggi berarti sangat mirip secara deskripsi.

3. **Menghasilkan top-N rekomendasi:**

   * Ketika pengguna melihat satu produk, sistem mencari produk lain dengan nilai cosine similarity tertinggi terhadap produk tersebut.
   * Sistem akan merekomendasikan **N produk teratas** (misalnya 5 atau 10) yang paling mirip.


Pada tahap ini, dilakukan proses pembuatan sistem rekomendasi berbasis **Content-Based Filtering** menggunakan **cosine similarity** antar vektor TF-IDF dari deskripsi produk (kemungkinan dari kolom `corpus`). Pertama, dihitung nilai kemiripan (cosine similarity) antar semua produk, lalu disimpan dalam bentuk DataFrame `similarity_df` yang indeks dan kolomnya adalah nama produk. Selanjutnya, dibuat fungsi `get_recommendations()` yang menerima nama produk dan mengembalikan top-N produk paling mirip berdasarkan nilai similarity tertinggi, dengan mengecualikan produk itu sendiri. Fungsi ini akan menampilkan nama, rating, dan harga dari produk-produk yang direkomendasikan. Contohnya, untuk produk **"REDMI Note 12 Pro 5G (Onyx Black, 128 GB)"**, sistem akan mengembalikan 5 varian produk serupa (misalnya beda warna atau kapasitas) yang memiliki deskripsi fitur mirip berdasarkan hasil pemrosesan TF-IDF.

Output :

![image](https://github.com/user-attachments/assets/3f1e0389-b5dc-4fae-8189-5dc45e5eb10d)

Output tersebut menunjukkan **Top 5 rekomendasi produk smartphone** yang mirip dengan:

### Penjelasan:

* Semua produk adalah **REDMI Note 12 Pro 5G**, menandakan kesamaan fitur inti.
* Variasi yang membedakan hanya pada **warna dan storage (128 GB vs 256 GB)**.
* **Rating semuanya sama**: 4.2
* **Harga** sedikit bervariasi tergantung pada varian:

  * 128 GB: sekitar 23.999.000 – 24.999.000
  * 256 GB: 26.999.000

### Kesimpulan:

Rekomendasi ini menyarankan **produk dengan spesifikasi dan fitur yang hampir identik**, cocok untuk user yang mempertimbangkan varian warna atau kapasitas penyimpanan lain dari model yang sama. Ini umum dilakukan dalam sistem rekomendasi berbasis konten (content-based), karena mempertimbangkan **kemiripan fitur/corpus teks produk** daripada popularitas pengguna lain.

#### **Tujuan:**

* Memberikan rekomendasi **produk-produk serupa berdasarkan konten** atau fitur yang dimiliki produk, seperti spesifikasi teknis, sistem operasi, RAM, prosesor, dll.
* Tidak bergantung pada perilaku pengguna lain atau rating, sehingga cocok untuk:

  * Produk baru (cold start item)
  * Pengguna yang belum banyak berinteraksi (anonymous user browsing)
* Memberikan rekomendasi yang **lebih personal dan relevan** dengan produk yang sedang dilihat pengguna.


### Collaborative Filtering

* Menggunakan pendekatan matrix factorization (SVD).
* Membangun matriks user-item berdasarkan `ratings`.
* Menghasilkan top-N rekomendasi untuk pengguna berdasarkan pola pengguna lain.

#### Collaborative Filtering – Penjelasan Cara Kerja dan Tujuan

#### **Cara Kerja:**

1. **Menggunakan pendekatan matrix factorization (SVD):**

   * Collaborative Filtering berbasis **SVD (Singular Value Decomposition)** memecah **matriks user-item** menjadi tiga matriks terpisah untuk menemukan **hubungan tersembunyi** antara pengguna dan produk.
   * SVD mengurangi dimensi data untuk mempermudah prediksi **preferensi pengguna terhadap produk** yang belum diberi rating.

2. **Membangun matriks user-item berdasarkan ratings:**

   * Matriks ini menyimpan informasi interaksi antara **pengguna** dan **produk** (biasanya berupa rating).
   * Baris = pengguna, kolom = produk, nilai = rating. Jika tidak ada interaksi, nilainya kosong (NaN atau 0).
   * Matriks ini menjadi input untuk algoritma SVD.

3. **Menghasilkan top-N rekomendasi untuk pengguna:**

   * Setelah matriks didekomposisi dan prediksi dilakukan, sistem **mengisi celah kosong** (produk yang belum dinilai pengguna).
   * Lalu, sistem memilih **N produk dengan skor prediksi tertinggi** untuk diberikan sebagai rekomendasi.


#### **Tujuan:**

* Memberikan **rekomendasi yang dipersonalisasi** berdasarkan pola dan perilaku pengguna lain.
* Tidak memerlukan informasi konten produk secara eksplisit, cukup berdasarkan data interaksi pengguna (misalnya rating).
* Menyediakan **rekomendasi lintas preferensi**, seperti menyarankan produk yang disukai oleh pengguna lain yang mirip, meskipun produk tersebut sangat berbeda dari yang biasa dilihat pengguna.
* Efektif untuk menangkap **tren kolektif dan selera umum pengguna**.

![image](https://github.com/user-attachments/assets/2d4436ac-0f71-4257-9993-05be474da5a2)

Pada tahap ini, dilakukan **pelatihan dan evaluasi model Collaborative Filtering** menggunakan algoritma **SVD (Singular Value Decomposition)** dari library `surprise`.

*Penjelasan* :

**`model = SVD()`**

* Membuat objek model rekomendasi menggunakan **algoritma SVD**.
* SVD merupakan teknik **matrix factorization** yang memetakan user dan item ke dalam **ruang vektor laten**, lalu menghitung prediksi rating berdasarkan kemiripan vektor.

**`model.fit(trainset)`**

* Melatih model menggunakan **data pelatihan** (`trainset`) yang sebelumnya telah dibagi.
* Model mempelajari pola interaksi antara user dan item berdasarkan rating yang tersedia.

**`predictions = model.test(testset)`**

* Menggunakan **data uji (`testset`)** untuk menguji seberapa baik model memprediksi rating yang tidak terlihat selama pelatihan.
* Menghasilkan daftar prediksi rating untuk setiap pasangan `(user, item)` di data uji.

**`rmse = accuracy.rmse(predictions)`**

* Menghitung **RMSE (Root Mean Squared Error)**, yaitu metrik evaluasi yang mengukur **rata-rata selisih kuadrat antara rating asli dan rating prediksi**.
* Semakin kecil nilai RMSE, semakin baik performa model.

**Tujuan**:

Mengukur **akurasi prediksi model Collaborative Filtering berbasis SVD** dengan mengevaluasi seberapa dekat rating yang diprediksi dibandingkan rating sebenarnya.

**Output: `RMSE: 0.2108`**

* Ini menunjukkan bahwa **model cukup akurat**, karena nilai RMSE-nya rendah (mendekati 0).

![image](https://github.com/user-attachments/assets/1ee79566-c0b0-4bf3-b80a-7db7b3f512ee)

Pada tahap ini, dilakukan proses untuk **menghasilkan top-N rekomendasi produk untuk setiap pengguna** berdasarkan model **Collaborative Filtering (SVD)** yang telah dilatih:


**`from collections import defaultdict`**

* Digunakan untuk membuat struktur data `defaultdict` yang secara otomatis menginisialisasi nilai default (dalam hal ini, list kosong) untuk setiap key baru.


**`def get_top_n(predictions, n=5):`**

* Fungsi ini digunakan untuk **mengelompokkan hasil prediksi berdasarkan user**, lalu **mengambil top-N item dengan prediksi rating tertinggi** untuk masing-masing user.

  * `predictions`: hasil prediksi dari model untuk pasangan `(user, item)`.
  * `top_n[uid]`: menyimpan daftar item dan rating yang diprediksi untuk user `uid`.
  * `sorted(..., reverse=True)[:n]`: menyortir berdasarkan nilai estimasi rating tertinggi dan mengambil N teratas.


**Prediksi untuk item yang belum pernah dilihat user:**

* Mendapatkan semua item dan user yang ada.

* Mengidentifikasi produk yang **sudah pernah diberi rating oleh user** dari data training (`trainset`).

* Menghitung produk **yang belum pernah dilihat** oleh user, yang menjadi **target rekomendasi**.

* Melakukan **prediksi rating** untuk semua pasangan user-item yang belum pernah dilihat.

* Mengambil **Top-5 item** dengan prediksi rating tertinggi untuk setiap user menggunakan fungsi `get_top_n`.


**Output:**

* Menampilkan **Top-5 produk** yang direkomendasikan untuk user tertentu, disusun berdasarkan **rating prediksi tertinggi**.


**Tujuan**:

* Memberikan **rekomendasi produk personalisasi untuk setiap user** berdasarkan pola interaksi pengguna lain terhadap produk yang sama.
* Memungkinkan sistem untuk **merekomendasikan item yang belum pernah dilihat**, namun diprediksi disukai oleh user tersebut.

### Evaluation

#### Content-based Filtering

* **Precision\@K** dan manual review dari rekomendasi yang diberikan.

![image](https://github.com/user-attachments/assets/54e83173-0a07-4889-be75-c992dece708a)

Pada tahap ini, dilakukan proses untuk menghitung metrik evaluasi Precision\@K terhadap hasil rekomendasi dari Content-based Filtering.

Fungsi:

def precision\_at\_k\_content\_based(product\_name, top\_n=5, threshold=4.0):

Fungsi ini digunakan untuk:

* Menghasilkan top-N rekomendasi berdasarkan item yang dilihat pengguna.
* Mengukur berapa banyak dari rekomendasi tersebut yang memiliki rating ≥ threshold.
* Menghitung Precision\@K, yaitu proporsi item yang dianggap relevan dalam top-N rekomendasi.

Parameter:

* product\_name: Nama produk yang sedang dilihat pengguna (sebagai input konten dasar untuk rekomendasi).
* top\_n: Jumlah maksimum item yang akan direkomendasikan.
* threshold: Nilai ambang rating minimum agar suatu item dianggap relevan (default = 4.0).

Langkah-langkah dalam fungsi:

1. recommendations = get\_recommendations(product\_name, top\_n=top\_n)

* Memanggil fungsi get\_recommendations() (berbasis cosine similarity dari TF-IDF) untuk mendapatkan top-N produk yang paling mirip dengan produk input.

2. if recommendations is None or recommendations.empty:

* Jika tidak ada rekomendasi yang dihasilkan (karena item tidak ditemukan atau data kosong), Precision\@K diset ke 0 dan hasil rekomendasi dikembalikan kosong.

3. relevant = recommendations\['ratings'] >= threshold

* Menentukan mana dari item rekomendasi yang dianggap relevan berdasarkan ambang batas rating.

4. precision = relevant.sum() / top\_n

* Menghitung Precision\@K = jumlah item relevan / jumlah item yang direkomendasikan (top\_n).

Output:

* precision: Nilai Precision\@K (float), antara 0 dan 1.
* recommendations: DataFrame dari hasil rekomendasi untuk ditinjau atau ditampilkan.

Tujuan:

* Mengevaluasi seberapa efektif rekomendasi berbasis konten dalam menyarankan item yang disukai pengguna.
* Digunakan untuk membandingkan performa dengan pendekatan lain (misalnya Collaborative Filtering).

Contoh penggunaan:

precision, recs = precision\_at\_k\_content\_based("OPPO F11 Pro", top\_n=5)
print("Precision\@K:", precision)
print(recs)

![image](https://github.com/user-attachments/assets/60675fe7-463a-476e-af89-64270211235a)

Pada tahap ini, dilakukan proses untuk menghasilkan top-N produk yang paling mirip dengan produk tertentu menggunakan pendekatan Content-based Filtering berbasis kemiripan (similarity matrix).

Fungsi:


def get_recommendations(product_name, top_n=5):


Fungsi ini digunakan untuk:

* Mengambil skor kemiripan (similarity score) antara produk yang sedang dilihat pengguna dan semua produk lainnya.
* Menyusun daftar top-N produk yang paling mirip (dengan skor tertinggi).
* Mengembalikan informasi penting seperti nama produk, rating, dan harga dari hasil rekomendasi.

Parameter:

* `product_name`: Nama produk acuan (string), yaitu produk yang sedang dilihat pengguna.
* `top_n`: Jumlah maksimum produk mirip yang akan direkomendasikan (default = 5).

Langkah-langkah dalam fungsi:

1. Cek apakah produk tersedia dalam similarity matrix:


if product_name not in similarity_df.columns:
    print(f"Produk '{product_name}' tidak ditemukan dalam similarity matrix.")
    return pd.DataFrame()


* Jika nama produk tidak ditemukan, fungsi akan mengembalikan DataFrame kosong dan mencetak pesan peringatan.

2. Ambil skor kemiripan dari `similarity_df`:


sim_scores = similarity_df[product_name]


* Mengambil kolom skor kemiripan antara `product_name` dan semua produk lainnya dalam bentuk Series.

3. Konversi ke Series jika `sim_scores` berupa DataFrame:


if isinstance(sim_scores, pd.DataFrame):
    sim_scores = sim_scores.iloc[:, 0]


* Penanganan ekstra untuk menghindari error jika `similarity_df` memiliki lebih dari satu kolom dengan nama yang sama.

4. Urutkan skor kemiripan secara menurun (descending):


sim_scores = sim_scores.sort_values(ascending=False)


* Produk dengan skor kemiripan tertinggi berada di atas.

5. Ambil top-N item (dilewati baris pertama karena itu adalah dirinya sendiri):


top_similar = sim_scores.iloc[1:top_n+1]


* Menghindari rekomendasi dirinya sendiri.

6. Ambil detail produk yang direkomendasikan:


recommendations = df[df['name'].isin(top_similar.index)][['name', 'ratings', 'price_clean']]


* Mengembalikan DataFrame berisi nama produk, rating, dan harga dari top-N produk mirip.

Output:

* `recommendations`: DataFrame yang berisi daftar produk yang paling mirip dengan produk acuan, lengkap dengan rating dan harga.

Tujuan:

* Memberikan rekomendasi produk alternatif yang mirip dari sisi konten (berdasarkan fitur teks/corpus).
* Menjadi dasar dalam evaluasi Precision\@K untuk pendekatan Content-based Filtering.

Contoh Hasil Evaluasi:


produk_dilihat = "REDMI Note 12 Pro 5G (Onyx Black, 128 GB)"
precision, rekomendasi = precision_at_k_content_based(produk_dilihat, top_n=5)


Output:


Precision@5 untuk 'REDMI Note 12 Pro 5G (Onyx Black, 128 GB)': 1.00

Top 5 Rekomendasi (Manual Review):
                                                name  ratings  price_clean
69    REDMI Note 12 Pro 5G (Stardust Purple, 128 GB)      4.2      23999.0
305   REDMI Note 12 Pro 5G (Glacier Blue, 128 GB)      4.2      23999.0
459   REDMI Note 12 Pro 5G (Onyx Black, 256 GB)      4.2      26999.0
619   REDMI Note 12 Pro 5G (Stardust Purple, 128 GB)      4.2      24999.0
1465  REDMI Note 12 Pro 5G (Glacier Blue, 128 GB)      4.2      24999.0


Catatan:

* `Precision@5 = 1.00` menunjukkan bahwa semua hasil rekomendasi termasuk dalam kategori produk relevan atau serupa secara konten.
* Manual review tetap disarankan untuk memastikan rekomendasi relevan dengan konteks kebutuhan pengguna.


#### Collaborative Filtering

* **RMSE** (Root Mean Squared Error) untuk mengukur akurasi prediksi rating.

![image](https://github.com/user-attachments/assets/59c35f58-0acd-4698-bb25-8442f1b1e6dc)

Pada tahap ini, dilakukan evaluasi performa model Collaborative Filtering berbasis SVD dengan menggunakan metrik RMSE (Root Mean Squared Error), yang mengukur seberapa akurat model memprediksi rating dibandingkan rating aktual dari pengguna.

Library yang Digunakan:

from surprise import accuracy

* accuracy: Modul dari pustaka surprise yang menyediakan fungsi untuk menghitung metrik evaluasi seperti:

  * RMSE (Root Mean Squared Error)
  * MAE (Mean Absolute Error)
* Fungsi utama yang digunakan di sini adalah accuracy.rmse(predictions), yang membandingkan prediksi model dengan nilai rating asli.

Prediksi Rating:

predictions = model.test(testset)

* model.test(testset): Menggunakan data uji (testset) untuk memprediksi rating semua pasangan user-item yang ada di testset.
* predictions: Hasil prediksi dari model SVD, berupa list of Prediction objects, masing-masing berisi:

  * user ID
  * item ID
  * rating aktual
  * rating prediksi
  * detail lainnya

Evaluasi dengan RMSE:

rmse = accuracy.rmse(predictions)

* Fungsi ini menghitung nilai Root Mean Squared Error antara rating aktual dan prediksi dari model.
* Nilai RMSE yang lebih kecil menunjukkan prediksi yang lebih akurat dan performa model yang lebih baik.

Output:

print(f"RMSE untuk Collaborative Filtering: {rmse:.4f}")

Contoh Hasil:
RMSE: 0.2124
RMSE untuk Collaborative Filtering: 0.2124

Tujuan:

* Menilai seberapa baik model Collaborative Filtering memprediksi rating pengguna.
* Digunakan sebagai indikator utama dalam mengevaluasi model SVD:

  * RMSE ≈ 0 → Prediksi sangat akurat
  * RMSE besar → Prediksi tidak mendekati nilai rating sebenarnya

Kesimpulan:

* Nilai RMSE yang rendah (misalnya di bawah 0.5 dalam skala 1–5) menandakan bahwa model cukup efektif dalam memahami preferensi pengguna.
* Evaluasi ini penting untuk membandingkan performa dengan pendekatan lain seperti Content-based Filtering (yang menggunakan Precision\@K).

#### Perbandingan Model

![image](https://github.com/user-attachments/assets/57c33945-db62-46f8-b79b-a3769fa4651d)

\=== EVALUASI MODEL REKOMENDASI ===

Content-Based Filtering (Precision\@5): 1.00

Collaborative Filtering (RMSE): 0.2107

Top 5 Rekomendasi Content-Based:

| name                                           | ratings | price\_clean |
| ---------------------------------------------- | ------- | ------------ |
| REDMI Note 12 Pro 5G (Stardust Purple, 128 GB) | 4.2     | 23999.0      |
| REDMI Note 12 Pro 5G (Glacier Blue, 128 GB)    | 4.2     | 23999.0      |
| REDMI Note 12 Pro 5G (Onyx Black, 256 GB)      | 4.2     | 26999.0      |
| REDMI Note 12 Pro 5G (Stardust Purple, 128 GB) | 4.2     | 24999.0      |
| REDMI Note 12 Pro 5G (Glacier Blue, 128 GB)    | 4.2     | 24999.0      |

Tujuan Evaluasi:

Evaluasi ini dilakukan untuk membandingkan performa dua pendekatan sistem rekomendasi:

* Content-Based Filtering → berbasis kesamaan konten produk.
* Collaborative Filtering (SVD) → berbasis interaksi pengguna dengan produk.

Evaluasi Content-Based Filtering:

Precision\@K digunakan untuk mengevaluasi relevansi hasil rekomendasi berdasarkan rating pengguna.

* Fungsi `precision_at_k_content_based()` akan:

  * Mengambil top-N rekomendasi yang paling mirip dengan produk dilihat (berdasarkan TF-IDF + cosine similarity).
  * Menghitung Precision\@5 = proporsi item dalam top-5 yang memiliki rating ≥ threshold (misal 4.0).
* Hasil evaluasi:

  * Precision\@5 = 1.00 → Artinya seluruh item dalam rekomendasi dinilai sangat relevan oleh pengguna.

Evaluasi Collaborative Filtering:

RMSE digunakan untuk mengukur akurasi prediksi rating berdasarkan interaksi user-item.

* Fungsi `accuracy.rmse(predictions)` akan:

  * Menghitung deviasi akar kuadrat rata-rata antara rating aktual dan prediksi model SVD.
* Hasil evaluasi:

  * RMSE = 0.2107 → Nilai ini sangat rendah, menandakan prediksi rating oleh model cukup akurat.

Interpretasi Top-5 Rekomendasi Content-Based:

Dari hasil rekomendasi:

* Semua produk sangat mirip secara konten (seri dan varian warna/memori dari REDMI Note 12 Pro 5G).
* Semua memiliki rating tinggi (≥ 4.2) → menunjukkan kualitas rekomendasi yang relevan.
* Harga bervariasi → memberi opsi produk serupa dengan rentang harga yang berbeda.

Kesimpulan Umum:

* Model Content-Based Filtering sangat berhasil dalam menemukan item yang serupa dan disukai (Precision tinggi).
* Model Collaborative Filtering juga memberikan prediksi akurat (RMSE rendah).
* Kedua pendekatan saling melengkapi:

  * Content-Based cocok untuk pengguna baru (cold start).
  * Collaborative Filtering unggul dalam menangkap preferensi implisit pengguna berdasarkan perilaku rating historis.


---

*Catatan: laporan dapat dilengkapi dengan visualisasi, kode snippet, atau tabel evaluasi sesuai kebutuhan.*
