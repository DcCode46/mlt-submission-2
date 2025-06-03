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

### Modeling

#### Content-based Filtering

* Menggunakan TF-IDF pada kolom `corpus`.
* Menghitung similarity antar produk menggunakan cosine similarity.
* Menghasilkan top-N rekomendasi berdasarkan item yang sedang dilihat pengguna.

#### Collaborative Filtering

* Menggunakan pendekatan matrix factorization (SVD).
* Membangun matriks user-item berdasarkan `ratings`.
* Menghasilkan top-N rekomendasi untuk pengguna berdasarkan pola pengguna lain.

### Evaluation

#### Content-based Filtering

* **Precision\@K** dan manual review dari rekomendasi yang diberikan.

#### Collaborative Filtering

* **RMSE** (Root Mean Squared Error) untuk mengukur akurasi prediksi rating.

Kedua model dibandingkan berdasarkan relevansi hasil dan potensi skalabilitas. Model content-based lebih unggul pada cold-start item, sementara model collaborative unggul pada penemuan preferensi tersembunyi pengguna.

---

*Catatan: laporan dapat dilengkapi dengan visualisasi, kode snippet, atau tabel evaluasi sesuai kebutuhan.*
