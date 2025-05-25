## Laporan Proyek Machine Learning - DWI NURCAHYO PURBONEGORO

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
2. Menghapus nilai null pada kolom `corpus`.
3. Membersihkan dan mengonversi kolom `price` menjadi numerik.
4. Menghilangkan baris duplikat.
5. Mengubah semua teks pada `corpus` menjadi huruf kecil dan membersihkannya.
6. Menggunakan TF-IDF Vectorizer untuk mentransformasikan `corpus` menjadi vektor fitur.
7. Melakukan scaling data jika diperlukan untuk model kolaboratif.

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
