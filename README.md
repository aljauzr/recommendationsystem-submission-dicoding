# Laporan Proyek Machine Learning - Al Jauzi Abdurrohman

## Project Overview

Di era digital saat ini, pengguna menghadapi tantangan dalam memilih konten yang sesuai dari sekian banyak pilihan yang tersedia. Salah satu solusi penting dalam mengatasi masalah ini adalah sistem rekomendasi. Sistem rekomendasi banyak digunakan dalam berbagai platform seperti e-commerce, layanan streaming, dan media sosial untuk memberikan rekomendasi yang relevan dan personal kepada pengguna.

Dalam proyek ini, digunakan MovieLens Small Dataset, yang merupakan dataset standar dalam penelitian sistem rekomendasi. Dataset ini berisi data mengenai pengguna, film, rating, serta genre dari film tersebut. Proyek ini bertujuan untuk membangun sistem rekomendasi film berdasarkan dua pendekatan utama, yaitu:
1. Content-Based Filtering (CBF): merekomendasikan film yang mirip dengan film yang disukai pengguna berdasarkan fitur konten (genre).
2. Collaborative Filtering (CF): merekomendasikan film berdasarkan kesamaan pola rating antar pengguna.

## Business Understanding

Sistem rekomendasi dapat meningkatkan pengalaman pengguna dalam memilih film, membantu platform dalam mempertahankan pengguna, serta meningkatkan waktu tonton dan kepuasan. Oleh karena itu, membangun sistem rekomendasi yang akurat menjadi sangat penting bagi bisnis yang bergerak di bidang penyediaan konten hiburan digital.

### Problem Statements
1. Bagaimana cara merekomendasikan film berdasarkan preferensi konten yang mirip dengan film yang disukai pengguna?
2. Bagaimana cara memberikan rekomendasi personal kepada pengguna berdasarkan pola rating pengguna lain yang memiliki selera serupa?
3. Bagaimana membandingkan efektivitas pendekatan Content-Based Filtering dan Collaborative Filtering dalam memberikan rekomendasi yang relevan?

### Goals
1. Membangun sistem rekomendasi menggunakan Content-Based Filtering dengan memanfaatkan informasi genre dari film yang pernah ditonton dan disukai pengguna.
2. Membangun sistem rekomendasi menggunakan Collaborative Filtering berbasis matrix factorization yang memanfaatkan pola rating pengguna lain.
3. Membandingkan performa dari kedua pendekatan untuk mengetahui mana yang lebih baik dalam konteks MovieLens dataset.

### Solution statements
Untuk mencapai tujuan proyek, digunakan dua pendekatan berikut:
1. Content-Based Filtering:
- Menggunakan data movies.csv untuk mengekstrak fitur film berdasarkan kolom genres.
- Menerapkan TF-IDF Vectorization atau MultiLabelBinarizer untuk mengubah genre menjadi vektor fitur.
- Menghitung kemiripan antar film menggunakan cosine similarity.
- Membangun sistem rekomendasi berdasarkan film yang pernah diberi rating tinggi oleh pengguna.
2. Collaborative Filtering:
- Menggunakan data ratings.csv yang memuat informasi userId, movieId, dan rating.
- Membangun user-item matrix dan menerapkan matrix factorization (misalnya SVD atau ALS).
- Menyediakan rekomendasi berdasarkan prediksi rating tertinggi dari model terhadap item yang belum diberi rating oleh pengguna.

Dengan menggunakan dua pendekatan ini, proyek bertujuan memberikan wawasan yang lebih luas tentang kekuatan dan kelemahan masing-masing metode dalam konteks sistem rekomendasi film.

## Data Understanding
Data yang digunakan adalah dataset rating terhadap film dari MovieLens Latest Dataset (recommended for education and development) (URL: https://grouplens.org/datasets/movielens/). Pada sel kode notebook, dataset ini dapat diunduh menggunakan kode berikut:
```sh
# Download file zip dari URL
!wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# Ekstrak isi file ZIP
!unzip -o ml-latest-small.zip
```
Dataset ini berisikan 4 file csv (3 csv merupakan variabel untuk film dan 1 csv merupakan data rating user terhadap film) dan 1 file README.txt yang berisi deskripsi dataset.

| Variabel | Jumlah Baris | Deskripsi |
| ------ | ------ | ------ |
| movies | 9742 | Metadata film, yaitu judul film dan genre film tersebut |
| links | 9742 | ID link film di imdb.com dan themoviedb.com |
| tags | 1589 | Tag pada film yang diberikan pengguna |

Untuk mengetahui variabel-variabel tersebut lebih mendalam, kita akan melakukan Exploratory Data Analysis dengan teknik Univariate Analysis.

### EDA - Univariate Analysis (movies)
Pertama, mari kita eksplorasi variabel movies, yaitu metadata film. Mari kita lihat kolom movies menggunakan fungsi info().
```sh
movies.info()
```
Output:
| # | Non-Null Count | Dtype |
| ------ | ------ | ------ |
| movieId | 9742 non-null | int64 |
| title | 9742 non-null | object |
| genres | 9742 non-null | object |

Berdasarkan output di atas, kita dapat mengetahui bahwa terdapat 9742 film yang ada pada dataset tersebut dan tidak ada missing value. Berikut adalah penjelasan dari kolom yang ada:
- `movieId` : identifier unik untuk setiap film
- `title` : judul film
- `genres` : genre film

Untuk melihat sampel datanya, mari kita terapkan fungsi head()
```sh
movies.head()
```
Output:
| movieId | title | genres |
| ------ | ------ | ------ |
| 1	| Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2 | Jumanji (1995) | Adventure\|Children\|Fantasy |
| 3	| Grumpier Old Men (1995)	| Comedy\|Romance |
| 4 | Waiting to Exhale (1995) | Comedy\|Drama\|Romance |
| 5 | Father of the Bride Part II (1995) | Comedy |

Berdasarkan output di atas, satu film bisa terdiri dari beberapa genre yang dipisahkan oleh tanda '|'. Untuk mengetahui ada genre apa saja pada data tersebut, terapkan kode berikut:
```sh
# Pecah genre berdasarkan '|', lalu flatten dan ambil yang unik
unique_genres = set()
for genre_list in movies['genres']:
    genres_split = genre_list.split('|')
    unique_genres.update(genres_split)
# Tampilkan hasil
print(sorted(unique_genres))
```
Output:
```sh
['(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
```

Terdapat 20 jenis genre berbeda dengan nama genre seperti terlihat pada output kode. Perhatikanlah jenis-jenis genre di atas. Pada modul selanjutnya kita akan gunakan data genre ini untuk memprediksi top-N rekomendasi bagi pengguna.
Naumn, terdapat nilai kolom genre = (no genres listed), artinya ada film yang genrenya tidak diketahui. Maka dari itu, kita akan memeriksa terlebih dahulu ada berapa film dengan kondisi tersebut. Terapkan kode berikut.
```sh
# Filter film yang tidak memiliki genre (no genres listed)
no_genre_movies = movies[movies['genres'] == '(no genres listed)']
# Tampilkan jumlah baris
print(f"Jumlah film yang tidak memiliki genre (no genres listed): {len(no_genre_movies)}")
```
Output:
```
Jumlah film yang tidak memiliki genre (no genres listed): 34
```
Jumlah 34 film jika dibandingkan dengan jumlah keseluruhan, yaitu 9742 film terbilang tidak signifikan. Maka dari itu, pada proses Data Preprocessing, data tersebut akan dihilangkan. Selanjutnya, kita eksplorasi variabel links, yaitu link dari film yang ada.

### EDA - Univariate Analysis (links)
Kedua, mari kita eksplorasi variabel links, yaitu ID link film di imdb.com dan themoviedb.com. Mari kita lihat kolom yang terdapat pada links menggunakan fungsi info().
```sh
links.info()
```
Output:
| # | Non-Null Count | Dtype |
| ------ | ------ | ------ |
| movieId | 9742 non-null | int64 |
| imdbId | 9742 non-null | int64 |
| tmdbId | 9734 non-null | float64 |

Berikut adalah penjelasan dari kolom yang ada:
- `movieId` : merupakan nilai identifier pada URL: https://movielens.org/movies/{movieId}
- `imdbId` :  merupakan nilai identifier pada URL: http://www.imdb.com/title/`{imdbId}
- `tmdbId` : merupakan nilai identifier pada URL: https://www.themoviedb.org/movie/{tmdbId}.

Pada kolom tmdbId, terdapat 8 baris yang kosong, namun karena kolom ini tidak diikutsertakan dalam proses perhitungan, maka dapat kita biarkan saja. Untuk melihat sampel datanya, mari kita terapkan fungsi head().
```sh
links.head()
```
Output:
| movieId | imdbId | tmdbId |
| ------ | ------ | ------ |
| 0	| 1 | 114709 | 862.0 |
| 1 | 2 | 113497 | 8844.0 |
| 2 | 3 | 113228 | 15602.0 |
| 3 | 4 | 114885 | 31357.0 |
| 4 | 5	| 113041 | 11862.0 |

### EDA - Univariate Analysis (tags)
Selanjutnya, kita eksplorasi variabel tags, yaitu tag pada film yang diberikan pengguna. Mari kita lihat kolom yang terdapat pada tags menggunakan fungsi info().
```sh
tags.info()
```
Output:
| # | Non-Null Count | Dtype |
| ------ | ------ | ------ |
| userId | 3683 non-null | int64 |
| movieId | 3683 non-null | int64 |
| tag | 3683 non-null | object |
| timestamp | 3683 non-null | int64 |

Berikut adalah penjelasan dari kolom yang ada:
- `userId` : merupakan user yang memberikan tag pada film
- `movieId` :  merupakan film yang diberi tag oleh user
- `tag` : merupakan tag yang diberikan oleh user
- `timestamp` : merupakan waktu ketika pengguna memberikan tag terhadap sebuah film. Nilainya berupa angka dalam format UNIX timestamp, yaitu jumlah detik sejak 1 Januari 1970 (epoch time)

Untuk melihat sampel datanya, mari kita terapkan fungsi head()
```sh
tags.head()
```
Output:
| userId | movieId | tag | timestamp |
| ------ | ------ | ------ | ------ |
| 2 | 60756 | funny | 1445714994 |
| 2 | 60756 | Highly quotable | 1445714996 |
| 2 | 60756 | will ferrell | 1445714992 |
| 2 | 89774 |Boxing story | 1445715207 |
| 2	| 89774 | MMA | 1445715200 |

Baris ke-1 dari output di atas memberikan arti bahwa user dengan userId = 2 memberikan tag 'funny' ke film dengan movieId = 60756 pada 24 Oktober 2015 pukul 14:49:54 UTC (setelah dikonversi dari UNIX timestamp).

Selanjutnya, kita eksplorasi variabel user. Namun, pada dataset ini user tidak memiliki file .csv nya sendiri. Pada ratings.csv, terdapat kolom userId, kolom ini akan kita gunakan untuk mengidentifikasi jumlah user yang telah memberikan rating terhadap film. Terapkan kode berikut.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.