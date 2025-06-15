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
- Menerapkan TF-IDF Vectorization untuk mengubah genre menjadi vektor fitur.
- Menghitung kemiripan antar film menggunakan cosine similarity.
- Membangun sistem rekomendasi berdasarkan film yang pernah diberi rating tinggi oleh pengguna.
2. Collaborative Filtering:
- Menggunakan data ratings.csv yang memuat informasi userId, movieId, dan rating.
- Membangun user-item matrix dan menerapkan matrix factorization.
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

Terdapat 20 jenis genre berbeda dengan nama genre seperti terlihat pada output kode. Perhatikanlah jenis-jenis genre di atas. Pada tahap selanjutnya kita akan gunakan data genre ini untuk memprediksi top-N rekomendasi bagi pengguna.
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
- `imdbId` :  merupakan nilai identifier pada URL: http://www.imdb.com/title/{imdbId}
- `tmdbId` : merupakan nilai identifier pada URL: https://www.themoviedb.org/movie/{tmdbId}.

Pada kolom tmdbId, terdapat 8 baris yang kosong, namun karena kolom ini tidak diikutsertakan dalam proses perhitungan, maka dapat kita biarkan saja. Untuk melihat sampel datanya, mari kita terapkan fungsi head().
```sh
links.head()
```
Output:
| movieId | imdbId | tmdbId |
| ------ | ------ | ------ |
| 1 | 114709 | 862.0 |
| 2 | 113497 | 8844.0 |
| 3 | 113228 | 15602.0 |
| 4 | 114885 | 31357.0 |
| 5	| 113041 | 11862.0 |

Berdasarkan output baris pertama, movieId = 1 memiliki link movielens https://movielens.org/movies/1, link imdb http://www.imdb.com/title/114709, dan link tmdb https://www.themoviedb.org/movie/862. Demikian seterusnya.

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

Baris ke-1 dari output di atas memberikan arti bahwa user dengan userId = 2 memberikan tag 'funny' ke film dengan movieId = 60756 pada 24 Oktober 2015 pukul 14:49:54 UTC (setelah dikonversi dari UNIX timestamp). Demikian seterusnya.

### EDA - Univariate Analysis (user)
Selanjutnya, kita eksplorasi variabel user. Namun, pada dataset ini user tidak memiliki file .csv nya sendiri. Pada ratings.csv, terdapat kolom userId, kolom ini akan kita gunakan untuk mengidentifikasi jumlah user yang telah memberikan rating terhadap film. Terapkan kode berikut.
```sh
print('Jumlah user yang memberikan rating: ', len(ratings.userId.unique()))
```
Output:
```sh
Jumlah user yang memberikan rating:  610
```
Berdasarkan output di atas, diketahui terdapat 610 user yang telah memberikan rating terhadap film. Data ini akan digunakan untuk tahap modelling dengan Collaborative Filtering.
Pada tahap modelling dengan Content-Based Filtering nanti, data yang dibutuhkan adalah judul film dan genre. Kita akan menghitung kesamaan (similarity) genre dan judul film kemudian membuat rekomendasi berdasarkan kesamaan ini.

### EDA - Univariate Analysis (ratings)
Selanjutnya, mari kita eksplorasi data yang akan kita gunakan pada model yaitu data ratings. Pertama, kita lihat seperti apa data pada variabel rating dengan fungsi head().
```sh
ratings.head()
```
Output:
| userId | movieId | rating | timestamp |
| ------ | ------ | ------ | ------ |
| 1 | 1	| 4.0 |	964982703 |
| 1	| 3	| 4.0	| 964981247 |
| 1	| 6	| 4.0 |	964982224 |
| 1	| 47 | 5.0 | 964983815 |
| 1	| 50 | 5.0 | 964982931 |

Dari fungsi rating.head(), kita dapat mengetahui bahwa data rating terdiri dari 4 kolom. Kolom-kolom tersebut antara lain:

- `userId` : merupakan ID user
- `movieId` : merupakan ID film
- `rating` : merupakan data rating yang diberikan user terhadap film
- `timestamp` : merupakan waktu ketika pengguna memberikan rating terhadap sebuah film. Nilainya berupa angka dalam format UNIX timestamp, yaitu jumlah detik sejak 1 Januari 1970 (epoch time)

Untuk melihat distribusi rating pada data, gunakan fungsi describe() dengan menerapkan kode berikut:
```sh
ratings.describe()
```
Output:
| | userId | movieId | rating | timestamp |
| ------ | ------ | ------ | ------ | ------ |
| count	| 100836.000000 | 100836.000000 | 100836.000000 | 1.008360e+05 |
| mean | 326.127564 | 19435.295718 | 3.501557 |	1.205946e+09 |
| std |	182.618491 | 35530.987199 |	1.042529 | 2.162610e+08 |
| min | 1.000000 | 1.000000 | 0.500000 | 8.281246e+08 |
| 25% | 177.000000 | 1199.000000 | 3.000000 | 1.019124e+09 |
| 50% | 325.000000 | 2991.000000 | 3.500000 | 1.186087e+09 |
| 75% | 477.000000 | 8122.000000 | 4.000000 | 1.435994e+09 |
| max | 610.000000 | 193609.000000 | 5.000000 |	1.537799e+09 |

Dari output di atas, diketahui bahwa terdapat 100836 total rating dan nilai maksimum rating adalah 5 dan nilai minimumnya adalah 0.5. Artinya, skala rating berkisar antara 0.5 hingga 5. 
Sampai di tahap ini, kita telah memahami variabel-variabel pada data yang kita miliki.

### Data Preprocessing - Memeriksa Missing Value pada Rating
Untuk memeriksa apakah terdapat missing value, jalankan kode berikut.
```sh
ratings.isnull().sum()
```
Output:
| | 0 |
| ------ | ------ |
| userId | 0 |
| movieId |	0 |
| rating | 0 |
| timestamp | 0 |
| title	| 0 |
| genres | 0 |

Tidak ada missing value yang ditemukan, kita dapat lanjut ke tahap berikutnya.

### Data Preprocessing - Menggabungkan Data Rating dengan Fitur Genre Film
Langkah selanjutnya adalah menggabungkan variabel ratings dengan variabel movies, yaitu metadata film yang mengandung judul film dan genrenya. Implementasikan kode berikut.
```sh
movies_ratings = pd.merge(ratings, movies, on='movieId', how='left')
movies_ratings
```
Output:
| userId | movieId | rating | timestamp | title | genres |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 1 | 1	| 4.0 |	964982703 |	Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 1	| 3	| 4.0 |	964981247 |	Grumpier Old Men (1995) | Comedy\|Romance |
| 1	| 6	| 4.0 |	964982224 |	Heat (1995)	| Action\|Crime\|Thriller |
| 1	| 47 | 5.0 | 964983815 | Seven (a.k.a. Se7en) (1995) | Mystery\|Thriller |
| 1	| 50 | 5.0 | 964982931 | Usual Suspects, The (1995) | Crime\|Mystery\|Thriller |
| ... | ... | ... | ... | ... | ... |

Inilah data yang akan kita gunakan untuk membuat sistem rekomendasi. Namun, sebagai trivia, kita akan mencari tahu film apa yang memiliki rating tertinggi, jalankan kode berikut:
```sh
movies_ratings[['movieId', 'title', 'rating']].groupby(['movieId', 'title']).sum().reset_index().sort_values(by='rating', ascending=False)
```
Output:
|  | movieId | title | rating |
| ------ | ------ | ------ | ------ |
| 277 |	318	| Shawshank Redemption, The (1994) | 1404.0 |
| 314 |	356	| Forrest Gump (1994) | 1370.0 |
| 257 |	296	| Pulp Fiction (1994) | 1288.5 |
| 1938 | 2571 | Matrix, The (1999) | 1165.5 |
| 510 |	593	| Silence of the Lambs, The (1991) | 1161.0 |
| ... |	... | ... |	... |

Film yang paling banyak mendapatkan rating adalah film The Shawshank Redemption (1994) dengan jumlah rating 1404. Sebaliknya, film yang paling sedikit mendapatkan jumlah rating adalah fillm Lionheart (1990) dengan jumlah rating 0.5.

Berikutnya, mari kita menuju tahapan Data Preparation.

## Data Preparation

### Mengatasi Missing Value
Pada tahap EDA - Univariate Analysis (movies), kita mendapat informasi 'Jumlah film yang tidak memiliki genre (no genres listed): 34'. Informasi tersebut akan kita tangani dalam tahap ini, yaitu dengan cara menghapus baris yang memiliki genre '(no genres listed)'. Terapkan kode berikut.
```sh
movies_ratings = movies_ratings[movies_ratings['genres'] != '(no genres listed)']
```
Setelah menjalankan kode tersebut, data movies_ratings yang sebelumnya memiliki 100836 baris, sekarang menjadi 100789 baris. Mari kita cek kembali datanya apakah ada missing value atau tidak. Jalankan kode berikut.
```sh
movies_ratings.isnull().sum()
```
Output:
|  | 0 |
| ------ | ------ |
| userId | 0 |
| movieId |	0 |
| rating | 0 |
| timestamp | 0 |
| title | 0 |
| genres | 0 |

Ok, sekarang data kita sudah bersih. Mari lanjutkan ke tahap berikutnya!

### Mempersiapkan Data
Dalam sistem rekomendasi berbasis Content Based Filtering, penting untuk memastikan satu film memiliki satu baris nilai genre (bisa terdiri dari satu genre atau beberapa genre yang dipisahkan oleh tanda '|'). Tujuannya agar sistem dapat merekomendasikan film yang mirip dengan yang disukai user sebelumnya dan tidak terjadi dobel atau rangkap kategori dalam satu film. Dalam model CBF, kita hanya butuh 1 baris per film untuk mengambil fitur filmnya — tidak perlu semua rating dari user. Maka dari itu, kita perlu menghapus nilai kolom movieId yang duplikat dari dataframe movie_ratings.

Sedangkan untuk model Collaborative Filtering, nanti kita akan tetap mempertahankan dataframe movies_ratings.

Berikutnya, kita bisa melanjutkan ke tahap persiapan. Buatlah variabel bernama preparation yang dikhususkan untuk model Content Based Filtering. Jalankan kode berikut.
```sh
preparation = movies_ratings
```
Selanjutnya, kita hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, kita perlu menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, kita membuang data duplikat pada kolom ‘movieId’. Implementasikan kode berikut.
```sh
preparation = preparation.drop_duplicates('movieId')
```
Setelah menjalankan kode tersebut, data preparation yang sebelumnya memiliki 100789 baris, sekarang menjadi 9690 baris. Selanjutnya, kita perlu melakukan konversi data series menjadi list. Dalam hal ini, kita menggunakan fungsi tolist() dari library numpy. Implementasikan kode berikut.
```sh
movie_id = preparation['movieId'].tolist()
movie_title = preparation['title'].tolist()
movie_genres = preparation['genres'].tolist()
```
Tahap berikutnya, kita akan membuat dictionary untuk menentukan pasangan key-value pada data movie_id, movie_title, dan movie_genres yang telah kita siapkan sebelumnya.
```sh
movie_new = pd.DataFrame({
    'id': movie_id,
    'movie_title': movie_title,
    'movie_genres': movie_genres
})
```
Output:
| id | movie_title | movie_genres |
| ------ | ------ | ------ |
| 1	| Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 3	| Grumpier Old Men (1995) |	Comedy\|Romance |
| 6	| Heat (1995) | Action\|Crime\|Thriller |
| 47 | Seven (a.k.a. Se7en) (1995) | Mystery\|Thriller |
| 50 | Usual Suspects, The (1995) |	Crime\|Mystery\|Thriller |

Okay, data kini telah siap untuk dimasukkan ke dalam pemodelan. Yuk, kita lanjut ke tahap berikutnya!

## Modelling
Pada tahap Modelling, kita akan menggunakan dua pendekatan, yaitu **Content-Based Filtering** dan **Collaborative Filtering**.

Pada Content-Based Filtering, rekomendasi film diberikan berdasarkan kemiripan konten dengan film yang telah diberi rating oleh pengguna. Oleh karena itu, kita akan menerapkan teknik TF-IDF untuk mengubah informasi genre menjadi representasi numerik, kemudian menghitung cosine similarity antar film berdasarkan kesamaan genre tersebut. Dengan begitu, kita dapat mengetahui seberapa mirip suatu film dengan film lain yang disukai oleh pengguna.

Sementara itu, pada Collaborative Filtering, pendekatan yang digunakan berfokus pada pola interaksi antara pengguna dan film. Kita perlu melakukan encoding data pengguna dan film, membagi data menjadi training dan testing, lalu melatih model menggunakan data tersebut. Setelah pelatihan selesai, model akan dievaluasi untuk mengetahui seberapa baik kemampuannya dalam memahami preferensi pengguna dan memberikan rekomendasi film yang relevan.

Content-Based Filtering memiliki kelebihan dalam memberikan rekomendasi secara personal tanpa memerlukan data dari pengguna lain, dan tetap bisa memberikan rekomendasi meskipun jumlah pengguna sedikit (cold start problem pada pengguna tidak terlalu berdampak). Namun, kelemahannya adalah model ini terbatas pada item yang memiliki fitur yang jelas dan tidak dapat merekomendasikan item yang sangat berbeda dari preferensi sebelumnya.
Di sisi lain, Collaborative Filtering mampu menemukan pola yang lebih kompleks dalam perilaku pengguna dan dapat merekomendasikan item di luar preferensi konten sebelumnya. Namun, model ini memerlukan data interaksi yang cukup besar agar akurat, dan dapat mengalami masalah cold start ketika menghadapi pengguna atau item baru yang belum memiliki cukup data.

### Content Based Filtering
Sampai di sini, kita telah melewati serangkaian tahapan untuk membuat sistem rekomendasi mulai dari:

1. Data Understanding.
2. Univariate Exploratory Data Analysis.
3. Data Preprocessing.
4. Data Preparation.

Kini, saatnya kita mengembangkan sistem rekomendasi dengan pendekatan Content Based Filtering. Untuk modelling pada tahap ini, kita akan memasukkan data yang telah dimiliki sebelumnya (movie_new) ke dalam variabel data. Jalankan kode berikut.
```sh
data = movie_new
```

### Content Based Filtering - TF-IDF Vectorizer
Pada tahap ini, kita akan membangun sistem rekomendasi sederhana berdasarkan genre film. Teknik yang akan digunakan adalah dengan TF-IDF Vectorizer yang bertujuan untuk menemukan representasi fitur penting dari setiap genre film. 
TF-IDF merupakan singkatan dari Term Frequency - Inverse Document Frequency, yaitu sebuah metode untuk mengubah teks menjadi representasi numerik (vektor) yang bisa digunakan dalam perhitungan kesamaan (similarity). Dalam konteks Content Based Filtering, TF-IDF digunakan untuk merepresentasikan konten film (dalam hal ini genre) ke dalam bentuk vektor numerik. Kita akan menggunakan fungsi TfidfVectorizer() dari library sklearn. Jalankan kode berikut.
```sh
vectorizer = TfidfVectorizer()
# Melakukan perhitungan idf pada data genres
vectorizer.fit(data['movie_genres'])
# Mapping array dari fitur index integer ke fitur nama
vectorizer.get_feature_names_out()
```
Output:
```sh
array(['action', 'adventure', 'animation', 'children', 'comedy', 'crime',
       'documentary', 'drama', 'fantasy', 'fi', 'film', 'horror', 'imax',
       'musical', 'mystery', 'noir', 'romance', 'sci', 'thriller', 'war',
       'western'], dtype=object)
```
Selanjutnya, lakukan fit dan transformasi ke dalam bentuk matriks.
```
tfidf_matrix = vectorizer.fit_transform(data['movie_genres'])
tfidf_matrix.shape
```
Output:
```
(9690, 21)
```
Perhatikanlah, matriks yang kita miliki berukuran (9690, 21). Nilai 9690 merupakan ukuran data dan 21 merupakan matriks genre. Untuk menghasilkan vektor tf-idf dalam bentuk matriks, kita menggunakan fungsi todense(). Jalankan kode berikut.
```sh
tfidf_matrix.todense()
```
Output:
```sh
matrix([[0.        , 0.41677501, 0.51640289, ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.54892995, 0.        , 0.        , ..., 0.54217851, 0.        ,
         0.        ],
        ...,
        [0.64131101, 0.        , 0.        , ..., 0.63342334, 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.6246757 , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ]])
```
Selanjutnya, mari kita lihat matriks tf-idf untuk beberapa film (movie_title) dan genre film (genres). Terapkan kode berikut.
```sh
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=vectorizer.get_feature_names_out(),
    index=data.movie_title
).sample(21, axis=1).sample(10, axis=0)
```
Output:
| movie_title                | crime | musical | western | imax | horror   | thriller | mystery  | action   | noir | documentary | drama    | war | adventure | comedy   | film | fi     | animation | sci     | fantasy | romance |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| Swelter (2014)            | 0.0    | 0.0     | 0.0     | 0.0  | 0.000000 | 0.633423 | 0.000000 | 0.641311 | 0.0  | 0.0         | 0.433007 | 0.0 | 0.000000  | 0.000000 | 0.0  | 0.0000 | 0.0       | 0.0000  | 0.0     | 0.0     |
| Ruby in Paradise (1993)   | 0.0    | 0.0     | 0.0     | 0.0  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0  | 0.0         | 1.000000 | 0.0 | 0.000000  | 0.000000 | 0.0  | 0.0000 | 0.0       | 0.0000  | 0.0     | 0.0     |
| Coffee Town (2013)        | 0.0    | 0.0     | 0.0     | 0.0  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0  | 0.0         | 0.000000 | 0.0 | 0.000000  | 1.000000 | 0.0  | 0.0000 | 0.0       | 0.0000  | 0.0     | 0.0     |
| Yes Men, The (2003)       | 0.0    | 0.0     | 0.0     | 0.0  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0  | 1.0         | 0.000000 | 0.0 | 0.000000  | 0.000000 | 0.0  | 0.0000 | 0.0       | 0.0000  | 0.0     | 0.0     |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ...| ... |

Output matriks tf-idf di atas menunjukkan film Swelter (2014) memiliki genre thriller, action, dan drama. Hal ini terlihat dari nilai matriks 0.633423 pada genre drama, 0.641311 pada action, dan 0.433007 pada drama. Selanjutnya, film Ruby in Paradise (1993) termasuk dalam genre drama yang ditunjukkan oleh nilai matriks 1.0 pada genre drama. Sedangkan, film Yes Men, The (2003) termasuk dalam genre documentary.
Demikian seterusnya, matriks tf-idf ini memudahkan kita dalam menganalisis seberapa kuat keterkaitan antara film dan masing-masing genre, serta memungkinkan kita mengelompokkan film berdasarkan kedekatan nilai tf-idf antar genre.

### Content Based Filtering - Cosine Similarity
Pada tahap sebelumnya, kita telah berhasil mengidentifikasi korelasi antara restoran dengan kategori masakannya. Sekarang, kita akan menghitung derajat kesamaan (similarity degree) antar restoran dengan teknik cosine similarity. Di sini, kita menggunakan fungsi cosine_similarity dari library sklearn. Jalankan kode berikut.
```sh
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
```
Output:
```sh
array([[1.        , 0.15245713, 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.15245713, 1.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 1.        , ..., 0.69546334, 0.33868574,
        0.        ],
       ...,
       [0.        , 0.        , 0.69546334, ..., 1.        , 0.39568417,
        0.        ],
       [0.        , 0.        , 0.33868574, ..., 0.39568417, 1.        ,
        0.78088428],
       [0.        , 0.        , 0.        , ..., 0.        , 0.78088428,
        1.        ]])
```
Pada tahapan ini, kita menghitung cosine similarity dataframe tfidf_matrix yang kita peroleh pada tahapan sebelumnya. Dengan satu baris kode untuk memanggil fungsi cosine similarity dari library sklearn, kita telah berhasil menghitung kesamaan (similarity) antar film. Kode di atas menghasilkan keluaran berupa matriks kesamaan dalam bentuk array.
Selanjutnya, mari kita lihat matriks kesamaan setiap film dengan menampilkan judul film dalam 5 sampel kolom (axis = 1) dan 10 sampel baris (axis=0). Jalankan kode berikut.
```sh
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['movie_title'], columns=data['movie_title'])
print('Shape:', cosine_sim_df.shape)
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
```
Output:
```sh
Shape: (9690, 9690)
```
| movie_title                        | It's a Very Merry Muppet Christmas Movie (2002) | 12 Chairs (1971) | Drive Me Crazy (1999) | Way of the Dragon, The (a.k.a. Return of the Dragon) (1972) | Heartless (2009) |
|-----------------------------------|--------------------------------------------------|-------------------|------------------------|-------------------------------------------------------------|------------------|
| Last Seduction, The (1994)        | 0.000000                                         | 0.000000          | 0.000000               | 0.526772                                                    | 0.233336         |
| Cradle Will Rock (1999)           | 0.000000                                         | 0.000000          | 0.000000               | 0.000000                                                    | 0.000000         |
| Return with Honor (1998)          | 0.000000                                         | 0.000000          | 0.000000               | 0.000000                                                    | 0.000000         |
| Platoon (1986)                    | 0.000000                                         | 0.000000          | 0.000000               | 0.000000                                                    | 0.000000         |
| Day of the Locust, The (1975)     | 0.000000                                         | 0.000000          | 0.000000               | 0.000000                                                    | 0.000000         |
| The Dark Tower (2017)             | 0.000000                                         | 0.000000          | 0.000000               | 0.000000                                                    | 0.413621         |
| Tigerland (2000)                  | 0.000000                                         | 0.000000          | 0.000000               | 0.000000                                                    | 0.000000         |
| Silence, The (Tystnaden) (1963)   | 0.000000                                         | 0.000000          | 0.000000               | 0.000000                                                    | 0.000000         |
| Opera (1987)                      | 0.000000                                         | 0.000000          | 0.000000               | 0.395372                                                    | 0.643067         |
| Harvey Girls, The (1946)          | 0.131031                                         | 0.151159          | 0.159679               | 0.000000                                                    | 0.000000         |

Dengan cosine similarity, kita berhasil mengidentifikasi kesamaan antara satu film dengan film lainnya. Shape (9690, 9690) merupakan ukuran matriks similarity dari data yang kita miliki. Berdasarkan data yang ada, matriks di atas sebenarnya berukuran 9690 film x 9690 film (masing-masing dalam sumbu X dan Y). Artinya, kita mengidentifikasi tingkat kesamaan pada 9690 nama film. Tapi tentu kita tidak bisa menampilkan semuanya. Oleh karena itu, kita hanya memilih 10 film pada baris vertikal dan 5 film pada sumbu horizontal seperti pada contoh di atas.
Nah, dengan data kesamaan (similarity) film yang diperoleh dari kode sebelumnya, kita akan merekomendasikan daftar film yang mirip (similar) dengan film yang sebelumnya pernah ditonton pengguna.

### Content Based Filtering - Mendapatkan Rekomendasi
Sebelumnya, kita telah memiliki data similarity (kesamaan) antar film. Kini, tibalah saatnya  menghasilkan sejumlah film yang akan direkomendasikan kepada user. Untuk lebih memahami bagaimana cara kerjanya, lihatlah kembali matriks similarity pada tahap sebelumnya. Sebagai gambaran, mari kita ambil satu contoh berikut.
User X pernah menonton film X. Kemudian, saat user tersebut berencana untuk menonton film lain, sistem akan merekomendasikan film Y. Nah, rekomendasi film ini berdasarkan kesamaan yang dihitung dengan cosine similarity pada tahap sebelumnya.

Kita akan membuat fungsi `movie_recommendations` untuk mendapatkan daftar rekomendasi film berdasarkan kemiripan konten. Fungsi ini memanfaatkan data cosine similarity antar judul film, lalu mencari film dengan tingkat kemiripan tertinggi terhadap film yang diberikan oleh pengguna.
Fungsi ini menerima beberapa parameter:
- **judul_film**: judul film yang dijadikan acuan rekomendasi.
- **similarity_data**: matriks dataframe yang berisi skor kemiripan antar film.
- **items**: dataframe yang berisi metadata film, seperti judul dan genre.
- **k**: jumlah film yang direkomendasikan.

Proses dalam fungsi meliputi pencarian `k` film dengan nilai kemiripan tertinggi (selain film itu sendiri), dan hasil akhirnya berupa dataframe yang memuat daftar film rekomendasi beserta informasinya.

Setelah itu, kita dapat menggunakan fungsi tersebut untuk mencari rekomendasi film yang mirip dengan judul film yang kita inginkan. Jalankan kode berikut untuk mendapatkan rekomendasi film yang mirip dengan film Toy Story (1995).
```sh
movie_recommendations('Toy Story (1995)')
```
Output:
| movie_title | movie_genres |
| ------ | ------ |
| Monsters, Inc. (2001) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| Moana (2016) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| Turbo (2013) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| Wild, The (2006) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| Toy Story 2 (1999) | Adventure\|Animation\|Children\|Comedy\|Fantasy |

Berhasil! sistem kita memberikan rekomendasi 5 judul film yang mirip dengan Toy Story (1995) dengan genre Adventure|Animation|Children|Comedy|Fantasy. Selanjutnya kita akan melakukan tahap modelling menggunakan Collaborative Filtering.

### Collaborative Filtering
Pahami terlebih dahulu data rating yang kita miliki. Ingatlah saat kita melakukan load data di awal dan membaca file rating_final.csv. Saat itu, kita membuat variabel rating dan menetapkan data pada variabel tersebut. Kemudian, kita menghapus rating pada data film dengan genres = '(no_genres_listed)' dan menyimpannya dalam variabel movies_ratings. Untuk memudahkan supaya tidak tertukar dengan fitur ‘rating’ pada data, kita ubah nama variabel movies_ratings (tanpa kolom genres dan title) menjadi variabel df.
```sh
df = movies_ratings.drop(['genres', 'title'], axis=1)
```

### Collaborative Filtering - Data Preparation
Sekarang kita memasuki tahap preparation. Pada tahap ini, kita perlu melakukan persiapan data untuk menyandikan (encode) fitur userId dan movieId ke dalam indeks integer. Meskipun userId dan movieId sudah berupa angka, nilai-nilai tersebut merupakan kategori, bukan angka dalam arti matematis. Misalnya, userId 10 dan 3 tidak memiliki hubungan kuantitatif yang bermakna bagi model. Oleh karena itu, kita melakukan encoding ulang agar setiap ID dipetakan ke indeks integer yang berurutan dan lebih efisien untuk diproses oleh model pembelajaran mesin. Terapkan kode berikut:
```sh
user_ids = df['userId'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
```
Selanjutnya, lakukan hal yang sama pada fitur ‘movieId’.
```sh
movie_ids = df['movieId'].unique().tolist()
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
```
Berikutnya, petakan userId dan movieId ke dataframe yang berkaitan.
```sh
df['user'] = df['userId'].map(user_to_user_encoded)
df['movie'] = df['movieId'].map(movie_to_movie_encoded)
```
Tahap persiapan telah selesai. Berikut adalah hal-hal yang telah kita lakukan pada tahap ini:
- Memahami data rating yang kita miliki.
- Menyandikan (encode) fitur ‘userId’ dan ‘movieId’ ke dalam indeks integer.
- Memetakan ‘userID’ dan ‘movieId’ ke dataframe yang berkaitan.
- Mengecek beberapa hal dalam data seperti jumlah user, jumlah film, kemudian mengubah nilai rating menjadi float.

Tahap persiapan ini penting dilakukan agar data siap digunakan untuk
pemodelan. Namun sebelumnya, kita perlu membagi data untuk training dan validasi terlebih dahulu.

### Collaborative Filtering - Membagi Data untuk Training dan Validasi
Sebelum membagi data menjadi data training dan validasi, datanya perlu diacak terlebih dahulu agar distribusinya menjadi acak dan tidak terurut, sehingga tidak terjadi bias selama pelatihan model.
```sh
df = df.sample(frac=1, random_state=42)
```
- frac=1 berarti "fraction" = 100% dari data diambil, tapi urutannya diacak.
- random_state=42 berarti menetapkan seed untuk pengacakan, agar hasil acakan selalu sama setiap kali dijalankan. Angka 42 bisa diganti berapa pun, tapi disarankan untuk tetap konsisten agar replikasi bisa dilakukan.

Selanjutnya, kita bagi data train dan validasi dengan komposisi 80:20. Namun sebelumnya, kita perlu memetakan (mapping) data user dan film menjadi satu value terlebih dahulu. Lalu, buatlah rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training.
```sh
x = df[['user', 'movie']].values
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:])
print(x, y)
```
Kode tersebut digunakan untuk mempersiapkan data sebelum dimasukkan ke dalam model dengan cara memisahkan fitur dan label, melakukan normalisasi rating, serta membagi data menjadi data latih dan validasi. Pertama, `x = df[['user', 'movie']].values` mengambil kolom hasil encoding user dan movie sebagai fitur input dan mengubahnya menjadi array NumPy. Kemudian, `y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values` menormalisasi nilai rating agar berada dalam rentang 0 hingga 1, yang mempermudah proses pelatihan model. Selanjutnya, `train_indices = int(0.8 * df.shape[0])` menghitung jumlah baris yang digunakan untuk data latih berdasarkan 80% dari total data. Terakhir, data dibagi menjadi `x_train, x_val, y_train, dan y_val` menggunakan slicing berdasarkan indeks tersebut, dan `print(x, y)` digunakan untuk menampilkan isi array input dan label yang telah diproses.

Output:
```sh
[[   6  783]
 [ 413 5917]
 [ 508  875]
 ...
 [ 479 5869]
 [   5  712]
 [ 102 1129]] [0.66666667 0.66666667 0.66666667 ... 0.11111111 0.55555556 0.88888889]
 ```
 Data telah siap untuk dimasukkan ke dalam model untuk dievaluasi.

## Evaluation
Pada tahap ini, model menghitung skor kecocokan antara user dan film dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan film. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan film. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan film. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.
Pada tahap ini, model menghitung skor kecocokan antara user dan film menggunakan teknik embedding, yang memungkinkan model merepresentasikan user dan film dalam bentuk vektor berdimensi rendah. Proses ini dilakukan oleh kelas RecommenderNet, yang merupakan turunan dari tf.keras.Model. Berikut penjelasan dari tahapannya:
1. Embedding Layer
Model membuat dua embedding layer terpisah untuk:
    - User (self.user_embedding)
    Menerjemahkan ID user menjadi vektor berdimensi embedding_size.
    - Film (self.movie_embedding)
    Menerjemahkan ID film menjadi vektor berdimensi embedding_size.

    Selain itu, model juga membuat bias embedding untuk masing-masing:
    - self.user_bias: memberikan nilai bias untuk tiap user.
    - self.movie_bias: memberikan nilai bias untuk tiap film.

2. Dot Product (Perkalian Vektor)
Pada fungsi call(), model menerima input berisi pasangan [user_id, movie_id].
Kemudian, dilakukan:
- Pengambilan embedding user dan film berdasarkan ID masing-masing.
- Pengambilan bias user dan bias film.
- Penghitungan dot product antara embedding user dan embedding film, yaitu tf.tensordot(user_vector, movie_vector, 2), yang menghasilkan skor kecocokan mentah.

3. Penambahan Bias
Hasil dot product tersebut ditambahkan dengan user_bias dan movie_bias untuk mengakomodasi preferensi atau kecenderungan masing-masing user dan film yang tidak ditangkap oleh embedding saja.

4. Aktivasi Sigmoid
Nilai akhir (skor kecocokan) dilewatkan ke fungsi aktivasi sigmoid, yang akan memetakan skor tersebut ke dalam rentang [0, 1].
Nilai ini dapat diinterpretasikan sebagai probabilitas seberapa besar kemungkinan user akan menyukai film tersebut.

Selanjutnya, lakukan proses compile terhadap model.
```sh
model = RecommenderNet(num_users, num_movie, 50)
model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.RootMeanSquaredError()])
```
Model ini merupakan model rekomendasi sederhana yang dibangun dengan arsitektur RecommenderNet. Model ini dirancang untuk memprediksi tingkat kecocokan antara pengguna dan film berdasarkan embedding. Jumlah pengguna (num_users) dan film (num_movie) ditentukan berdasarkan dataset, dan 50 merupakan ukuran dari embedding vector yang digunakan untuk mewakili setiap pengguna dan film dalam ruang fitur laten.
Selanjutnya, model di-compile dengan beberapa parameter, yaitu:
- Loss function BinaryCrossentropy: Digunakan untuk mengukur seberapa jauh prediksi model dari nilai target sebenarnya, dalam konteks klasifikasi biner (antara user dan film).
- Optimizer Adam dengan LR = 0.001: Digunakan untuk mengupdate bobot model selama training untuk meminimalkan loss. Nilai 0.001 merupakan nilai standar yang memberikan laju pembelajaran yang stabil, tidak terlalu cepat sehingga menghindari overshooting, dan tidak terlalu lambat agar proses training efisien. Adam adalah salah satu optimizer yang paling umum karena menggabungkan kelebihan dari dua metode lainnya: AdaGrad dan RMSProp. Adam dapat menangani data yang sparse dan memiliki performa konvergensi yang baik dengan sedikit penyesuaian hyperparameter.
- Metrik evaluasi RMSE: Digunakan untuk mengevaluasi performa model selama training dan testing dengan menghitung akar dari rata-rata kuadrat selisih antara prediksi dan nilai aktual. Semakin kecil nilai RMSE, maka semakin akurat model dalam memprediksi.

RMSE dirumuskan sebagai berikut:

![Formula](images/Formula.png)

Di mana:
- 𝑛: jumlah data (jumlah pasangan prediksi-aktual)
- ŷ𝑖: nilai yang diprediksi oleh model pada data ke-i
- 𝑦𝑖: nilai aktual (ground truth) pada data ke-i
- (ŷ𝑖 - 𝑦𝑖)^2 : kuadrat dari selisih antara prediksi dan aktual

RMSE adalah akar dari rata-rata kuadrat selisih antara prediksi dan nilai aktual. Cocok digunakan untuk mengukur performa sistem rekomendasi yang memprediksi angka seperti rating. Nilai RMSE yang rendah menunjukkan model yang akurat.

Selanjutnya, kita akan melatih model dengan data training dan data validation yang telah ditetapkan sebelumnya dengan 25 epoch dan nilai batch size = 32. Penggunaan batch_size = 32 berarti model akan memproses 32 data sekaligus sebelum memperbarui bobotnya. Ukuran ini memberikan keseimbangan antara kecepatan training, efisiensi memori, dan akurasi pembelajaran, serta merupakan salah satu pilihan batch size yang paling umum digunakan dalam praktik deep learning. Penggunaan epoch = 25 berarti model akan dilatih selama 25 putaran penuh. Jumlah ini cukup untuk memberikan model waktu belajar yang memadai tanpa terlalu lama sehingga berisiko overfitting.
```sh
history = model.fit(x = x_train, y = y_train, batch_size = 32, epochs = 25, validation_data = (x_val, y_val))
```
Berikut adalah visualisasi grafik RMSE selama proses training dan validation:
![RMSE](images/RMSE.png)

Sumbu x menunjukkan epoch dan sumbu y menunjukkan nilai RMSE. Perhatikanlah, proses training dan validation model cukup smooth dan model konvergen pada epochs sekitar 25. Dari proses ini, kita memperoleh nilai error akhir sebesar 0.1875 dan error pada data validasi sebesar 0.2007. Nilai tersebut cukup bagus untuk sistem rekomendasi. Mari kita cek, apakah model ini bisa membuat rekomendasi dengan baik?

### Mendapatkan Rekomendasi Film
Untuk mendapatkan rekomendasi film, pertama kita ambil sampel user secara acak dan definisikan variabel movie_not_watched yang merupakan daftar film yang belum pernah ditonton oleh user. Daftar movie_not_watched inilah yang akan menjadi film yang kita rekomendasikan.

Sebelumnya, user telah memberi rating pada beberapa film yang telah mereka tonton. Kita menggunakan rating ini untuk membuat rekomendasi film yang mungkin cocok untuk user. Nah, film yang akan direkomendasikan tentulah film yang belum pernah ditonton oleh user. Oleh karena itu, kita perlu membuat variabel movie_not_watched sebagai daftar film untuk direkomendasikan pada user. Variabel movie_not_watched diperoleh dengan menggunakan operator bitwise (~) pada variabel movie_watched_by_user.
Terapkan kode berikut.
```sh
movie_df = movie_new
df = movies_ratings.drop(['genres', 'title'], axis=1)
user_id = df.userId.sample(1).iloc[0]
movie_watched_by_user = df[df.userId == user_id]
# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
movie_not_watched = movie_df[~movie_df['id'].isin(movie_watched_by_user.movieId.values)]['id']
movie_not_watched = list(
    set(movie_not_watched)
    .intersection(set(movie_to_movie_encoded.keys()))
)
movie_not_watched = [[movie_to_movie_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)
```
Kode tersebut bertujuan untuk menyiapkan data rekomendasi film bagi seorang pengguna secara acak. Pertama, dipilih satu userId secara acak dari dataset, lalu diambil daftar film yang sudah ditonton oleh user tersebut. Selanjutnya, digunakan operator bitwise NOT (~) untuk mendapatkan daftar film yang belum ditonton oleh user dengan membandingkan semua film yang ada di dataset. Daftar film ini kemudian disaring agar hanya mencakup film yang telah diencoding (ada di movie_to_movie_encoded), lalu setiap ID film dikonversi ke bentuk numerik. Setelah itu, ID user juga dikonversi ke bentuk numerik, dan dibentuk array pasangan user-film dalam bentuk numerik, yang siap digunakan sebagai input bagi model rekomendasi untuk memprediksi skor kecocokan antara user dan film.

Selanjutnya, untuk memperoleh rekomendasi restoran, gunakan fungsi model.predict() dari library Keras dengan menerapkan kode berikut.
```sh
ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_watched[x][0]) for x in top_ratings_indices
]
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Movies with high ratings from user')
print('----' * 8)
top_movie_user = (
    movie_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df['id'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.movie_title, ':', row.movie_genres)
print('----' * 8)
print('Top 10 Movies Recommendation')
print('----' * 8)
recommended_movie = movie_df[movie_df['id'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.movie_title, ':', row.movie_genres)
```
Output:
```sh
298/298 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
Showing recommendations for users: 279
===========================
Movies with high ratings from user
--------------------------------
Star Wars: Episode IV - A New Hope (1977) : Action|Adventure|Sci-Fi
Pulp Fiction (1994) : Comedy|Crime|Drama|Thriller
Alien (1979) : Horror|Sci-Fi
Thing, The (1982) : Action|Horror|Sci-Fi|Thriller
Blade Runner (1982) : Action|Sci-Fi|Thriller
--------------------------------
Top 10 Movies Recommendation
--------------------------------
Inside Job (2010) : Documentary
Ran (1985) : Drama|War
Gallipoli (1981) : Drama|War
Lawrence of Arabia (1962) : Adventure|Drama|War
Streetcar Named Desire, A (1951) : Drama
Band of Brothers (2001) : Action|Drama|War
Trial, The (Procès, Le) (1962) : Drama
Captain Fantastic (2016) : Drama
Paths of Glory (1957) : Drama|War
Yi Yi (2000) : Drama
```
Kita telah berhasil memberikan rekomendasi kepada user, dan sebagai contoh, hasil di atas adalah rekomendasi untuk user dengan userId 279. Dari output tersebut, kita dapat melihat dua bagian utama: "Movies with high ratings from user", yang menunjukkan daftar film yang sebelumnya diberi rating tinggi oleh user tersebut, dan "Top 10 Movies Recommendation", yang merupakan hasil rekomendasi berdasarkan preferensi user. Menariknya, banyak film yang direkomendasikan memiliki genre yang konsisten dengan preferensi genre user, seperti Drama, War, dan Documentary yang mendominasi daftar rekomendasi. Hal ini menunjukkan bahwa sistem rekomendasi berhasil mengenali pola preferensi pengguna dan memberikan saran yang relevan.

Sampai tahap ini, kita telah berhasil membangun sistem rekomendasi menggunakan teknik Collaborative Filtering, yang mengandalkan data interaksi pengguna (seperti rating) tanpa memerlukan informasi konten film secara langsung. Ingatlah bahwa setiap teknik memiliki pendekatan berbeda — misalnya, Collaborative Filtering memerlukan data rating dari banyak pengguna, sementara Content Based Filtering bekerja hanya dengan data atribut item (seperti genre film) dan preferensi user target tanpa bergantung pada pengguna lain.

Untuk menjawab Problem Statements dan Goals pada proses Business Understanding, kita akan mengevaluasi masalah tersebut secara bertahap.
### Problem Statements - Jawaban
1. Bagaimana cara merekomendasikan film berdasarkan preferensi konten yang mirip dengan film yang disukai pengguna?
- Jawaban: Dengan menggunakan pendekatan Content-Based Filtering, sistem menganalisis atribut konten film seperti genre, dan mencocokkannya dengan film-film yang pernah diberi rating tinggi oleh pengguna. Sistem kemudian merekomendasikan film lain yang memiliki kemiripan konten, meskipun belum pernah dirating oleh pengguna lain.
2. Bagaimana cara memberikan rekomendasi personal kepada pengguna berdasarkan pola rating pengguna lain yang memiliki selera serupa?
- Jawaban: Dengan menggunakan teknik Collaborative Filtering, sistem mengidentifikasi pengguna lain yang memiliki pola rating mirip dengan user target. Berdasarkan kesamaan ini, sistem dapat memprediksi film yang kemungkinan besar akan disukai oleh user, meskipun user belum pernah menonton atau memberi rating film tersebut.
3. Bagaimana membandingkan efektivitas pendekatan Content-Based Filtering dan Collaborative Filtering dalam memberikan rekomendasi yang relevan?
- Jawaban: Efektivitas kedua pendekatan dibandingkan melalui evaluasi metrik kinerja, seperti Root Mean Squared Error (RMSE). Dalam percobaan ini, model Collaborative Filtering mencapai nilai RMSE training sebesar 0.1875 dan RMSE validation sebesar 0.2007, yang menunjukkan model ini mampu memberikan prediksi rating yang cukup akurat. Selain itu, relevansi rekomendasi pada Content Based Filtering dapat dilihat dari kesesuaian genre film yang dicari dengan genre film yang ditampilkan pada tahap Content Based Filtering - Mendapatkan Rekomendasi. Pada tahap tersebut, genre film yang dicari adalah genre film yang dimiliki oleh film Toy Story (1995), lalu sistem menampilkan 5 film dengan genre yang sama.

### Goals - Jawaban
1. Membangun sistem rekomendasi menggunakan Content-Based Filtering dengan memanfaatkan informasi genre dari film yang pernah ditonton dan disukai pengguna.
- Jawaban: Sistem telah berhasil dikembangkan untuk menganalisis genre film yang disukai user, dan menghasilkan rekomendasi berdasarkan kemiripan konten. Hal ini memungkinkan sistem memberikan rekomendasi yang bersifat personal meskipun tanpa mempertimbangkan data pengguna lain.
2. Membangun sistem rekomendasi menggunakan Collaborative Filtering berbasis matrix factorization yang memanfaatkan pola rating pengguna lain.
- Jawaban: Sistem rekomendasi dengan pendekatan Collaborative Filtering berhasil dibangun menggunakan teknik embedding dan matrix factorization. Model ini dilatih dengan 25 epoch dan batch size 32, serta dievaluasi menggunakan metrik RMSE, menghasilkan performa yang cukup baik (RMSE validasi 0.2007).
3. Membandingkan performa dari kedua pendekatan untuk mengetahui mana yang lebih baik dalam konteks MovieLens dataset.
- Jawaban: Berdasarkan hasil evaluasi, pendekatan Collaborative Filtering memberikan hasil prediksi rating yang lebih akurat (dengan nilai RMSE yang rendah). Namun demikian, pendekatan Content-Based Filtering tetap bermanfaat terutama dalam kondisi cold-start (misalnya pengguna baru), karena tidak bergantung pada rating dari pengguna lain.

**---Ini adalah bagian akhir laporan---**
