# Integrasi Segmentasi Kerusakan Jalan Menggunakan U-Net untuk Penentuan Rute

Repositori ini berisi seluruh kode sumber yang digunakan dalam pelaksanaan Tugas Akhir Program Sarjana Teknologi Informasi. Fokus penelitian ini adalah penerapan **segmentasi citra jalan menggunakan model U-Net** untuk mendeteksi kerusakan jalan, serta pemanfaatan hasil segmentasi tersebut dalam pembentukan **peta bobot (weight map)** yang dapat digunakan pada proses penentuan rute berbasis graf.

Kode disusun sebagai implementasi langsung dari metodologi penelitian dan digunakan untuk mendukung proses eksperimen, evaluasi, serta dokumentasi ilmiah.

## Abstrak

Penentuan rute pada sistem navigasi umumnya mempertimbangkan jarak dan waktu tempuh, namun belum memasukkan kondisi fisik permukaan jalan sebagai variabel utama. Kondisi jalan yang rusak dapat memengaruhi kenyamanan dan keselamatan pengguna, sehingga diperlukan pendekatan tambahan untuk merepresentasikan kondisi tersebut secara komputasional.

Penelitian ini mengimplementasikan segmentasi citra jalan menggunakan model U-Net untuk mengidentifikasi beberapa jenis kerusakan jalan, yaitu lubang jalan, retakan, dan rutting. Hasil segmentasi diolah menjadi peta penalti dan peta bobot yang merepresentasikan tingkat kerusakan jalan. Peta bobot ini dirancang agar dapat digunakan sebagai biaya tambahan pada algoritma penentuan rute, sehingga rute yang dihasilkan dapat mempertimbangkan kondisi permukaan jalan.


## Tujuan Penelitian

Tujuan dari pengembangan sistem dalam repositori ini adalah:

1. Mengimplementasikan model U-Net untuk melakukan segmentasi kerusakan jalan berbasis citra.
2. Mengidentifikasi beberapa jenis kerusakan jalan dalam satu proses segmentasi.
3. Mengonversi hasil segmentasi menjadi peta penalti dan peta bobot.
4. Menyediakan keluaran sistem yang dapat digunakan sebagai masukan pada proses penentuan rute berbasis graf.


## Gambaran Umum Metodologi

Alur pemrosesan sistem yang diimplementasikan dalam repositori ini terdiri dari tahapan berikut:

1. **Akuisisi Citra Jalan**  
   Citra jalan diperoleh dari dataset yang telah disiapkan dan digunakan sebagai data masukan sistem.

2. **Pra-pemrosesan Citra**  
   - Perubahan ukuran citra menjadi 256 × 256 piksel  
   - Normalisasi nilai piksel

3. **Segmentasi Menggunakan Model U-Net**  
   Model U-Net menghasilkan peta segmentasi multi-kelas dengan empat kelas, yaitu:
   - Latar belakang
   - Lubang jalan
   - Retakan jalan
   - Rutting

4. **Pasca-pemrosesan Segmentasi**  
   - Penerapan nilai ambang untuk mengurangi noise  
   - Operasi dilasi morfologi  
   - Penyaringan Gaussian untuk penghalusan hasil

5. **Pembentukan Peta Penalti**  
   Setiap jenis kerusakan diberikan nilai penalti berbeda sesuai tingkat keparahan kerusakan.

6. **Pembentukan Peta Bobot (Weight Map)**  
   Peta bobot dibentuk dengan menambahkan peta penalti ke biaya dasar, sehingga area dengan tingkat kerusakan tinggi memiliki nilai biaya lebih besar.

7. **Visualisasi dan Penyimpanan Metadata**  
   Sistem menghasilkan visualisasi hasil segmentasi dan peta bobot, serta menyimpan metadata dalam format JSON.
   

## Parameter Konfigurasi

Beberapa parameter utama yang digunakan dalam sistem ini meliputi:

| Parameter | Keterangan |
|---------|------------|
| `IMG_SIZE` | Ukuran citra masukan |
| `PENALTIES` | Nilai penalti setiap jenis kerusakan |
| `BASE_COST` | Biaya dasar peta bobot |
| `DILATE_ITERS` | Jumlah iterasi dilasi morfologi |
| `GAUSSIAN_SIGMA` | Parameter penghalusan Gaussian |
| `DETECTION_THRESHOLD` | Ambang batas tingkat kerusakan |

Perbedaan nilai parameter digunakan untuk keperluan eksperimen dan analisis.

## Keluaran Sistem

Untuk setiap citra masukan, sistem menghasilkan keluaran berupa:

1. Citra jalan asli  
2. Hasil segmentasi citra dalam representasi warna  
3. Peta bobot awal  
4. Peta bobot akhir berbasis penalti  

Selain itu, sistem menghasilkan file metadata yang memuat:
* Persentase area kerusakan
* Jumlah piksel untuk setiap jenis kerusakan
* Waktu pemrosesan
* Lokasi penyimpanan hasil


## Cara Menjalankan Program

1. Instal dependensi yang dibutuhkan:
   ```bash
   pip install numpy opencv-python tensorflow matplotlib scipy tqdm

2. Sesuai Path
   
- MODEL_PATH = "direktori_model"
- IMAGE_DIR = "direktori_citra"
- SAVE_DIR  = "direktori_output"

3. Jalankan Program
   ```bash
python main.py


## Variasi Eksperimen

Repositori ini mencakup beberapa konfigurasi eksperimen yang digunakan dalam proses pengujian sistem. Variasi eksperimen dilakukan untuk mengamati pengaruh perubahan konfigurasi terhadap hasil segmentasi dan pembentukan peta bobot.

Variasi eksperimen meliputi perbedaan model U-Net dan DeepLabV3+ terlatih, perbedaan konfigurasi fungsi kerugian, serta perbedaan nilai ambang tingkat kerusakan. Seluruh variasi eksperimen dijalankan menggunakan alur pemrosesan yang sama, sehingga perbedaan hasil yang diperoleh dapat dikaitkan dengan perubahan konfigurasi yang diterapkan.


## Catatan Akademik

Repositori ini dikembangkan sebagai bagian dari pelaksanaan Tugas Akhir Program Sarjana Teknologi Informasi. Seluruh kode sumber dan eksperimen yang disertakan digunakan untuk mendukung proses penelitian, evaluasi, serta penulisan laporan ilmiah sesuai dengan metodologi yang telah ditetapkan.


## Penulis

Zaki Waliyan Isnanto  
