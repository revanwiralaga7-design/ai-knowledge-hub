# AI Knowledge Hub

MVP chatbot publik dengan panel admin dataset. Modelnya adalah **AI basic dari nol** berbasis TF-IDF + cosine similarity yang dibangun sendiri dari dataset. Ini bukan LLM generatif: jawaban berasal dari entry paling relevan dalam data Anda. Desain ini sengaja ringan dan dapat berjalan pada VPS 1 GB RAM.

## Fitur
- Chat publik multi-bahasa
- Login admin dan pengelolaan dataset
- Import dataset `CSV`, `JSON`, `TXT`, dan `MD`
- Import langsung dari link/dataset ID Hugging Face (maksimum 5.000 baris per import agar aman untuk VPS kecil)
- Tombol **Latih ulang AI basic** untuk membangun indeks kata dari nol
- Mencegah jawaban asal: jika relevansi rendah, bot mengaku belum tahu
- Ekspor pasangan Q&A ke JSONL untuk eksperimen/fine-tuning di Google Colab
- Data lokal tersimpan di `data/knowledge.json`, tanpa database eksternal

## Menjalankan

```bash
cp .env.example .env
nano .env # buat ADMIN_PASSWORD dan SESSION_SECRET yang kuat
npm install
npm start
```

Buka `http://IP-VPS-ANDA:3000`. Untuk production, letakkan di balik Nginx/Caddy HTTPS dan ubah opsi cookie `secure:true` pada `server.js`.

## Format dataset

### CSV

```csv
question,answer
"Apa jam operasional?","Setiap hari pukul 09.00 hingga 21.00 WIB."
"How do I contact support?","Contact support@example.com."
```

Kolom alternatif yang diterima: `pertanyaan`/`jawaban`, `prompt`/`response`, atau `input`/`output`.

### JSON

```json
[
  {"question":"Apa kebijakan refund?","answer":"Refund dapat diajukan maksimal tujuh hari."},
  {"question":"Where are you located?","answer":"We operate online."}
]
```

### TXT atau MD

Pisahkan setiap topik dengan satu baris kosong. Sistem menjadikan setiap paragraf sebagai knowledge entry.

## AI dari nol vs fine-tuning

Dataset Q&A langsung dilatih menjadi indeks pencarian relevansi oleh website. Ini cocok untuk AI basic yang faktual dan ringan.

Untuk model generatif dari nol, dataset dan compute yang dibutuhkan jauh lebih besar. Google Colab dapat digunakan untuk eksperimen model kecil/fine-tune, tetapi bukan untuk membuat LLM berkualitas dari nol. Dari dashboard, download **Export JSONL untuk fine-tuning Colab**, lalu gunakan file itu di notebook Anda.

## Catatan keamanan
- Jangan gunakan password admin default.
- `.env` tidak boleh diunggah ke GitHub.
- Endpoint admin dibuat untuk MVP. Tambahkan HTTPS, rate limit, akun admin/database, backup, dan validasi file yang lebih ketat sebelum dibuka ke internet.
