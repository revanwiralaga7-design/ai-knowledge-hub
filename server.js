import express from 'express';
import multer from 'multer';
import cookieParser from 'cookie-parser';
import dotenv from 'dotenv';
import fs from 'fs/promises';
import { createWriteStream } from 'fs';
import { pipeline } from 'stream/promises';
import { Readable } from 'stream';
import { execFile } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { fileURLToPath } from 'url';
import { parse as csvParse } from 'csv-parse/sync';
import crypto from 'crypto';

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, 'data');
const DB_FILE = path.join(DATA_DIR, 'knowledge.json');
const INDEX_FILE = path.join(DATA_DIR, 'index.json');
const PORT = Number(process.env.PORT || 3000);
const app = express();
const execFileAsync = promisify(execFile);
const upload = multer({ dest: path.join(DATA_DIR, 'uploads'), limits: { fileSize: 25 * 1024 * 1024 } });
app.use(express.json({ limit: '2mb' }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

const stopWords = new Set('yang dan di ke dari untuk dengan atau pada adalah ini itu saya kamu kami anda sebagai dalam agar juga tidak bisa akan sudah lebih oleh karena tentang the a an is are to of in'.split(' '));
const tokenize = (text = '') => (text.toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter(w => w.length > 1 && !stopWords.has(w)));
const id = () => crypto.randomUUID();
async function read(file, fallback) { try { return JSON.parse(await fs.readFile(file, 'utf8')); } catch { return fallback; } }
async function write(file, value) { await fs.writeFile(file, JSON.stringify(value, null, 2)); }
async function db() { return read(DB_FILE, { datasets: [], entries: [], training: { status: 'Belum dilatih', updatedAt: null } }); }
function isAdmin(req) { return req.cookies.ai_admin === process.env.SESSION_SECRET && Boolean(process.env.SESSION_SECRET); }
function requireAdmin(req, res, next) { if (!isAdmin(req)) return res.status(401).json({ error: 'Login admin diperlukan.' }); next(); }

function makeIndex(entries) {
  const docs = entries.map(e => tokenize(`${e.question || ''} ${e.answer || ''} ${e.content || ''}`));
  const df = {};
  docs.forEach(words => new Set(words).forEach(w => { df[w] = (df[w] || 0) + 1; }));
  const n = Math.max(docs.length, 1);
  const vectors = docs.map(words => {
    const tf = {}; words.forEach(w => tf[w] = (tf[w] || 0) + 1);
    const v = {}; let norm = 0;
    Object.entries(tf).forEach(([w,c]) => { const x = (c / words.length) * Math.log((n + 1) / ((df[w] || 0) + 1) + 1); v[w] = x; norm += x*x; });
    return { v, norm: Math.sqrt(norm) || 1 };
  });
  return { df, n, vectors, generatedAt: new Date().toISOString() };
}
function search(query, data, index) {
  const words = tokenize(query); if (!words.length || !index.vectors?.length) return null;
  const tf = {}; words.forEach(w => tf[w] = (tf[w] || 0) + 1);
  const q = {}; let qn = 0;
  Object.entries(tf).forEach(([w,c]) => { const x = (c / words.length) * Math.log((index.n + 1) / ((index.df[w] || 0) + 1) + 1); q[w] = x; qn += x*x; });
  qn = Math.sqrt(qn) || 1;
  let best = null;
  index.vectors.forEach((item, i) => { let dot = 0; Object.entries(q).forEach(([w,x]) => dot += x * (item.v[w] || 0)); const score = dot / (qn * item.norm); if (!best || score > best.score) best = { score, entry: data.entries[i] }; });
  return best?.score > 0.09 ? best : null;
}
function toEntries(name, text, ext) {
  if (ext === '.json') {
    const parsed = JSON.parse(text); const rows = Array.isArray(parsed) ? parsed : (parsed.data || parsed.questions || []);
    return rows.map(r => ({ id:id(), source:name, type:'qa', question:r.question || r.pertanyaan || r.prompt || r.instruction || r.input || '', answer:r.answer || r.jawaban || r.response || r.completion || r.output || '', content:'' })).filter(r => r.question && r.answer);
  }
  if (ext === '.csv') {
    const rows = csvParse(text, { columns:true, skip_empty_lines:true, relax_column_count:true });
    return rows.map(r => ({ id:id(), source:name, type:'qa', question:r.question || r.pertanyaan || r.prompt || r.input || '', answer:r.answer || r.jawaban || r.response || r.completion || r.output || '', content:'' })).filter(r => r.question && r.answer);
  }
  const chunks = text.replace(/\r/g,'').split(/\n\s*\n/).map(x => x.trim()).filter(Boolean);
  return chunks.map(content => ({ id:id(), source:name, type:'document', question:'', answer:'', content:content.slice(0, 5000) }));
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
async function fetchHfJson(url, label) {
  // Dataset Viewer membatasi burst request. Retry agar import panjang tidak mati di tengah.
  let lastStatus = '';
  for (let attempt = 0; attempt < 8; attempt++) {
    try {
      const response = await fetch(url, { headers: { 'User-Agent': 'AI-Knowledge-Hub/1.0', 'Accept': 'application/json' } });
      if (response.ok) return response.json();
      lastStatus = `HTTP ${response.status}`;
      const retryable = [429, 500, 502, 503, 504].includes(response.status);
      if (!retryable) {
        const body = await response.text().catch(() => '');
        throw new Error(`${label} ditolak oleh Hugging Face (${lastStatus}). ${body.slice(0, 160)}`);
      }
      const retryAfter = Number(response.headers.get('retry-after'));
      await sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : Math.min(15000, 700 * 2 ** attempt));
    } catch (error) {
      if (attempt === 7) throw new Error(`${label} gagal setelah 8 percobaan (${lastStatus || error.message}).`);
      if (lastStatus) continue;
      await sleep(Math.min(15000, 700 * 2 ** attempt));
    }
  }
  throw new Error(`${label} gagal (${lastStatus}).`);
}
function hfDatasetId(value = '') {
  const clean = value.trim().replace(/\/$/, '');
  const match = clean.match(/huggingface\.co\/datasets\/([^/?#]+\/[^/?#]+)/i);
  const candidate = match ? match[1] : clean.replace(/^datasets\//, '');
  if (!/^[\w.-]+\/[\w.-]+$/.test(candidate)) throw new Error('Masukkan link atau ID dataset Hugging Face yang valid, misalnya HuggingFaceH4/CodeAlpaca_20K.');
  return candidate;
}
async function importHuggingFace(datasetInput) {
  const dataset = hfDatasetId(datasetInput);
  const splitData = await fetchHfJson(
    `https://datasets-server.huggingface.co/splits?dataset=${encodeURIComponent(dataset)}`,
    'Metadata dataset'
  );
  const target = splitData.splits?.find(s => s.split === 'train') || splitData.splits?.[0];
  if (!target) throw new Error('Hugging Face tidak menyediakan split yang dapat diimpor untuk dataset ini.');
  const rows = [];
  // Ambil seluruh split train. Pagination berhenti hanya saat Hugging Face
  // tidak lagi mengembalikan baris data.
  for (let offset = 0; ; offset += 100) {
    const url = new URL('https://datasets-server.huggingface.co/rows');
    url.searchParams.set('dataset', dataset);
    url.searchParams.set('config', target.config);
    url.searchParams.set('split', target.split);
    url.searchParams.set('offset', String(offset));
    url.searchParams.set('length', '100');
    const page = await fetchHfJson(url, `Pengambilan data pada baris ${offset}`);
    const pageRows = (page.rows || []).map(item => item.row || item);
    rows.push(...pageRows);
    if (pageRows.length < 100) break;
    // Sedikit jeda agar Hugging Face tidak memberi rate limit pada import panjang.
    await sleep(90);
  }
  const entries = toEntries(`HF:${dataset}`, JSON.stringify(rows), '.json');
  if (!entries.length) throw new Error('Dataset berhasil diambil, tetapi format kolomnya bukan Q&A yang dikenali. Dataset harus memiliki question/answer, prompt/completion, atau input/output.');
  return { dataset, entries, downloaded: rows.length, config: target.config, split: target.split };
}

async function downloadParquet(url, destination) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const response = await fetch(url, { headers: { 'User-Agent': 'AI-Knowledge-Hub/1.0' } });
    if (response.ok && response.body) {
      await pipeline(Readable.fromWeb(response.body), createWriteStream(destination));
      return;
    }
    if (![429, 500, 502, 503, 504].includes(response.status)) throw new Error(`File dataset tidak bisa diunduh (HTTP ${response.status}).`);
    await sleep(Math.min(60000, 1500 * 2 ** attempt));
  }
  throw new Error('File dataset tetap terkena rate limit Hugging Face setelah beberapa percobaan.');
}
async function importHuggingFaceParquet(datasetInput) {
  const dataset = hfDatasetId(datasetInput);
  // Mengambil daftar file asli langsung dari repository, tidak melalui Dataset Viewer
  // yang memberi HTTP 429 saat membaca ribuan halaman.
  const tree = await fetchHfJson(`https://huggingface.co/api/datasets/${dataset}/tree/main?recursive=true&expand=false`, 'Daftar file dataset');
  const parquetFiles = tree.filter(file => file.type === 'file' && file.path.endsWith('.parquet'));
  if (!parquetFiles.length) return importHuggingFace(dataset);
  const jobDir = path.join(DATA_DIR, 'uploads', `hf-${Date.now()}`);
  await fs.mkdir(jobDir, { recursive: true });
  try {
    const localFiles = [];
    for (let index = 0; index < parquetFiles.length; index++) {
      const file = parquetFiles[index];
      const local = path.join(jobDir, `${index}.parquet`);
      await downloadParquet(`https://huggingface.co/datasets/${dataset}/resolve/main/${file.path}?download=true`, local);
      localFiles.push(local);
    }
    try { await execFileAsync('python3', ['-c', 'import pyarrow'], { timeout: 30000 }); }
    catch { await execFileAsync('python3', ['-m', 'pip', 'install', '--user', 'pyarrow'], { timeout: 300000, maxBuffer: 1024 * 1024 }); }
    const resultFile = path.join(jobDir, 'rows.json');
    await execFileAsync('python3', [path.join(__dirname, 'hf_parquet_to_json.py'), resultFile, ...localFiles], { timeout: 1800000, maxBuffer: 1024 * 1024 });
    const raw = await fs.readFile(resultFile, 'utf8');
    const entries = toEntries(`HF:${dataset}`, raw, '.json');
    if (!entries.length) throw new Error('Dataset berhasil diunduh, tetapi tidak memiliki kolom Q&A yang dikenali.');
    return { dataset, entries, downloaded: JSON.parse(raw).length, config: 'repository-parquet', split: 'all parquet files' };
  } finally { await fs.rm(jobDir, { recursive: true, force: true }); }
}

app.get('/api/status', async (req,res) => { const data=await db(); res.json({ admin:isAdmin(req), datasets:data.datasets, entries:data.entries.length, training:data.training }); });
app.post('/api/login', (req,res) => { if (!process.env.ADMIN_PASSWORD || req.body.password !== process.env.ADMIN_PASSWORD) return res.status(401).json({error:'Password salah.'}); res.cookie('ai_admin', process.env.SESSION_SECRET, { httpOnly:true, sameSite:'lax', secure:false, maxAge: 86400000 }); res.json({ok:true}); });
app.post('/api/logout', (req,res) => { res.clearCookie('ai_admin'); res.json({ok:true}); });
app.post('/api/chat', async (req,res) => { const message=String(req.body.message || '').trim(); if (!message) return res.status(400).json({error:'Tulis pesan terlebih dahulu.'}); const data=await db(), index=await read(INDEX_FILE, {}), result=search(message,data,index); if (!result) return res.json({answer:'Saya belum menemukan jawaban yang cukup relevan pada dataset. Coba gunakan pertanyaan lain atau minta admin menambahkan data.', confidence:0, source:null}); const e=result.entry; const answer=e.answer || e.content; res.json({answer, confidence:Math.round(result.score*100), source:e.source}); });
app.post('/api/train', requireAdmin, async (req,res) => { const data=await db(); data.training={status:'Melatih indeks AI...',updatedAt:new Date().toISOString()}; await write(DB_FILE,data); const index=makeIndex(data.entries); await write(INDEX_FILE,index); data.training={status:'Siap digunakan',updatedAt:new Date().toISOString(),model:'TF-IDF retrieval model (from scratch)'}; await write(DB_FILE,data); res.json({ok:true, entries:data.entries.length}); });
app.post('/api/datasets', requireAdmin, upload.single('file'), async (req,res) => { if(!req.file) return res.status(400).json({error:'Pilih file dataset.'}); const ext=path.extname(req.file.originalname).toLowerCase(); if(!['.txt','.md','.csv','.json'].includes(ext)) return res.status(400).json({error:'Format yang didukung: TXT, MD, CSV, JSON.'}); try { const text=await fs.readFile(req.file.path,'utf8'); const entries=toEntries(req.file.originalname,text,ext); if(!entries.length) throw new Error('Tidak ada data valid. CSV/JSON perlu kolom question dan answer (atau pertanyaan dan jawaban).'); const data=await db(); data.entries.push(...entries); data.datasets.unshift({id:id(),name:req.file.originalname,type:ext.slice(1).toUpperCase(),count:entries.length,createdAt:new Date().toISOString()}); data.training={status:'Perlu dilatih ulang',updatedAt:data.training.updatedAt}; await write(DB_FILE,data); res.json({ok:true,count:entries.length}); } catch(e) { res.status(400).json({error:e.message}); } finally { await fs.unlink(req.file.path).catch(()=>{}); } });
app.post('/api/datasets/huggingface', requireAdmin, async (req,res) => {
  try {
    const imported = await importHuggingFaceParquet(String(req.body.dataset || ''));
    const data = await db();
    data.entries.push(...imported.entries);
    data.datasets.unshift({ id:id(), name:`HF:${imported.dataset}`, type:'HUGGING FACE', count:imported.entries.length, createdAt:new Date().toISOString(), config:imported.config, split:imported.split });
    // Import dari Hugging Face langsung membangun ulang model agar satu link cukup.
    await write(INDEX_FILE, makeIndex(data.entries));
    data.training={status:'Siap digunakan',updatedAt:new Date().toISOString(),model:'TF-IDF retrieval model (from scratch)'};
    await write(DB_FILE,data);
    res.json({ok:true, count:imported.entries.length, downloaded:imported.downloaded, dataset:imported.dataset, trained:true});
  } catch (error) { res.status(400).json({error:error.message}); }
});
app.get('/api/export/jsonl', requireAdmin, async (req,res) => { const data=await db(); const lines=data.entries.filter(e=>e.question&&e.answer).map(e=>JSON.stringify({messages:[{role:'user',content:e.question},{role:'assistant',content:e.answer}]})).join('\n'); res.setHeader('Content-Type','application/jsonl'); res.setHeader('Content-Disposition','attachment; filename=training-dataset.jsonl'); res.send(lines); });
app.delete('/api/datasets/:id', requireAdmin, async (req,res) => { const data=await db(), d=data.datasets.find(x=>x.id===req.params.id); if(!d) return res.status(404).json({error:'Dataset tidak ditemukan'}); data.datasets=data.datasets.filter(x=>x.id!==d.id); data.entries=data.entries.filter(e=>e.source!==d.name); data.training={status:'Perlu dilatih ulang',updatedAt:data.training.updatedAt}; await write(DB_FILE,data); res.json({ok:true}); });

await fs.mkdir(path.join(DATA_DIR,'uploads'),{recursive:true});
app.listen(PORT,'0.0.0.0',()=>console.log(`AI Knowledge Hub: http://0.0.0.0:${PORT}`));
