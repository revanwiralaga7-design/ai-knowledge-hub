import express from 'express';
import multer from 'multer';
import cookieParser from 'cookie-parser';
import dotenv from 'dotenv';
import fs from 'fs/promises';
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
    return rows.map(r => ({ id:id(), source:name, type:'qa', question:r.question || r.prompt || r.input || '', answer:r.answer || r.response || r.output || '', content:'' })).filter(r => r.question && r.answer);
  }
  if (ext === '.csv') {
    const rows = csvParse(text, { columns:true, skip_empty_lines:true, relax_column_count:true });
    return rows.map(r => ({ id:id(), source:name, type:'qa', question:r.question || r.pertanyaan || r.prompt || r.input || '', answer:r.answer || r.jawaban || r.response || r.output || '', content:'' })).filter(r => r.question && r.answer);
  }
  const chunks = text.replace(/\r/g,'').split(/\n\s*\n/).map(x => x.trim()).filter(Boolean);
  return chunks.map(content => ({ id:id(), source:name, type:'document', question:'', answer:'', content:content.slice(0, 5000) }));
}

app.get('/api/status', async (req,res) => { const data=await db(); res.json({ admin:isAdmin(req), datasets:data.datasets, entries:data.entries.length, training:data.training }); });
app.post('/api/login', (req,res) => { if (!process.env.ADMIN_PASSWORD || req.body.password !== process.env.ADMIN_PASSWORD) return res.status(401).json({error:'Password salah.'}); res.cookie('ai_admin', process.env.SESSION_SECRET, { httpOnly:true, sameSite:'lax', secure:false, maxAge: 86400000 }); res.json({ok:true}); });
app.post('/api/logout', (req,res) => { res.clearCookie('ai_admin'); res.json({ok:true}); });
app.post('/api/chat', async (req,res) => { const message=String(req.body.message || '').trim(); if (!message) return res.status(400).json({error:'Tulis pesan terlebih dahulu.'}); const data=await db(), index=await read(INDEX_FILE, {}), result=search(message,data,index); if (!result) return res.json({answer:'Saya belum menemukan jawaban yang cukup relevan pada dataset. Coba gunakan pertanyaan lain atau minta admin menambahkan data.', confidence:0, source:null}); const e=result.entry; const answer=e.answer || e.content; res.json({answer, confidence:Math.round(result.score*100), source:e.source}); });
app.post('/api/train', requireAdmin, async (req,res) => { const data=await db(); data.training={status:'Melatih indeks AI...',updatedAt:new Date().toISOString()}; await write(DB_FILE,data); const index=makeIndex(data.entries); await write(INDEX_FILE,index); data.training={status:'Siap digunakan',updatedAt:new Date().toISOString(),model:'TF-IDF retrieval model (from scratch)'}; await write(DB_FILE,data); res.json({ok:true, entries:data.entries.length}); });
app.post('/api/datasets', requireAdmin, upload.single('file'), async (req,res) => { if(!req.file) return res.status(400).json({error:'Pilih file dataset.'}); const ext=path.extname(req.file.originalname).toLowerCase(); if(!['.txt','.md','.csv','.json'].includes(ext)) return res.status(400).json({error:'Format yang didukung: TXT, MD, CSV, JSON.'}); try { const text=await fs.readFile(req.file.path,'utf8'); const entries=toEntries(req.file.originalname,text,ext); if(!entries.length) throw new Error('Tidak ada data valid. CSV/JSON perlu kolom question dan answer (atau pertanyaan dan jawaban).'); const data=await db(); data.entries.push(...entries); data.datasets.unshift({id:id(),name:req.file.originalname,type:ext.slice(1).toUpperCase(),count:entries.length,createdAt:new Date().toISOString()}); data.training={status:'Perlu dilatih ulang',updatedAt:data.training.updatedAt}; await write(DB_FILE,data); res.json({ok:true,count:entries.length}); } catch(e) { res.status(400).json({error:e.message}); } finally { await fs.unlink(req.file.path).catch(()=>{}); } });
app.get('/api/export/jsonl', requireAdmin, async (req,res) => { const data=await db(); const lines=data.entries.filter(e=>e.question&&e.answer).map(e=>JSON.stringify({messages:[{role:'user',content:e.question},{role:'assistant',content:e.answer}]})).join('\n'); res.setHeader('Content-Type','application/jsonl'); res.setHeader('Content-Disposition','attachment; filename=training-dataset.jsonl'); res.send(lines); });
app.delete('/api/datasets/:id', requireAdmin, async (req,res) => { const data=await db(), d=data.datasets.find(x=>x.id===req.params.id); if(!d) return res.status(404).json({error:'Dataset tidak ditemukan'}); data.datasets=data.datasets.filter(x=>x.id!==d.id); data.entries=data.entries.filter(e=>e.source!==d.name); data.training={status:'Perlu dilatih ulang',updatedAt:data.training.updatedAt}; await write(DB_FILE,data); res.json({ok:true}); });

await fs.mkdir(path.join(DATA_DIR,'uploads'),{recursive:true});
app.listen(PORT,'0.0.0.0',()=>console.log(`AI Knowledge Hub: http://0.0.0.0:${PORT}`));
