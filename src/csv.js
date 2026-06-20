// Minimal robust CSV parser (RFC-4180-ish: quoted fields, escaped quotes, commas in quotes).
const fs = require('fs');

function parseCSV(path) {
  const text = fs.readFileSync(path, 'utf8').replace(/^﻿/, '');
  const rows = [];
  let field = '';
  let record = [];
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ',') { record.push(field); field = ''; }
      else if (c === '\r') { /* skip */ }
      else if (c === '\n') { record.push(field); rows.push(record); record = []; field = ''; }
      else field += c;
    }
  }
  if (field.length > 0 || record.length > 0) { record.push(field); rows.push(record); }
  // drop trailing empty record
  return rows.filter(r => r.length > 1 || (r.length === 1 && r[0] !== ''));
}

function loadDataset(path) {
  const rows = parseCSV(path);
  const header = rows[0];
  const idx = Object.fromEntries(header.map((h, i) => [h.trim(), i]));
  const data = rows.slice(1).map(r => ({
    id: r[idx.id],
    text: r[idx.text],
    label: (r[idx.label] || '').trim(),
    bucket: (r[idx.bucket] || '').trim(),
  }));
  return data;
}

module.exports = { parseCSV, loadDataset };
