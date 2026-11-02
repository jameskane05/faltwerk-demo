// src/rans.ts
var PROB_BITS = 15;
var PROB_SCALE = 1 << PROB_BITS;
var RANS_L = 1 << 23;
var RAW_CHUNK = 15;
var ESC = 16;
var NTOK = ESC + 33;
var INC = 24;
var IDX_INC = 4;
var COUNT_LIMIT = 1 << 16;
var REBUILD_FIRST = 16;
var REBUILD_MAX = 1024;
var FIND_LINEAR_MAX = 64;
var LUT_SHIFT = PROB_BITS - 7;
var AdaptiveModel = class {
  n;
  counts;
  total;
  cdf;
  /** first symbol whose interval reaches into slot bucket b (slot >> LUT_SHIFT) */
  lut = new Uint16Array(1 << PROB_BITS - LUT_SHIFT);
  interval = REBUILD_FIRST;
  until = REBUILD_FIRST;
  inc;
  constructor(n, inc = INC) {
    if (n < 1 || n > PROB_SCALE / 2) throw new Error("alphabet size out of range");
    this.n = n;
    this.inc = inc;
    this.counts = new Uint32Array(n).fill(1);
    this.total = n;
    this.cdf = new Uint32Array(n + 1);
    this.rebuild();
  }
  rebuild() {
    const n = this.n, counts = this.counts, total = this.total, cdf = this.cdf;
    const spare = PROB_SCALE - n;
    let s = 0, best = 0;
    for (let i = 0; i < n; i++) {
      const f = 1 + Math.floor(counts[i] * spare / total);
      cdf[i + 1] = f;
      s += f;
      if (counts[i] > counts[best]) best = i;
    }
    cdf[best + 1] += PROB_SCALE - s;
    let acc = 0;
    cdf[0] = 0;
    for (let i = 0; i < n; i++) {
      acc += cdf[i + 1];
      cdf[i + 1] = acc;
    }
    const lut = this.lut;
    let sym = 0;
    for (let b = 0; b < lut.length; b++) {
      const lo = b << LUT_SHIFT;
      while (cdf[sym + 1] <= lo) sym++;
      lut[b] = sym;
    }
  }
  update(s) {
    const counts = this.counts;
    counts[s] += this.inc;
    this.total += this.inc;
    if (this.total > COUNT_LIMIT) {
      let t = 0;
      for (let i = 0; i < this.n; i++) {
        const c = counts[i] + 1 >>> 1;
        counts[i] = c;
        t += c;
      }
      this.total = t;
    }
    if (--this.until === 0) {
      this.rebuild();
      if (this.interval < REBUILD_MAX) this.interval <<= 1;
      this.until = this.interval;
    }
  }
};
var RansDecoder = class {
  buf;
  pos;
  end;
  x;
  constructor(buf, pos = 0, end = buf.length) {
    this.buf = buf;
    this.end = end;
    if (end - pos === 0) {
      this.pos = pos;
      this.x = 0;
      return;
    }
    if (end - pos < 4) throw new Error("rANS stream shorter than 4 bytes");
    this.x = (buf[pos] | buf[pos + 1] << 8 | buf[pos + 2] << 16 | buf[pos + 3] << 24) >>> 0;
    this.pos = pos + 4;
  }
  renorm(x) {
    const buf = this.buf, end = this.end;
    let pos = this.pos;
    while (x < RANS_L) {
      if (pos >= end) throw new Error("rANS stream overrun");
      x = x << 8 | buf[pos++];
    }
    this.pos = pos;
    return x;
  }
  get(m) {
    const x = this.x;
    const slot = x & PROB_SCALE - 1;
    const cdf = m.cdf;
    const b = slot >>> LUT_SHIFT;
    let s = m.lut[b];
    if (m.n <= FIND_LINEAR_MAX) {
      while (cdf[s + 1] <= slot) s++;
    } else {
      let hi = b + 1 < m.lut.length ? m.lut[b + 1] + 1 : m.n;
      while (hi - s > 1) {
        const mid = s + hi >>> 1;
        if (cdf[mid] <= slot) s = mid;
        else hi = mid;
      }
    }
    const start = cdf[s];
    this.x = this.renorm((cdf[s + 1] - start) * (x >>> PROB_BITS) + slot - start);
    m.update(s);
    return s;
  }
  getBits(nbits) {
    let value = 0;
    while (nbits > 0) {
      const nb = nbits > RAW_CHUNK ? RAW_CHUNK : nbits;
      nbits -= nb;
      const x = this.x;
      value = value * (1 << nb) + (x & (1 << nb) - 1);
      this.x = this.renorm(x >>> nb);
    }
    return value;
  }
  getUint(m) {
    const t = this.get(m);
    if (t < ESC) return t;
    const n = t - ESC;
    if (n === 0) return ESC;
    if (n <= RAW_CHUNK) {
      const x = this.x;
      const bits = x & (1 << n) - 1;
      this.x = this.renorm(x >>> n);
      return (1 << n) + bits + ESC - 1;
    }
    return 2 ** n + this.getBits(n) + ESC - 1;
  }
  getSint(m) {
    const u = this.getUint(m);
    if (u < 1073741824) return u & 1 ? -(u + 1 >> 1) : u >> 1;
    return u % 2 === 0 ? u / 2 : -(u + 1) / 2;
  }
  done() {
    return this.pos === this.end && this.x === RANS_L;
  }
};

// src/knn.ts
var OCC = 4;
var SPAN_MAX = 2 ** 25;
function icbrt(v) {
  if (v <= 0n) return 0;
  let x = 1n << BigInt(Math.floor((v.toString(2).length + 2) / 3));
  for (; ; ) {
    const y = (2n * x + v / (x * x)) / 3n;
    if (y >= x) break;
    x = y;
  }
  while (x * x * x > v) x -= 1n;
  while ((x + 1n) ** 3n <= v) x += 1n;
  return Number(x);
}
function gridCell(lo, hi, n, pos) {
  const ext = [0, 1, 2].map((c) => Math.max(1, hi[c] - lo[c] + 1));
  const vol = BigInt(ext[0]) * BigInt(ext[1]) * BigInt(ext[2]);
  let cell = Math.max(1, icbrt(vol * 8n / BigInt(Math.max(1, n))));
  if (pos) {
    while (cell > 1) {
      const ny = Math.floor((hi[1] - lo[1]) / cell) + 1, nz = Math.floor((hi[2] - lo[2]) / cell) + 1;
      const set = /* @__PURE__ */ new Set();
      for (let i = 0; i < n; i++) {
        const a = Math.floor((pos[i * 3] - lo[0]) / cell), b = Math.floor((pos[i * 3 + 1] - lo[1]) / cell), c = Math.floor((pos[i * 3 + 2] - lo[2]) / cell);
        set.add((a * ny + b) * nz + c);
      }
      if (n <= OCC * set.size) break;
      cell = Math.max(1, Math.floor(cell / 2));
    }
  }
  return cell;
}
function causalKnn(pos, n, k) {
  const out = new Int32Array(n * k).fill(-1);
  if (n === 0 || k === 0) return out;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
    const v = pos[i * 3 + c];
    if (v < lo[c]) lo[c] = v;
    if (v > hi[c]) hi[c] = v;
  }
  const maxExt = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  if (maxExt >= SPAN_MAX) throw new Error("position span too large for exact double arithmetic (knn)");
  let cell = gridCell(lo, hi, n, pos);
  const dims = () => [0, 1, 2].map((c) => Math.floor((hi[c] - lo[c]) / cell) + 1);
  while (dims().reduce((a, b) => a * b, 1) >= 2 ** 52) cell *= 2;
  const [nx, ny, nz] = dims();
  const total = nx * ny * nz;
  const keyOf = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    keyOf[i] = (Math.floor((pos[i * 3] - lo[0]) / cell) * ny + Math.floor((pos[i * 3 + 1] - lo[1]) / cell)) * nz + Math.floor((pos[i * 3 + 2] - lo[2]) / cell);
  }
  const idx = new Int32Array(n);
  let startArr = null;
  let cellMap = null;
  let nonEmpty = 0;
  if (total <= 2 ** 23) {
    startArr = new Int32Array(total + 1);
    for (let i = 0; i < n; i++) startArr[keyOf[i] + 1]++;
    for (let c = 0; c < total; c++) {
      if (startArr[c + 1]) nonEmpty++;
      startArr[c + 1] += startArr[c];
    }
    const cur = startArr.slice(0, total);
    for (let i = 0; i < n; i++) idx[cur[keyOf[i]]++] = i;
  } else {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => keyOf[a] - keyOf[b] || a - b);
    cellMap = /* @__PURE__ */ new Map();
    for (let t = 0; t < n; t++) {
      idx[t] = order[t];
      const e = cellMap.get(keyOf[order[t]]);
      if (e) e[1] = t + 1;
      else {
        cellMap.set(keyOf[order[t]], [t, t + 1]);
        nonEmpty++;
      }
    }
  }
  const bestD = new Float64Array(k), bestJ = new Int32Array(k);
  let cnt = 0;
  const insert = (d2, j) => {
    if (cnt === k && !(d2 < bestD[k - 1] || d2 === bestD[k - 1] && j < bestJ[k - 1])) return;
    let q = cnt < k ? cnt : k - 1;
    while (q > 0 && (d2 < bestD[q - 1] || d2 === bestD[q - 1] && j < bestJ[q - 1])) {
      bestD[q] = bestD[q - 1];
      bestJ[q] = bestJ[q - 1];
      q--;
    }
    bestD[q] = d2;
    bestJ[q] = j;
    if (cnt < k) cnt++;
  };
  for (let i = 1; i < n; i++) {
    const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
    const cx = Math.floor((px - lo[0]) / cell), cy = Math.floor((py - lo[1]) / cell), cz = Math.floor((pz - lo[2]) / cell);
    cnt = 0;
    let r = 0;
    const scan = (s, e) => {
      for (let t = s; t < e; t++) {
        const j = idx[t];
        if (j >= i) break;
        const dx = pos[j * 3] - px, dy = pos[j * 3 + 1] - py, dz = pos[j * 3 + 2] - pz;
        insert(dx * dx + dy * dy + dz * dz, j);
      }
    };
    for (; ; ) {
      for (let dx = -r; dx <= r; dx++) {
        const ax = cx + dx;
        if (ax < 0 || ax >= nx) continue;
        for (let dy = -r; dy <= r; dy++) {
          const ay = cy + dy;
          if (ay < 0 || ay >= ny) continue;
          const edgeXY = Math.abs(dx) === r || Math.abs(dy) === r;
          const step = edgeXY ? 1 : 2 * r;
          for (let dz = -r; dz <= r; dz += step) {
            const az = cz + dz;
            if (az < 0 || az >= nz) continue;
            const key = (ax * ny + ay) * nz + az;
            if (startArr) scan(startArr[key], startArr[key + 1]);
            else {
              const e = cellMap.get(key);
              if (e) scan(e[0], e[1]);
            }
          }
        }
      }
      if (cnt === k && bestD[k - 1] <= r * cell * (r * cell)) break;
      if (cnt === Math.min(k, i) && r * cell > maxExt) break;
      r++;
      if ((2 * r + 1) ** 3 > 8 * nonEmpty) {
        cnt = 0;
        for (let j = 0; j < i; j++) {
          const dx = pos[j * 3] - px, dy = pos[j * 3 + 1] - py, dz = pos[j * 3 + 2] - pz;
          insert(dx * dx + dy * dy + dz * dz, j);
        }
        break;
      }
    }
    for (let t = 0; t < cnt; t++) out[i * k + t] = bestJ[t];
  }
  return out;
}

// src/csgs.ts
var VERSION = 1;
var VERSION_SN_VQ = 2;
var VERSION_PRED = 3;
var VERSION_FOLLOW = 4;
var VERSION_RESCALE = 5;
var VERSION_BSPLINE = 6;
var FLAG_KEY = 1;
var FLAG_SHARED_KNOT = 2;
var FLAG_SN_VQ = 4;
var FLAG_PRED = 8;
var FLAG_FOLLOW = 16;
var FLAG_RESCALE = 32;
var FLAG_BSPLINE = 64;
var PRED_TEMPORAL = 1;
var PRED_SPATIAL = 2;
var PRED_VEL0 = 4;
var FOLLOW_K = 4;
var FOLLOW_CELL = 8;
var ST_K = 3;
var SN_CH = 3;
var MAX_CB = 16384;
function snDim(deg) {
  return SN_CH * ((deg + 1) ** 2 - 1);
}
function maxStages(p) {
  return p.snVq ? Math.max(1, ...p.snVq.map((l) => l.sizes.length)) : 0;
}
var CRC_TABLES = (() => {
  const t = new Uint32Array(4 * 256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 3988292384 ^ c >>> 1 : c >>> 1;
    t[n] = c >>> 0;
  }
  for (let n = 0; n < 256; n++) for (let k = 1; k < 4; k++) t[k * 256 + n] = t[(k - 1) * 256 + n] >>> 8 ^ t[t[(k - 1) * 256 + n] & 255];
  return t;
})();
function crc32(b, start = 0, end = b.length) {
  const T = CRC_TABLES;
  let c = 4294967295, i = start;
  for (; i + 4 <= end; i += 4) {
    c ^= b[i] | b[i + 1] << 8 | b[i + 2] << 16 | b[i + 3] << 24;
    c = T[768 + (c & 255)] ^ T[512 + (c >>> 8 & 255)] ^ T[256 + (c >>> 16 & 255)] ^ T[c >>> 24];
  }
  for (; i < end; i++) c = T[(c ^ b[i]) & 255] ^ c >>> 8;
  return (c ^ 4294967295) >>> 0;
}
function unpackHeader(buf) {
  if (buf.length < 4 || buf[0] !== 67 || buf[1] !== 83 || buf[2] !== 71 || buf[3] !== 83)
    throw new Error("not a .csgs section");
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 4;
  const ver = buf[o++];
  if (ver < VERSION || ver > VERSION_BSPLINE) throw new Error(`unsupported .csgs version ${ver}`);
  const flags = buf[o++];
  if (flags & ~(FLAG_KEY | FLAG_SHARED_KNOT | FLAG_SN_VQ | FLAG_PRED | FLAG_FOLLOW | FLAG_RESCALE | FLAG_BSPLINE))
    throw new Error(`unsupported .csgs flags 0x${flags.toString(16)}`);
  const vq = !!(flags & FLAG_SN_VQ), predF = !!(flags & FLAG_PRED), followF = !!(flags & FLAG_FOLLOW), rescaleF = !!(flags & FLAG_RESCALE);
  const bspline = !!(flags & FLAG_BSPLINE);
  if (ver !== (bspline ? VERSION_BSPLINE : rescaleF ? VERSION_RESCALE : followF ? VERSION_FOLLOW : predF ? VERSION_PRED : vq ? VERSION_SN_VQ : VERSION))
    throw new Error("version and flags (SN_VQ / pred / follow / rescale / bspline) disagree");
  const index = dv.getUint32(o, true);
  o += 4;
  const g0 = dv.getUint32(o, true);
  o += 4;
  const L = dv.getUint16(o, true);
  o += 2;
  const nframes = dv.getUint16(o, true);
  o += 2;
  const nCont = dv.getUint32(o, true);
  o += 4;
  const nNew = dv.getUint32(o, true);
  o += 4;
  const idBase = dv.getUint32(o, true);
  o += 4;
  const keyInterval = buf[o++];
  const G = buf[o++];
  const groups = [];
  for (let g = 0; g < G; g++) {
    groups.push([String.fromCharCode(buf[o]), buf[o + 1]]);
    o += 2;
  }
  const nk = buf[o++];
  const ks = Array.from(buf.subarray(o, o + nk));
  o += nk;
  let pred = null, knnK = 0;
  if (predF) {
    knnK = buf[o++];
    pred = buf.slice(o, o + G);
    o += G;
    if (pred.some((mo) => mo > 7)) throw new Error("pred needs one mode 0..7 per group");
    const spatial = pred.map((mo) => mo & PRED_SPATIAL);
    if (spatial.some((x) => x)) {
      const gp = groups.findIndex(([t, d]) => t === "P" && d === 3);
      if (knnK < 1) throw new Error("knn_k 1..255 for spatial prediction");
      if (gp < 0) throw new Error("spatial prediction needs a group 'P' with 3 dims");
      if (spatial.slice(0, gp).some((x) => x)) throw new Error("group 'P' must be coded before any spatially predicted group");
    } else if (knnK) throw new Error("knn_k only with spatial prediction");
  }
  const nSn = dv.getUint16(o, true);
  o += 2;
  const snPred = buf.slice(o, o + nSn);
  o += nSn;
  let snVq = null, snZero = null;
  if (vq) {
    snZero = buf.slice(o, o + nSn);
    o += nSn;
    const nl = buf[o++];
    snVq = [];
    for (let l = 0; l < nl; l++) {
      const deg = buf[o++], ns2 = buf[o++];
      const sizes = [];
      for (let s = 0; s < ns2; s++) {
        sizes.push(dv.getUint16(o, true));
        o += 2;
      }
      snVq.push({ deg, sizes });
    }
    const degs = snVq.map((l) => l.deg);
    if (!degs.length || degs.some((d, i) => d < 1 || d > 3 || i > 0 && d <= degs[i - 1]) || snDim(degs[degs.length - 1]) > nSn)
      throw new Error("sn_vq levels: strictly increasing degrees in 1..3 with sn_dim(deg) <= n_sn");
    if (snVq.some((l) => !l.sizes.length || l.sizes.some((s) => s < 1 || s > MAX_CB)))
      throw new Error(`sn_vq stages: 1..255 stages of 1..${MAX_CB} entries`);
  }
  const nq = groups.reduce((a, [, d]) => a + d, 0) + nSn;
  const offsets = new Float32Array(nq), scales = new Float32Array(nq);
  for (let i = 0; i < nq; i++) {
    offsets[i] = dv.getFloat32(o, true);
    o += 4;
  }
  for (let i = 0; i < nq; i++) {
    scales[i] = dv.getFloat32(o, true);
    o += 4;
  }
  const ns = buf[o++];
  const lens = [];
  for (let i = 0; i < ns; i++) {
    lens.push(dv.getUint32(o, true));
    o += 4;
  }
  if (!(L >= 1 && L <= 253) || !ks.length) throw new Error("bad L or ks");
  const params = { L, ks, groups, nSn, offsets, scales, sharedKnot: !!(flags & FLAG_SHARED_KNOT), keyInterval, snVq, snZero, pred, knnK, bspline };
  const h = { key: !!(flags & FLAG_KEY), follow: followF, rescale: rescaleF, index, g0, nframes, nCont, nNew, idBase, snPred };
  return { params, h, lens, o };
}
var Models = class {
  birth;
  death;
  ids;
  k;
  first;
  diff;
  sn;
  cbRes = [];
  deg = [];
  idx = [];
  follow;
  // [kidx of group P][bit of the previous knot]
  constructor(p, nframes) {
    const nk = p.ks.length;
    this.follow = Array.from({ length: nk }, () => [new AdaptiveModel(2), new AdaptiveModel(2)]);
    this.birth = Array.from({ length: nframes }, () => new AdaptiveModel(nframes));
    this.death = [new AdaptiveModel(p.L + 2), new AdaptiveModel(p.L + 2)];
    this.ids = new AdaptiveModel(NTOK);
    this.k = p.groups.map(() => Array.from({ length: nk + 1 }, () => new AdaptiveModel(nk)));
    this.first = p.groups.map(([, d]) => [0, 1].map(() => Array.from({ length: d }, () => new AdaptiveModel(NTOK))));
    this.diff = p.groups.map(([, d]) => Array.from({ length: nk }, () => Array.from({ length: d }, () => Array.from({ length: 4 }, () => new AdaptiveModel(NTOK)))));
    this.sn = Array.from({ length: p.nSn }, () => Array.from({ length: 6 }, () => new AdaptiveModel(NTOK)));
    if (p.snVq) {
      const nl = p.snVq.length + 1;
      this.cbRes = Array.from({ length: p.nSn }, () => Array.from({ length: 6 }, () => new AdaptiveModel(NTOK)));
      this.deg = Array.from({ length: nl }, () => new AdaptiveModel(nl));
      this.idx = p.snVq.map((l) => l.sizes.map((s) => new AdaptiveModel(s, IDX_INC)));
    }
  }
};
function bucket6(v) {
  if (v < 0) v = -v;
  return v <= 2 ? v : v <= 4 ? 3 : v <= 8 ? 4 : 5;
}
function bucket(v) {
  if (v < 0) v = -v;
  return v >= 2 ? 3 : v + 1;
}
function vqRow(out, o, p, level, idxs, io, cb) {
  out.set(p.snZero, o);
  if (level === 0) return;
  const stages = cb[level - 1];
  const d = snDim(p.snVq[level - 1].deg);
  const base = stages[0], b0 = idxs[io] * d;
  for (let c = 0; c < d; c++) {
    let v = base[b0 + c];
    for (let s = 1; s < stages.length; s++) v += stages[s][idxs[io + s] * d + c];
    out[o + c] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}
function endVelocity(row, o, K, d, nframes, out, po) {
  if (K < 2 || nframes < 2) {
    for (let c = 0; c < d; c++) out[po + c] = 0;
    return;
  }
  const den = nframes - 1;
  for (let c = 0; c < d; c++)
    out[po + c] = Math.floor((2 * (row[o + (K - 1) * d + c] - row[o + (K - 2) * d + c]) * (K - 1) + den) / (2 * den));
}
function resampleAt(row, o, Kn, d, j, Ki, c) {
  if (Kn === 1 || Ki === 1) return row[o + c];
  const num = j * (Kn - 1), den = Ki - 1;
  let q = Math.floor(num / den);
  const rem = num - q * den;
  if (rem === 0 || q >= Kn - 1) return row[o + Math.min(q, Kn - 1) * d + c];
  const a = row[o + q * d + c], b = row[o + (q + 1) * d + c];
  return a + Math.floor((2 * (b - a) * rem + den) / (2 * den));
}
function predictDiffs(out, row, off, kk, nbs, nb0, cnt, d, Ki) {
  const len = (Ki - 1) * d;
  out.fill(0, 0, len);
  for (let t = 0; t < cnt; t++) {
    const j0 = nbs[nb0 + t], o = off[j0], Kn = kk[j0];
    for (let c = 0; c < d; c++) {
      let prev = resampleAt(row, o, Kn, d, 0, Ki, c);
      for (let j = 1; j < Ki; j++) {
        const cur = resampleAt(row, o, Kn, d, j, Ki, c);
        out[(j - 1) * d + c] += cur - prev;
        prev = cur;
      }
    }
  }
  for (let t = 0; t < len; t++) out[t] = Math.floor((2 * out[t] + cnt) / (2 * cnt));
}
function decodeKnotsPred(p, d, key, st, m, g, ki, k, nc, n, off, row, gpRow, gpOff, knn) {
  const dm = p.groups[g][1], mode = p.pred[g];
  const firstRes = m.first[g][0], firstAbs = m.first[g][1], mdiff = m.diff[g];
  const prevAbs = new Int32Array(dm);
  const last = st ? st.last[g] : null, vel = st ? st.vel[g] : null;
  const vel0 = (mode & PRED_VEL0) !== 0;
  for (let i = 0; i < n; i++) {
    const b = off[i];
    if (!key && i < nc) {
      if (p.sharedKnot) for (let c = 0; c < dm; c++) row[b + c] = last[i * dm + c];
      else if (vel0) for (let c = 0; c < dm; c++) row[b + c] = last[i * dm + c] + vel[i * dm + c] + d.getSint(firstRes[c]);
      else for (let c = 0; c < dm; c++) row[b + c] = last[i * dm + c] + d.getSint(firstRes[c]);
    } else {
      for (let c = 0; c < dm; c++) {
        row[b + c] = prevAbs[c] + d.getSint(firstAbs[c]);
        prevAbs[c] = row[b + c];
      }
    }
  }
  const kg = k[g];
  let nbs = null;
  if (mode & PRED_SPATIAL) {
    if (!knn.v) {
      const pr = gpRow ?? row, po = gpOff ?? off;
      const pos = new Int32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const b = po[i];
        pos[i * 3] = pr[b];
        pos[i * 3 + 1] = pr[b + 1];
        pos[i * 3 + 2] = pr[b + 2];
      }
      knn.v = causalKnn(pos, n, p.knnK);
    }
    nbs = knn.v;
  }
  const knnK = p.knnK;
  const temporal = (mode & PRED_TEMPORAL) !== 0;
  let maxK = 1;
  for (let i = 0; i < n; i++) if (kg[i] > maxK) maxK = kg[i];
  const pred = new Float64Array((maxK - 1) * dm);
  for (let i = 0; i < n; i++) {
    const K = kg[i], b = off[i];
    let hasPred = false;
    if (nbs && K > 1) {
      let cnt = 0;
      while (cnt < knnK && nbs[i * knnK + cnt] >= 0) cnt++;
      if (cnt) {
        predictDiffs(pred, row, off, kg, nbs, i * knnK, cnt, dm, K);
        hasPred = true;
      }
    }
    const mk = mdiff[ki[g][i]];
    for (let c = 0; c < dm; c++) {
      const mc = mk[c];
      let pr = 0, has = false, prevV = 0;
      for (let j = 1; j < K; j++) {
        const r = d.getSint(mc[has ? bucket(pr) : 0]);
        let v = r;
        if (hasPred) {
          if (temporal && j >= 2) v = r + prevV + (pred[(j - 1) * dm + c] - pred[(j - 2) * dm + c]);
          else v = r + pred[(j - 1) * dm + c];
        } else if (temporal && j >= 2) v = r + prevV;
        row[b + j * dm + c] = row[b + (j - 1) * dm + c] + v;
        pr = r;
        has = true;
        prevV = v;
      }
    }
  }
}
function followDen(p) {
  const D = Math.max(...p.ks) - 1;
  if (D < 1 || p.ks.some((k) => k > 1 && D % (k - 1))) throw new Error("follow mode needs every K-1 to divide max(ks)-1");
  return D;
}
function followAt(row, o, K, u, D, c) {
  if (K === 1) return D * row[o + c];
  const s = D / (K - 1);
  const seg = Math.floor(u / s), rem = u - seg * s;
  let v = row[o + seg * 3 + c] * (s - rem);
  if (rem) v += row[o + (seg + 1) * 3 + c] * rem;
  return v * (D / s);
}
function applyFollow(p, sec) {
  const mask = sec.follow;
  if (!mask) return;
  const gP = p.groups.findIndex(([t]) => t === "P");
  const D = followDen(p);
  const row = sec.knots[gP], off = sec.knotOff[gP], K = sec.k[gP], n = sec.n;
  const cand = [];
  for (let i = 0; i < n; i++) if (mask[i] === 0) cand.push(i);
  let F = FOLLOW_CELL;
  const spanOf = () => {
    const lo2 = [Infinity, Infinity, Infinity], hi2 = [-Infinity, -Infinity, -Infinity];
    for (const i of cand) for (let c = 0; c < 3; c++) {
      const v = Math.floor(row[off[i] + c] / F);
      if (v < lo2[c]) lo2[c] = v;
      if (v > hi2[c]) hi2[c] = v;
    }
    return { lo: lo2, hi: hi2 };
  };
  let { lo, hi } = spanOf();
  while (cand.length && (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1) >= 2 ** 52) {
    F *= 2;
    ({ lo, hi } = spanOf());
  }
  const cellOf = (v) => Math.floor(v / F);
  const ny = cand.length ? hi[1] - lo[1] + 1 : 1, nz = cand.length ? hi[2] - lo[2] + 1 : 1, nx = cand.length ? hi[0] - lo[0] + 1 : 1;
  const cells = /* @__PURE__ */ new Map();
  for (const i of cand) {
    const key = ((cellOf(row[off[i]]) - lo[0]) * ny + (cellOf(row[off[i] + 1]) - lo[1])) * nz + (cellOf(row[off[i] + 2]) - lo[2]);
    const l = cells.get(key);
    if (l) l.push(i);
    else cells.set(key, [i]);
  }
  const d2 = (i, q0, q1, q2) => {
    const o = off[i], a = row[o] - q0, b = row[o + 1] - q1, c = row[o + 2] - q2;
    return a * a + b * b + c * c;
  };
  const byDist = (a, b) => a[0] - b[0] || a[1] - b[1];
  const neighbours = (q0, q1, q2, span) => {
    const cq0 = cellOf(q0), cq1 = cellOf(q1), cq2 = cellOf(q2);
    let found = [];
    for (let r = 0; r <= span; r++) {
      if ((2 * r + 1) ** 3 - (2 * r - 1) ** 3 > cells.size) {
        found = cand.map((i) => [d2(i, q0, q1, q2), i]).sort(byDist);
        return found.slice(0, FOLLOW_K).map((x) => x[1]);
      }
      for (let dx = -r; dx <= r; dx++) {
        const ax = cq0 + dx - lo[0];
        if (ax < 0 || ax >= nx) continue;
        for (let dy = -r; dy <= r; dy++) {
          const ay = cq1 + dy - lo[1];
          if (ay < 0 || ay >= ny) continue;
          const edgeXY = Math.abs(dx) === r || Math.abs(dy) === r;
          const step = edgeXY ? 1 : 2 * r;
          for (let dz = -r; dz <= r; dz += step) {
            const az = cq2 + dz - lo[2];
            if (az < 0 || az >= nz) continue;
            const l = cells.get((ax * ny + ay) * nz + az);
            if (l) for (const i of l) found.push([d2(i, q0, q1, q2), i]);
          }
        }
      }
      if (found.length >= FOLLOW_K) {
        found.sort(byDist);
        if (found[FOLLOW_K - 1][0] <= r * F * (r * F)) return found.slice(0, FOLLOW_K).map((x) => x[1]);
      }
    }
    found.sort(byDist);
    return found.slice(0, FOLLOW_K).map((x) => x[1]);
  };
  for (let i = 0; i < n; i++) {
    const m = mask[i];
    if (!m) continue;
    const Ki = K[i], o = off[i];
    if (m & 1 || Ki < 2 || Ki > 31 || m >>> Ki) throw new Error("follow mask: knot 0 or a knot beyond K masked");
    let nb = [];
    if (cand.length) {
      const qx = cellOf(row[o]), qy = cellOf(row[o + 1]), qz = cellOf(row[o + 2]);
      const span = Math.max(Math.abs(qx - lo[0]), Math.abs(qx - hi[0]), Math.abs(qy - lo[1]), Math.abs(qy - hi[1]), Math.abs(qz - lo[2]), Math.abs(qz - hi[2]));
      nb = neighbours(row[o], row[o + 1], row[o + 2], span);
    }
    const st = D / (Ki - 1);
    let a = 0;
    for (let j = 1; j < Ki; j++) {
      if (!(m >>> j & 1)) {
        a = j;
        continue;
      }
      for (let c = 0; c < 3; c++) {
        if (nb.length) {
          let S = 0;
          for (const nn of nb) S += followAt(row, off[nn], K[nn], j * st, D, c) - followAt(row, off[nn], K[nn], a * st, D, c);
          const q = D * nb.length;
          row[o + j * 3 + c] = row[o + a * 3 + c] + Math.floor((S + Math.floor(q / 2)) / q);
        } else row[o + j * 3 + c] = row[o + a * 3 + c];
      }
    }
  }
}
var sameTable = (a, b, from, to) => {
  for (let i = from; i < to; i++) if (a[i] !== b[i]) return false;
  return true;
};
function rescaleState(st, pOld, pNew) {
  const out = { ...st, params: pNew, last: [], vel: [] };
  let q0 = 0;
  pNew.groups.forEach(([, d], g) => {
    const last = new Int32Array(st.last[g].length), vel = new Int32Array(st.vel[g].length);
    const m = last.length / d;
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < d; c++) {
        const oo = pOld.offsets[q0 + c], so = pOld.scales[q0 + c], on = pNew.offsets[q0 + c], sn = pNew.scales[q0 + c];
        last[i * d + c] = Math.floor((oo + st.last[g][i * d + c] * so - on) / sn + 0.5);
        vel[i * d + c] = Math.floor(st.vel[g][i * d + c] * so / sn + 0.5);
      }
    }
    q0 += d;
    out.last.push(last);
    out.vel.push(vel);
  });
  return out;
}
function stateAfter(sec, p, prev = null) {
  const cont = sec.nframes + 1;
  let m = 0;
  for (let i = 0; i < sec.n; i++) if (sec.death[i] === cont) m++;
  const kidx = new Map(p.ks.map((k, i) => [k, i]));
  const vq = !!p.snVq, ms = maxStages(p);
  const degs = vq ? p.snVq.map((l) => l.deg) : [];
  const needVel = true;
  const st = {
    ids: new Uint32Array(m),
    sn: new Uint8Array(m * p.nSn),
    k0: new Uint8Array(m),
    last: p.groups.map(([, d]) => new Int32Array(m * d)),
    nextIndex: sec.index + 1,
    vel: needVel ? p.groups.map(([, d]) => new Int32Array(m * d)) : [],
    params: p,
    snLevel: new Uint8Array(vq ? m : 0),
    snIdx: new Uint16Array(vq ? m * ms : 0),
    snCb: vq ? sec.snCb ?? prev?.snCb ?? null : null
  };
  let j = 0;
  for (let i = 0; i < sec.n; i++) {
    if (sec.death[i] !== cont) continue;
    st.ids[j] = sec.ids[i];
    st.sn.set(sec.sn.subarray(i * p.nSn, (i + 1) * p.nSn), j * p.nSn);
    st.k0[j] = kidx.get(sec.k[0][i]);
    p.groups.forEach(([, d], g) => {
      const endOff = sec.knotOff[g][i + 1];
      st.last[g].set(sec.knots[g].subarray(endOff - d, endOff), j * d);
      if (needVel) endVelocity(sec.knots[g], sec.knotOff[g][i], sec.k[g][i], d, sec.nframes, st.vel[g], j * d);
    });
    if (vq) {
      const dg = sec.snDeg[i];
      st.snLevel[j] = dg ? degs.indexOf(dg) + 1 : 0;
      st.snIdx.set(sec.snIdx.subarray(i * ms, (i + 1) * ms), j * ms);
    }
    j++;
  }
  return st;
}
function decodeSection(buf, st) {
  if (buf.length < 8) throw new Error("CRC mismatch");
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (crc32(buf, 0, buf.length - 4) !== dv.getUint32(buf.length - 4, true)) throw new Error("CRC mismatch");
  const { params: p, h, lens, o: o0 } = unpackHeader(buf);
  const { key, nCont: nc, nNew: nn, nframes: nf } = h;
  const vq = !!p.snVq;
  if (lens.length !== 5 + (vq ? 3 : 0) + (h.follow ? 1 : 0) + p.groups.length || o0 + lens.reduce((a, b) => a + b, 0) + 4 !== buf.length)
    throw new Error("stream table does not match the section size");
  if (!key && (!st || nc !== st.ids.length)) throw new Error("n_cont differs from the decoder state");
  if (key && h.rescale) throw new Error("RESCALE flag on a key section");
  if (!key && st.params) {
    const nKnot = p.groups.reduce((a, [, dd]) => a + dd, 0), old = st.params;
    if (!sameTable(old.offsets, p.offsets, nKnot, p.offsets.length) || !sameTable(old.scales, p.scales, nKnot, p.scales.length))
      throw new Error("the SN quantisation table changed inside a key interval");
    const changed = !sameTable(old.offsets, p.offsets, 0, nKnot) || !sameTable(old.scales, p.scales, 0, nKnot);
    if (changed !== h.rescale) throw new Error("quantisation table change and the RESCALE flag disagree");
    if (h.rescale) st = rescaleState(st, old, p);
  }
  const n = nc + nn, G = p.groups.length, nk = p.ks.length;
  const m = new Models(p, nf);
  const starts = [];
  let o = o0;
  for (const ln of lens) {
    starts.push([o, o + ln]);
    o += ln;
  }
  const dec2 = (i) => new RansDecoder(buf, starts[i][0], starts[i][1]);
  const end = (d2) => {
    if (!(d2.done() || d2.x === 0 && d2.pos === d2.end)) throw new Error("stream not consumed exactly");
  };
  let d = dec2(0);
  const birth = new Uint16Array(nn);
  let pb = 0;
  for (let i = 0; i < nn; i++) {
    pb = d.get(m.birth[pb]);
    birth[i] = pb;
  }
  end(d);
  d = dec2(1);
  const death = new Uint16Array(n);
  for (let i = 0; i < n; i++) death[i] = d.get(m.death[i < nc ? 0 : 1]);
  end(d);
  d = dec2(2);
  const ids = new Uint32Array(n);
  if (key) {
    let prev = 0;
    for (let i = 0; i < nc; i++) {
      prev += d.getUint(m.ids);
      ids[i] = prev;
    }
  } else {
    ids.set(st.ids);
  }
  end(d);
  for (let i = 0; i < nn; i++) ids[nc + i] = h.idBase + i;
  d = dec2(ST_K);
  const ki = p.groups.map(() => new Uint8Array(n));
  for (let i = 0; i < n; i++) {
    const c0 = !key && i < nc ? st.k0[i] : nk;
    const k0 = d.get(m.k[0][c0]);
    ki[0][i] = k0;
    for (let g = 1; g < G; g++) ki[g][i] = d.get(m.k[g][k0]);
  }
  end(d);
  const k = ki.map((row) => row.map((x) => p.ks[x]));
  const knotOff = [], knots = [];
  const knnCache = { v: null };
  for (let g = 0; g < G; g++) {
    const dm = p.groups[g][1];
    const off = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) off[i + 1] = off[i] + k[g][i] * dm;
    const row = new Int32Array(off[n]);
    d = dec2(ST_K + 1 + g);
    if (p.pred) {
      const gp = p.pred[g] & PRED_SPATIAL ? p.groups.findIndex(([t, dd]) => t === "P" && dd === 3) : -1;
      decodeKnotsPred(
        p,
        d,
        key,
        st,
        m,
        g,
        ki,
        k,
        nc,
        n,
        off,
        row,
        gp >= 0 && gp !== g ? knots[gp] : null,
        gp >= 0 && gp !== g ? knotOff[gp] : null,
        knnCache
      );
      end(d);
      knotOff.push(off);
      knots.push(row);
      continue;
    }
    const firstRes = m.first[g][0], firstAbs = m.first[g][1], mdiff = m.diff[g];
    const prevAbs = new Int32Array(dm);
    const last = st ? st.last[g] : null;
    for (let i = 0; i < n; i++) {
      const K = k[g][i], b = off[i];
      if (!key && i < nc) {
        if (p.sharedKnot) for (let c = 0; c < dm; c++) row[b + c] = last[i * dm + c];
        else for (let c = 0; c < dm; c++) row[b + c] = last[i * dm + c] + d.getSint(firstRes[c]);
      } else {
        for (let c = 0; c < dm; c++) {
          row[b + c] = prevAbs[c] + d.getSint(firstAbs[c]);
          prevAbs[c] = row[b + c];
        }
      }
      const mk = mdiff[ki[g][i]];
      for (let c = 0; c < dm; c++) {
        const mc = mk[c];
        let pd = 0, has = false;
        for (let j = 1; j < K; j++) {
          const v = d.getSint(mc[has ? bucket(pd) : 0]);
          row[b + j * dm + c] = row[b + (j - 1) * dm + c] + v;
          pd = v;
          has = true;
        }
      }
    }
    end(d);
    knotOff.push(off);
    knots.push(row);
  }
  d = dec2(ST_K + 1 + G);
  const ns = p.nSn;
  const sn = new Uint8Array(n * ns);
  if (!key) sn.set(st.sn);
  const res = new Int32Array(ns);
  const snPred = h.snPred;
  let snDeg = null, snIdx = null, snCb = null;
  if (!vq) {
    for (let i = key ? 0 : nc; i < n; i++) {
      for (let c = 0; c < ns; c++) {
        const v = d.getSint(m.sn[c][c >= SN_CH ? bucket6(res[c - SN_CH]) : 0]);
        res[c] = v;
        const val = snPred[c] + v;
        if (val < 0 || val > 255) throw new Error("SN value out of range");
        sn[i * ns + c] = val;
      }
    }
    end(d);
  } else {
    end(d);
    const levels = p.snVq, degs = levels.map((l) => l.deg), ms = maxStages(p);
    d = dec2(ST_K + 2 + G);
    let cb;
    if (key) {
      snCb = [];
      for (const { deg, sizes } of levels) {
        const dim = snDim(deg), lvl = [];
        sizes.forEach((size, si) => {
          const entries = new Int16Array(size * dim);
          const mod = si === 0 ? m.sn : m.cbRes;
          for (let e = 0; e < size; e++) {
            for (let c = 0; c < dim; c++) {
              const v = d.getSint(mod[c][c >= SN_CH ? bucket6(res[c - SN_CH]) : 0]);
              res[c] = v;
              const val = si === 0 ? snPred[c] + v : v;
              if (si === 0 && (val < 0 || val > 255) || si > 0 && (v < -255 || v > 255)) throw new Error("codebook entry out of range");
              entries[e * dim + c] = val;
            }
          }
          lvl.push(entries);
        });
        snCb.push(lvl);
      }
      cb = snCb;
    } else {
      if (!st.snCb) throw new Error("delta section without codebooks in the decoder state");
      cb = st.snCb;
    }
    end(d);
    d = dec2(ST_K + 3 + G);
    const lv = new Uint8Array(n);
    if (!key) lv.set(st.snLevel);
    let prevLv = 0;
    for (let i = key ? 0 : nc; i < n; i++) {
      prevLv = d.get(m.deg[prevLv]);
      lv[i] = prevLv;
    }
    end(d);
    d = dec2(ST_K + 4 + G);
    snDeg = new Uint8Array(n);
    snIdx = new Uint16Array(n * ms);
    if (!key) snIdx.set(st.snIdx);
    for (let i = 0; i < n; i++) snDeg[i] = lv[i] ? degs[lv[i] - 1] : 0;
    for (let i = key ? 0 : nc; i < n; i++) {
      const l = lv[i];
      if (l) {
        const mi = m.idx[l - 1];
        for (let si = 0; si < mi.length; si++) snIdx[i * ms + si] = d.get(mi[si]);
      }
      vqRow(sn, i * ns, p, l, snIdx, i * ms, cb);
    }
    end(d);
  }
  const sec = { index: h.index, g0: h.g0, nframes: nf, key, n, nCont: nc, ids, birth, death, k, knotOff, knots, sn, snDeg, snIdx, snCb, follow: null };
  if (h.follow) {
    const gP = p.groups.findIndex(([t]) => t === "P");
    if (gP < 0) throw new Error("follow mode needs a group 'P'");
    d = dec2(lens.length - 1);
    const mask = new Uint32Array(n);
    let any = false;
    for (let i = 0; i < n; i++) {
      let mk = 0, pb2 = 0;
      const KP = k[gP][i], mm = m.follow[ki[gP][i]];
      if (KP > 32) throw new Error("follow masks support at most 32 knots");
      for (let j = 1; j < KP; j++) {
        pb2 = d.get(mm[pb2]);
        mk |= pb2 << j;
      }
      mask[i] = mk >>> 0;
      if (mk) any = true;
    }
    end(d);
    if (any) {
      sec.follow = mask;
      applyFollow(p, sec);
    }
  }
  return { params: p, sec };
}
var Decoder = class {
  state = null;
  params = null;
  decode(buf) {
    const { h } = unpackHeader(buf);
    let st = null;
    if (!h.key) {
      if (!this.state || this.state.nextIndex !== h.index)
        throw new Error("non-key section without its previous section (seek to a key section)");
      st = this.state;
    }
    const { params, sec } = decodeSection(buf, st);
    this.params = params;
    this.state = stateAfter(sec, params, this.state);
    return sec;
  }
};

// src/pack.ts
var GROUP_TAGS = ["P", "R", "S", "O", "D"];
var GROUP_DIMS = [3, 4, 3, 1, 3];
var HDR_U32 = 8;
var SN_U32 = 12;
function checkParams(p) {
  const ok = p.groups.length === 5 && p.groups.every(([t, d], g) => t === GROUP_TAGS[g] && d === GROUP_DIMS[g]);
  if (!ok || p.nSn !== 45 || p.ks.some((k) => k > 63))
    throw new Error(`renderer expects groups P3 R4 S3 O1 D3, 45 SN, K<=63; got ${JSON.stringify(p.groups)} n_sn=${p.nSn}`);
}
function packSection(sec, p, targetN = 0) {
  checkParams(p);
  const nc = sec.n, n = Math.max(nc, targetN | 0);
  const hdr = new Uint32Array(n * HDR_U32);
  let total = 0;
  for (let g = 0; g < 5; g++) total += sec.knots[g].length;
  const knots = new Float32Array(total + countReplicaP(sec, n));
  const base = new Uint32Array(5);
  let q0 = 0, o = 0;
  for (let g = 0; g < 5; g++) {
    base[g] = o;
    const d = GROUP_DIMS[g], src = sec.knots[g];
    for (let i = 0; i < src.length; i++) {
      const c = i % d;
      knots[o + i] = p.offsets[q0 + c] + src[i] * p.scales[q0 + c];
    }
    o += src.length;
    q0 += d;
  }
  for (let i = 0; i < nc; i++) {
    const h = i * HDR_U32;
    hdr[h] = i < sec.nCont ? 0 : sec.g0 + sec.birth[i - sec.nCont];
    hdr[h + 1] = sec.g0 + sec.death[i];
    let kp = 0;
    for (let g = 0; g < 5; g++) {
      kp |= sec.k[g][i] << 6 * g;
      hdr[h + 3 + g] = base[g] + sec.knotOff[g][i];
    }
    hdr[h + 2] = p.bspline ? (kp | 2147483648) >>> 0 : kp;
  }
  for (let r = nc, src = 0; r < n; r++, src = (src + 1) % nc) {
    const copy = Math.floor((r - nc) / nc) + 1;
    hdr.copyWithin(r * HDR_U32, src * HDR_U32, (src + 1) * HDR_U32);
    const kP = sec.k[0][src];
    const from = base[0] + sec.knotOff[0][src];
    hdr[r * HDR_U32 + 3] = o;
    const dx = 6e-3 * copy, dy = -4e-3 * copy, dz = 5e-3 * copy;
    for (let j = 0; j < kP; j++) {
      knots[o++] = knots[from + 3 * j] + dx;
      knots[o++] = knots[from + 3 * j + 1] + dy;
      knots[o++] = knots[from + 3 * j + 2] + dz;
    }
  }
  const sn = new Uint32Array(n * SN_U32);
  const sn8 = new Uint8Array(sn.buffer);
  for (let i = 0; i < n; i++) {
    const s = i < nc ? i : (i - nc) % nc;
    sn8.set(sec.sn.subarray(s * 45, s * 45 + 45), i * SN_U32 * 4);
  }
  const qs = 14;
  return {
    index: sec.index,
    g0: sec.g0,
    nframes: sec.nframes,
    n,
    nCoded: nc,
    hdr,
    knots,
    sn,
    snOffset: p.offsets.slice(qs, qs + 45),
    snScale: p.scales.slice(qs, qs + 45)
  };
}
function countReplicaP(sec, n) {
  let c = 0;
  for (let r = sec.n, src = 0; r < n; r++, src = (src + 1) % sec.n) c += sec.k[0][src] * 3;
  return c;
}

// src/worker.ts
var dec = new Decoder();
var chain = Promise.resolve();
self.onmessage = (e) => {
  const job = e.data;
  const tq = performance.now();
  const fetched = job.buf ? Promise.resolve({ b: new Uint8Array(job.buf), t: tq }) : fetch(job.url).then(async (r) => {
    if (!r.ok) throw new Error(`${job.url}: HTTP ${r.status}`);
    const b = new Uint8Array(await r.arrayBuffer());
    return { b, t: performance.now() };
  });
  chain = chain.then(async () => {
    try {
      const { b, t: tf } = await fetched;
      const t0 = performance.now();
      const sec = dec.decode(b);
      const t1 = performance.now();
      const ps = packSection(sec, dec.params, job.targetN);
      const t2 = performance.now();
      const timing = {
        bytes: b.length,
        key: sec.key,
        n_coded: sec.n,
        n_cont: sec.nCont,
        nframes: sec.nframes,
        fetch_ms: tf - tq,
        wait_ms: t0 - tf,
        decode_ms: t1 - t0,
        pack_ms: t2 - t1
      };
      self.postMessage({ seq: job.seq, ps, timing, clip: job.clip }, [ps.hdr.buffer, ps.knots.buffer, ps.sn.buffer]);
    } catch (err) {
      self.postMessage({ seq: job.seq, error: String(err), clip: job.clip });
    }
  });
};
