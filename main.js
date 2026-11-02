// src/shaders.ts
var ACTOR_DROP_Y = 0.02;
var EVAL_WGSL = (
  /* wgsl */
  `
struct EvalU { t: f32, g0: f32, span: f32, n: u32 };
@group(0) @binding(0) var<uniform> U: EvalU;
@group(0) @binding(1) var<storage, read> hdr: array<u32>;
@group(0) @binding(2) var<storage, read> knots: array<f32>;
@group(0) @binding(3) var<storage, read_write> world: array<vec4f>;

var<private> BSP: bool = false;                  // v6 (hdr word 2 bit 31): clamped cubic B-spline instead of the hat basis
struct BW { s0: u32, p: u32, w: vec4f };
fn knv(i: i32, p: i32, K: i32) -> f32 { return clamp(f32(i - p) / f32(K - p), 0.0, 1.0); }
fn bspl(K: u32) -> BW {                         // local support (NURBS book A2.2), = web/src/bspline.ts bsplineLocal
  let p = i32(min(3u, K - 1u));
  let Ki = i32(K);
  let u = clamp((U.t - U.g0) / U.span, 0.0, 1.0);
  var s = Ki - 1;
  if (u < 1.0) { s = min(Ki - 1, p + i32(floor(u * f32(Ki - p)))); }
  var N = array<f32, 4>(1.0, 0.0, 0.0, 0.0);
  var le = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var ri = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var j = 1; j <= p; j++) {
    le[j] = u - knv(s + 1 - j, p, Ki);
    ri[j] = knv(s + j, p, Ki) - u;
    var saved = 0.0;
    for (var r = 0; r < j; r++) {
      let temp = N[r] / (ri[r + 1] + le[j - r]);
      N[r] = saved + ri[r + 1] * temp;
      saved = le[j - r] * temp;
    }
    N[j] = saved;
  }
  return BW(u32(s - p), u32(p), vec4f(N[0], N[1], N[2], N[3]));
}
fn seg(K: u32) -> vec2f {                       // (segment j, weight of knot j+1)
  if (K <= 1u) { return vec2f(0.0, 0.0); }
  let km1 = f32(K - 1u);
  let tau = clamp((U.t - U.g0) * km1 / U.span, 0.0, km1);
  let j = min(floor(tau), km1 - 1.0);
  return vec2f(j, tau - j);
}
fn ev1(off: u32, K: u32) -> f32 {
  if (BSP) {
    let b = bspl(K); var acc = 0.0;
    for (var r = 0u; r <= b.p; r++) { acc += b.w[r] * knots[off + b.s0 + r]; }
    return acc;
  }
  let s = seg(K); let a = off + u32(s.x);
  if (K <= 1u) { return knots[a]; }
  return mix(knots[a], knots[a + 1u], s.y);
}
fn ev3(off: u32, K: u32) -> vec3f {
  if (BSP) {
    let b = bspl(K); var acc = vec3f(0.0);
    for (var r = 0u; r <= b.p; r++) { let a = off + (b.s0 + r) * 3u; acc += b.w[r] * vec3f(knots[a], knots[a + 1u], knots[a + 2u]); }
    return acc;
  }
  let s = seg(K); let a = off + u32(s.x) * 3u;
  let A = vec3f(knots[a], knots[a + 1u], knots[a + 2u]);
  if (K <= 1u) { return A; }
  return mix(A, vec3f(knots[a + 3u], knots[a + 4u], knots[a + 5u]), s.y);
}
fn ev4(off: u32, K: u32) -> vec4f {
  if (BSP) {
    let b = bspl(K); var acc = vec4f(0.0);
    for (var r = 0u; r <= b.p; r++) { let a = off + (b.s0 + r) * 4u; acc += b.w[r] * vec4f(knots[a], knots[a + 1u], knots[a + 2u], knots[a + 3u]); }
    return acc;
  }
  let s = seg(K); let a = off + u32(s.x) * 4u;
  let A = vec4f(knots[a], knots[a + 1u], knots[a + 2u], knots[a + 3u]);
  if (K <= 1u) { return A; }
  return mix(A, vec4f(knots[a + 4u], knots[a + 5u], knots[a + 6u], knots[a + 7u]), s.y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  let h = i * 8u;
  let fi = floor(U.t);
  let alive = f32(hdr[h]) <= fi && fi < f32(hdr[h + 1u]);
  if (!alive) { world[i * 4u] = vec4f(0.0); return; }
  let kp = hdr[h + 2u];
  BSP = (kp >> 31u) != 0u;
  let p = ev3(hdr[h + 3u], kp & 63u);           // 6 bit K per group (Bad Apple streams: K up to 33)
  var q = ev4(hdr[h + 4u], (kp >> 6u) & 63u);
  q = q / max(length(q), 1e-12);                // nlerp
  let s = ev3(hdr[h + 5u], (kp >> 12u) & 63u);
  let o = ev1(hdr[h + 6u], (kp >> 18u) & 63u);
  let d = ev3(hdr[h + 7u], (kp >> 24u) & 63u);
  world[i * 4u] = vec4f(p.x, p.y - ${ACTOR_DROP_Y}, p.z, 1.0);
  world[i * 4u + 1u] = q;
  world[i * 4u + 2u] = vec4f(s, o);
  world[i * 4u + 3u] = vec4f(d, 0.0);
}
`
);
var EVALQ_WGSL = (
  /* wgsl */
  `
@group(0) @binding(0) var<uniform> H: array<vec4f, 16>;
@group(0) @binding(1) var<storage, read> qd: array<u32>;
@group(0) @binding(2) var<storage, read_write> world: array<vec4f>;
@group(0) @binding(3) var<storage, read> p16: array<u32>;      // WB-1313: 16-bit positions (flag 4), else a dummy

fn hf(k: u32) -> f32 { return H[k >> 2u][k & 3u]; }
fn dq(k: u32, q: f32) -> f32 { return hf(4u + k) + q * hf(18u + k); }
fn u8at(w: u32, b: u32) -> f32 { return f32((w >> (8u * b)) & 255u); }
fn i8at(w: u32, b: u32) -> f32 { return f32(extractBits(bitcast<i32>(w), 8u * b, 8u)); }
fn qmul(a: vec4f, b: vec4f) -> vec4f {          // Hamilton product, wxyz
  return vec4f(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w, a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
               a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y, a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = bitcast<u32>(H[0].z);
  if (i >= n) { return; }
  let w0 = qd[i * 4u]; let w1 = qd[i * 4u + 1u]; let w2 = qd[i * 4u + 2u]; let w3 = qd[i * 4u + 3u];
  var p = vec3f(dq(0u, f32(w0 & 1023u)), dq(1u, f32((w0 >> 10u) & 1023u)), dq(2u, f32((w0 >> 20u) & 1023u)));
  if ((bitcast<u32>(H[0].w) & 4u) != 0u) {
    let a = p16[i * 2u]; let b = p16[i * 2u + 1u];
    p = vec3f(dq(0u, f32(a & 65535u)), dq(1u, f32(a >> 16u)), dq(2u, f32(b & 65535u)));
  }
  var q = vec4f(dq(3u, i8at(w1, 0u)), dq(4u, i8at(w1, 1u)), dq(5u, i8at(w1, 2u)), dq(6u, i8at(w1, 3u)));
  q = q / max(length(q), 1e-12);
  var s = vec3f(dq(7u, u8at(w2, 0u)), dq(8u, u8at(w2, 1u)), dq(9u, u8at(w2, 2u)));
  let o = dq(10u, u8at(w2, 3u));
  let d = vec3f(dq(11u, u8at(w3, 0u)), dq(12u, u8at(w3, 1u)), dq(13u, u8at(w3, 2u)));
  if ((bitcast<u32>(H[0].w) & 1u) != 0u) {
    p = vec3f(dot(H[8], vec4f(p, 1.0)), dot(H[9], vec4f(p, 1.0)), dot(H[10], vec4f(p, 1.0)));
    q = qmul(H[11], q);
    s = s + vec3f(H[12].x);
  }
  world[i * 4u] = vec4f(p.x, p.y - ${ACTOR_DROP_Y}, p.z, 1.0);
  world[i * 4u + 1u] = q;
  world[i * 4u + 2u] = vec4f(s, o);
  world[i * 4u + 3u] = vec4f(d, 0.0);
}
`
);
var PROJECT_WGSL = (
  /* wgsl */
  `
struct Cam { view: mat4x4f, campos: vec4f, intr: vec4f, size: vec4f, n: u32, n2: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var<storage, read> world: array<vec4f>;
@group(0) @binding(2) var<storage, read> sn: array<u32>;
@group(0) @binding(3) var<storage, read> snq: array<f32>;      // 45 offsets, 45 scales
@group(0) @binding(4) var<storage, read_write> splat2d: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> vals: array<u32>;
@group(0) @binding(7) var<storage, read_write> counter: array<atomic<u32>>;   // drawIndirect args

fn shn(i: u32, k: u32) -> f32 {                 // dequantised SN scalar k of splat i
  let w = sn[i * 12u + (k >> 2u)];
  let b = f32((w >> ((k & 3u) * 8u)) & 255u);
  return snq[k] + b * snq[45u + k];
}
fn shc(i: u32, c: u32) -> vec3f {               // coefficient c (1..15) of splat i
  let k = (c - 1u) * 3u;
  return vec3f(shn(i, k), shn(i, k + 1u), shn(i, k + 2u));
}

fn sh_color(i: u32, dc: vec3f, dir: vec3f) -> vec3f {
  let x = dir.x; let y = dir.y; let z = dir.z;
  var r = 0.28209479177387814 * dc;
  if (C.p0 != 0u) { return max(r + 0.5, vec3f(0.0)); }    // SH0-only source (.splq): no higher bands
  r += 0.4886025119029199 * (-y * shc(i, 1u) + z * shc(i, 2u) - x * shc(i, 3u));
  let xx = x * x; let yy = y * y; let zz = z * z; let xy = x * y; let yz = y * z; let xz = x * z;
  r += 1.0925484305920792 * xy * shc(i, 4u) + -1.0925484305920792 * yz * shc(i, 5u)
     + 0.31539156525252005 * (2.0 * zz - xx - yy) * shc(i, 6u) + -1.0925484305920792 * xz * shc(i, 7u)
     + 0.5462742152960396 * (xx - yy) * shc(i, 8u);
  r += -0.5900435899266435 * y * (3.0 * xx - yy) * shc(i, 9u) + 2.890611442640554 * xy * z * shc(i, 10u)
     + -0.4570457994644658 * y * (4.0 * zz - xx - yy) * shc(i, 11u)
     + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shc(i, 12u)
     + -0.4570457994644658 * x * (4.0 * zz - xx - yy) * shc(i, 13u) + 1.445305721320277 * z * (xx - yy) * shc(i, 14u)
     + -0.5900435899266435 * x * (xx - 3.0 * yy) * shc(i, 15u);
  return max(r + 0.5, vec3f(0.0));
}

// C.p1: 0 own sort; 1 sort shared with later eyes (no screen cull); 2 splat2d only, reuse the shared order
fn hide(i: u32) { splat2d[i * 3u] = vec4f(0.0); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= C.n2) { return; }
  let own = C.p1 != 2u;
  if (own) {
    vals[i] = i;
    keys[i] = 0xffffffffu;
  }
  if (i >= C.n) { return; }
  let w0 = world[i * 4u];
  if (w0.w == 0.0) { return; }                   // not alive at t
  let p = w0.xyz;
  let mc = (C.view * vec4f(p, 1.0)).xyz;
  let z = mc.z;
  if (z < C.size.z || z > 1e10) { if (!own) { hide(i); } return; }   // C.size.z = near plane per scene (P13)
  let q = world[i * 4u + 1u];
  let so = world[i * 4u + 2u];
  let s = exp(so.xyz);
  let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
  // rotation columns (gsplat normalized_quat_to_rotmat, wxyz)
  let r0 = vec3f(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy));
  let r1 = vec3f(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx));
  let r2 = vec3f(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
  let sig = s.x * s.x * mat3x3f(r0 * r0.x, r0 * r0.y, r0 * r0.z)
          + s.y * s.y * mat3x3f(r1 * r1.x, r1 * r1.y, r1 * r1.z)
          + s.z * s.z * mat3x3f(r2 * r2.x, r2 * r2.y, r2 * r2.z);
  let W = mat3x3f(C.view[0].xyz, C.view[1].xyz, C.view[2].xyz);
  let cc = W * sig * transpose(W);
  let fx = C.intr.x; let fy = C.intr.y; let cx = C.intr.z; let cy = C.intr.w;
  let Wd = C.size.x; let Hd = C.size.y;
  let tfx = 0.5 * Wd / fx; let tfy = 0.5 * Hd / fy;
  let lxp = (Wd - cx) / fx + 0.3 * tfx; let lxn = cx / fx + 0.3 * tfx;
  let lyp = (Hd - cy) / fy + 0.3 * tfy; let lyn = cy / fy + 0.3 * tfy;
  let tx = z * min(lxp, max(-lxn, mc.x / z));
  let ty = z * min(lyp, max(-lyn, mc.y / z));
  let j0 = vec3f(fx / z, 0.0, -fx * tx / (z * z));
  let j1 = vec3f(0.0, fy / z, -fy * ty / (z * z));
  let a = dot(j0, cc * j0) + 0.3;
  let b = dot(j0, cc * j1);
  let c = dot(j1, cc * j1) + 0.3;
  let det = a * c - b * b;
  let op = 1.0 / (1.0 + exp(-so.w));
  if (det <= 0.0 || op < 1.0 / 255.0) { if (!own) { hide(i); } return; }
  let conic = vec3f(c / det, -b / det, a / det);
  let mean2 = vec2f(fx * mc.x / z + cx, fy * mc.y / z + cy);
  let bm = 0.5 * (a + c);
  let v1 = bm + sqrt(max(0.01, bm * bm - det));
  // beyond sqrt(2 ln(255 op)) sigma the fragment's alpha is under 1/255 and discarded
  let rad = ceil(min(3.0, sqrt(2.0 * log(255.0 * op))) * sqrt(v1));
  let off = rad <= 0.0 || mean2.x + rad <= 0.0 || mean2.x - rad >= Wd || mean2.y + rad <= 0.0 || mean2.y - rad >= Hd;
  if (off && C.p1 != 1u) { if (!own) { hide(i); } return; }
  let col = sh_color(i, world[i * 4u + 3u].xyz, normalize(p - C.campos.xyz));
  splat2d[i * 3u] = vec4f(mean2, z, rad);
  splat2d[i * 3u + 1u] = vec4f(conic, op);
  splat2d[i * 3u + 2u] = vec4f(col, 0.0);
  if (!own) { return; }
  keys[i] = bitcast<u32>(z);
  atomicAdd(&counter[1], 1u);
}
`
);
var SORT_WGSL = (
  /* wgsl */
  `
struct SortU { k: u32, j: u32, mode: u32, p: u32 };
@group(0) @binding(0) var<uniform> S: SortU;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals: array<u32>;

@compute @workgroup_size(256)
fn global_step(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let j = S.j;
  let i = 2u * j * (t / j) + (t % j);
  let l = i + j;
  if (l >= arrayLength(&keys)) { return; }
  let asc = (i & S.k) == 0u;
  let ki = keys[i]; let kl = keys[l];
  if ((ki > kl) == asc && ki != kl) {
    keys[i] = kl; keys[l] = ki;
    let vi = vals[i]; vals[i] = vals[l]; vals[l] = vi;
  }
}

var<workgroup> sk: array<u32, 512>;
var<workgroup> sv: array<u32, 512>;

fn lstep(t: u32, base: u32, k: u32, j: u32) {
  let i = 2u * j * (t / j) + (t % j);
  let l = i + j;
  let asc = ((base + i) & k) == 0u;
  let ki = sk[i]; let kl = sk[l];
  if ((ki > kl) == asc && ki != kl) {
    sk[i] = kl; sk[l] = ki;
    let vi = sv[i]; sv[i] = sv[l]; sv[l] = vi;
  }
}

// mode 0: full bitonic sort of every 512-block (k = 2..512); mode 1: finish merge stage S.k for j = 256..1
@compute @workgroup_size(256)
fn local_sort(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let t = lid.x;
  let base = wid.x * 512u;
  sk[t] = keys[base + t]; sk[t + 256u] = keys[base + t + 256u];
  sv[t] = vals[base + t]; sv[t + 256u] = vals[base + t + 256u];
  workgroupBarrier();
  if (S.mode == 0u) {
    for (var k = 2u; k <= 512u; k = k * 2u) {
      for (var j = k / 2u; j > 0u; j = j / 2u) {
        lstep(t, base, k, j);
        workgroupBarrier();
      }
    }
  } else {
    for (var j = 256u; j > 0u; j = j / 2u) {
      lstep(t, base, S.k, j);
      workgroupBarrier();
    }
  }
  keys[base + t] = sk[t]; keys[base + t + 256u] = sk[t + 256u];
  vals[base + t] = sv[t]; vals[base + t + 256u] = sv[t + 256u];
}
`
);
var RASTER_WGSL = (
  /* wgsl */
  `
struct Cam { view: mat4x4f, campos: vec4f, intr: vec4f, size: vec4f, n: u32, n2: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var<storage, read> splat2d: array<vec4f>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;

// Real-world depth (WebXR depth-sensing, matchDepthView false); see Occlusion in xr.ts.
// params: rawValueToMeters, kind (0 off, 1 float32, 2 uint16), depth size
// extra: near, margin (metres), far, w 1 = texels are hardware depth 0..1 between near and far (Quest), 0 = metres from near
// dbg.x 1: occluded fragments are tinted red instead of dropped
struct Occ { F: mat4x4f, zrow: vec4f, params: vec4f, extra: vec4f, dbg: vec4f };
@group(1) @binding(0) var<uniform> O: Occ;
@group(1) @binding(1) var depth_f32: texture_2d<f32>;
@group(1) @binding(2) var depth_u16: texture_2d<u32>;

fn real_occludes(px: vec2f, z: f32) -> bool {
  let kind = O.params.y;
  if (kind < 0.5) { return false; }
  let w = vec4f(px * z, z, 1.0);
  let h = O.F * w;
  if (h.w <= 0.0) { return false; }
  let uv = h.xy / h.w;
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return false; }
  let t = vec2i(uv * O.params.zw);
  var raw: f32;
  if (kind > 1.5) { raw = f32(textureLoad(depth_u16, t, 0).r); } else { raw = textureLoad(depth_f32, t, 0).r; }
  let dist = -dot(O.zrow, w);
  if (O.extra.w > 0.5) {
    let n = O.extra.x; let f = O.extra.z;
    let d = select(raw, raw / 65535.0, kind > 1.5);
    if (d <= 0.0 || d >= 1.0) { return false; }
    return n * f / (f - d * (f - n)) + O.extra.y < dist;
  }
  let meters = raw * O.params.x;
  return meters > 0.0 && meters + O.extra.y < dist - O.extra.x;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) d: vec2f,
  @location(1) conic: vec3f,
  @location(2) col: vec4f,
  @location(3) @interpolate(flat) z: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let idx = vals[ii];
  let a = splat2d[idx * 3u];
  let b = splat2d[idx * 3u + 1u];
  let c = splat2d[idx * 3u + 2u];
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  let px = a.xy + corner * a.w;
  var o: VOut;
  o.pos = vec4f(px.x / C.size.x * 2.0 - 1.0, 1.0 - px.y / C.size.y * 2.0, 0.0, 1.0);
  o.d = corner * a.w;
  o.conic = b.xyz;
  o.col = vec4f(c.xyz, b.w);
  o.z = a.z;
  return o;
}

struct FragOut {
  @location(0) color: vec4f,
  @location(1) depth: vec4f,
};

@fragment
fn fs(v: VOut) -> FragOut {
  let d = v.d;
  let sigma = 0.5 * (v.conic.x * d.x * d.x + v.conic.z * d.y * d.y) + v.conic.y * d.x * d.y;
  if (sigma < 0.0) { discard; }
  let alpha = min(0.999, v.col.w * exp(-sigma));
  if (alpha < 1.0 / 255.0) { discard; }
  var o: FragOut;
  let solid = alpha > 0.12 && v.z > 0.05;
  o.depth = vec4f(select(1e3, v.z, solid), 0.0, 0.0, 0.0);
  if (real_occludes(v.pos.xy, v.z)) {
    if (O.dbg.x < 0.5) { discard; }
    o.color = vec4f(alpha, 0.0, 0.0, alpha);
    return o;
  }
  o.color = vec4f(v.col.rgb * alpha, alpha);
  return o;
}

@fragment
fn fs_color(v: VOut) -> @location(0) vec4f {
  let d = v.d;
  let sigma = 0.5 * (v.conic.x * d.x * d.x + v.conic.z * d.y * d.y) + v.conic.y * d.x * d.y;
  if (sigma < 0.0) { discard; }
  let alpha = min(0.999, v.col.w * exp(-sigma));
  if (alpha < 1.0 / 255.0) { discard; }
  if (real_occludes(v.pos.xy, v.z)) {
    if (O.dbg.x < 0.5) { discard; }
    return vec4f(alpha, 0.0, 0.0, alpha);
  }
  return vec4f(v.col.rgb * alpha, alpha);
}
`
);
var SHADOW_RASTER_WGSL = (
  /* wgsl */
  `
struct Cam { view: mat4x4f, campos: vec4f, intr: vec4f, size: vec4f, n: u32, n2: u32, p0: u32, p1: u32 };
struct ShadowU { light: vec4f, params: vec4f, n: u32, stride: u32, pad0: u32, pad1: u32 };
// light.xyz = direction toward the light (unit), light.w = floor Y
// params: strength, max world radius, fade height, soft (sigma scale)
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var<storage, read> world: array<vec4f>;
@group(0) @binding(2) var<uniform> S: ShadowU;

struct Occ { F: mat4x4f, zrow: vec4f, params: vec4f, extra: vec4f, dbg: vec4f };
@group(1) @binding(0) var<uniform> O: Occ;
@group(1) @binding(1) var depth_f32: texture_2d<f32>;
@group(1) @binding(2) var depth_u16: texture_2d<u32>;

fn real_occludes(px: vec2f, z: f32) -> bool {
  let kind = O.params.y;
  if (kind < 0.5) { return false; }
  let w = vec4f(px * z, z, 1.0);
  let h = O.F * w;
  if (h.w <= 0.0) { return false; }
  let uv = h.xy / h.w;
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return false; }
  let t = vec2i(uv * O.params.zw);
  var raw: f32;
  if (kind > 1.5) { raw = f32(textureLoad(depth_u16, t, 0).r); } else { raw = textureLoad(depth_f32, t, 0).r; }
  let dist = -dot(O.zrow, w);
  if (O.extra.w > 0.5) {
    let n = O.extra.x; let f = O.extra.z;
    let d = select(raw, raw / 65535.0, kind > 1.5);
    if (d <= 0.0 || d >= 1.0) { return false; }
    return n * f / (f - d * (f - n)) + O.extra.y < dist;
  }
  let meters = raw * O.params.x;
  return meters > 0.0 && meters + O.extra.y < dist - O.extra.x;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) d: vec2f,
  @location(1) @interpolate(flat) rad: f32,
  @location(2) @interpolate(flat) op: f32,
  @location(3) @interpolate(flat) z: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(0.0, 0.0, 0.0, 1.0);
  o.d = vec2f(0.0); o.rad = 0.0; o.op = 0.0; o.z = 0.0;
  let idx = ii * max(S.stride, 1u);
  if (idx >= S.n) { return o; }
  let w0 = world[idx * 4u];
  if (w0.w == 0.0) { return o; }
  let L = S.light.xyz;
  let floorY = S.light.w;
  let Ly = L.y;
  if (Ly < 0.08) { return o; }
  let p = w0.xyz;
  let h = p.y - floorY;
  let fade = max(S.params.z, 1e-3);
  if (h < -0.02 || h > fade) { return o; }
  let t = h / Ly;
  let sp = vec3f(p.x - L.x * t, floorY, p.z - L.z * t);
  let mc = (C.view * vec4f(sp, 1.0)).xyz;
  let z = mc.z;
  if (z < C.size.z || z > 1e10) { return o; }
  let so = world[idx * 4u + 2u];
  let sc = 0.333 * (exp(so.x) + exp(so.y) + exp(so.z));
  // Volume splats larger than the cap become the scalloped outline. Dense small ones carry the shape.
  if (sc > S.params.y) { return o; }
  let worldR = clamp(sc * 1.5, 0.01, S.params.y);
  let fx = C.intr.x; let fy = C.intr.y;
  let mean2 = vec2f(fx * mc.x / z + C.intr.z, fy * mc.y / z + C.intr.w);
  let rad = worldR * 0.5 * (fx + fy) / z;
  if (rad < 0.75) { return o; }
  let opSplat = 1.0 / (1.0 + exp(-so.w));
  let fadeK = clamp(1.0 - h / fade, 0.0, 1.0);
  let op = min(0.72, opSplat * S.params.x * fadeK * fadeK);
  if (op < 1.0 / 255.0) { return o; }
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  let px = mean2 + corner * rad;
  o.pos = vec4f(px.x / C.size.x * 2.0 - 1.0, 1.0 - px.y / C.size.y * 2.0, 0.0, 1.0);
  o.d = corner * rad;
  o.rad = rad;
  o.op = op;
  o.z = z;
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  if (v.op < 1.0 / 255.0 || v.rad < 1e-3) { discard; }
  let soft = max(S.params.w, 0.35);
  let r = length(v.d) / v.rad;
  if (r > 1.0) { discard; }
  let alpha = v.op * exp(-r * r / (soft * soft));
  if (alpha < 1.0 / 255.0) { discard; }
  if (real_occludes(v.pos.xy, v.z)) { discard; }
  // Write coverage into every channel so a buggy alpha-only max-blend still survives.
  return vec4f(alpha, alpha, alpha, alpha);
}
`
);
var GRADE_WGSL = (
  /* wgsl */
  `
struct Grade { gain: vec4f, sat: vec4f, tone: vec4f }; // gain.w contrast; sat.x saturation, sat.y on
fn graded(c: vec4f, G: Grade) -> vec3f {
  let rgb = clamp(c.rgb, vec3f(0.0), vec3f(1.0));
  if (G.sat.y < 0.5 || c.a <= 1e-4) { return rgb; }
  var x = clamp(c.rgb / c.a, vec3f(0.0), vec3f(1.0));
  x = pow(min(pow(x, vec3f(2.2)) * G.gain.rgb, vec3f(1.0)), vec3f(1.0 / 2.2));
  x = (x - 0.5) * G.gain.w + 0.5;
  x = mix(vec3f(dot(x, vec3f(0.2126, 0.7152, 0.0722))), x, G.sat.x);
  x = clamp(x, vec3f(0.0), vec3f(1.0));
  let inv = 1.0 - x;
  x += inv * inv * G.tone.x + x * inv * G.tone.y + x * x * G.tone.z;
  return clamp(x, vec3f(0.0), vec3f(1.0)) * clamp(c.a, 0.0, 1.0);
}
`
);
var SHADOW_COMPOSE_WGSL = (
  /* wgsl */
  `
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var shadow: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let uv = vec2i(p.xy);
  let c = textureLoad(color, uv, 0);
  let shSize = vec2i(textureDimensions(shadow));
  let sp = min(uv, shSize - vec2i(1));
  let s = textureLoad(shadow, sp, 0);
  let sa = clamp(max(max(s.r, s.g), max(s.b, s.a)), 0.0, 1.0);
  let a = clamp(c.a, 0.0, 1.0);
  return vec4f(c.rgb, a + (1.0 - a) * sa);
}
`
);
var BLIT_WGSL = (
  /* wgsl */
  `
${GRADE_WGSL}
struct BlitU { bg: vec4f, grade: Grade };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: BlitU;
@group(0) @binding(2) var shadow: texture_2d<f32>;
fn shadowCov(p: vec2i) -> f32 {
  let shSize = vec2i(textureDimensions(shadow));
  let s = textureLoad(shadow, clamp(p, vec2i(0), shSize - vec2i(1)), 0);
  return clamp(max(max(s.r, s.g), max(s.b, s.a)), 0.0, 1.0);
}
fn shadowBlur(p: vec2i) -> f32 {
  var a = 0.0;
  var wsum = 0.0;
  for (var y = -4; y <= 4; y++) {
    for (var x = -4; x <= 4; x++) {
      let w = 1.0 / (1.0 + f32(x * x + y * y) * 0.35);
      a += w * shadowCov(p + vec2i(x, y) * 3);
      wsum += w;
    }
  }
  return a / wsum;
}
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let c = textureLoad(src, vec2i(p.xy), 0);
  let sa = shadowBlur(vec2i(p.xy));
  let hole = 1.0 - clamp(c.a, 0.0, 1.0);
  return vec4f(clamp(graded(c, U.grade) + hole * (1.0 - sa) * U.bg.rgb, vec3f(0.0), vec3f(1.0)), 1.0);
}
`
);
var BLIT_XR_WGSL = (
  /* wgsl */
  `
${GRADE_WGSL}
// hud (P40, WB-1341): x fps, y splats in thousands, z on, w digit height in px \u2014 a seven-segment readout inside the
// headset (DOM overlays are not visible in immersive-ar), below the centre of each eye.
// rs (P40 Quest render scale): xy = splat-target size / eye viewport size, z = 1 when they differ (bilinear upscale)
struct BlitU { origin: vec2f, flags: vec2f, grade: Grade, hud: vec4f, rs: vec4f }; // flags.x keepAlpha, flags.y encodeSrgb
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: BlitU;
@group(0) @binding(2) var shadow: texture_2d<f32>;
fn shadowCov(p: vec2i) -> f32 {
  let shSize = vec2i(textureDimensions(shadow));
  let s = textureLoad(shadow, clamp(p, vec2i(0), shSize - vec2i(1)), 0);
  return clamp(max(max(s.r, s.g), max(s.b, s.a)), 0.0, 1.0);
}
fn shadowBlur(p: vec2i) -> f32 {
  var a = 0.0;
  var wsum = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let w = 1.0 / (1.0 + f32(x * x + y * y) * 0.35);
      a += w * shadowCov(p + vec2i(x, y) * 2);
      wsum += w;
    }
  }
  return a / wsum;
}

fn lin2srgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

// segments a b c d e f g = bits 0..6; u, v in [0, 1] inside one digit cell (v down)
fn segOn(bits: u32, u: f32, v: f32) -> bool {
  let t = 0.16;
  let top = v < t; let bot = v > 1.0 - t; let mid = abs(v - 0.5) < t * 0.5;
  let left = u < t; let right = u > 1.0 - t; let inner = u >= t * 0.5 && u <= 1.0 - t * 0.5;
  let upper = v <= 0.5; let lower = v >= 0.5;
  return ((bits & 1u) != 0u && top && inner) || ((bits & 2u) != 0u && right && upper) ||
         ((bits & 4u) != 0u && right && lower) || ((bits & 8u) != 0u && bot && inner) ||
         ((bits & 16u) != 0u && left && lower) || ((bits & 32u) != 0u && left && upper) ||
         ((bits & 64u) != 0u && mid && inner);
}

/** 0 outside the readout, 1 background, 2 lit segment */
fn hudAt(p: vec2f, dims: vec2f) -> u32 {
  var digs = array<u32, 10>(0x3Fu, 0x06u, 0x5Bu, 0x4Fu, 0x66u, 0x6Du, 0x7Du, 0x07u, 0x7Fu, 0x6Fu);
  let dh = U.hud.w; let dw = dh * 0.55; let gap = dh * 0.25; let cell = dw + gap;
  let boxw = 7.0 * cell;
  let q = p - vec2f(dims.x * 0.5 - boxw * 0.5, dims.y * 0.66);
  if (q.x < -gap || q.y < -gap || q.x > boxw + gap || q.y > dh + gap) { return 0u; }
  let ci = i32(floor(q.x / cell));
  let u = (q.x - f32(ci) * cell) / dw; let v = q.y / dh;
  if (ci < 0 || ci > 6 || ci == 3 || u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) { return 1u; }
  var n = u32(max(U.hud.x, 0.0) + 0.5); var pos = 2 - ci;
  if (ci > 3) { n = u32(max(U.hud.y, 0.0) + 0.5); pos = 6 - ci; }
  n = min(n, 999u);
  var p10 = 1u;
  for (var k = 0; k < pos; k++) { p10 = p10 * 10u; }
  if (n < p10 && pos > 0) { return 1u; }
  return select(1u, 2u, segOn(digs[(n / p10) % 10u], u, v));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(src));
  var shadowTexel = clamp(vec2i(p.xy) - vec2i(U.origin), vec2i(0), dims - vec2i(1));
  var texel = shadowTexel;
  var hdims = vec2f(dims);
  var c = textureLoad(src, texel, 0);
  if (U.rs.z > 0.5) {                            // splats rendered below the eye resolution: bilinear upscale
    let q = p.xy - U.origin;
    let f = q * U.rs.xy - vec2f(0.5);
    let i0 = vec2i(floor(f)); let w = f - floor(f); let mx = dims - vec2i(1);
    let c00 = textureLoad(src, clamp(i0, vec2i(0), mx), 0);
    let c10 = textureLoad(src, clamp(i0 + vec2i(1, 0), vec2i(0), mx), 0);
    let c01 = textureLoad(src, clamp(i0 + vec2i(0, 1), vec2i(0), mx), 0);
    let c11 = textureLoad(src, clamp(i0 + vec2i(1, 1), vec2i(0), mx), 0);
    c = mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);
    let srcPx = vec2i(floor(f + vec2f(0.5)));
    shadowTexel = clamp(srcPx, vec2i(0), mx);
    texel = vec2i(q);
    hdims = vec2f(dims) / U.rs.xy;
  }
  var rgb = graded(c, U.grade);
  let sa = shadowBlur(shadowTexel);
  var a = select(1.0, clamp(c.a, 0.0, 1.0) + (1.0 - clamp(c.a, 0.0, 1.0)) * sa, U.flags.x > 0.5);
  if (U.flags.x < 0.5) { rgb = rgb + (1.0 - clamp(c.a, 0.0, 1.0)) * (1.0 - sa) * vec3f(0.0); }
  if (U.hud.z > 0.5) {
    let h = hudAt(vec2f(texel), hdims);
    if (h > 0u) {
      // lit: green >= 70 fps, amber >= 55, red below
      let lit = select(select(vec3f(0.9, 0.1, 0.1), vec3f(0.9, 0.6, 0.05), U.hud.x >= 55.0), vec3f(0.1, 0.85, 0.2), U.hud.x >= 70.0);
      rgb = select(vec3f(0.01), lit, h == 2u);
      a = 0.9;
    }
  }
  if (U.flags.y > 0.5) { rgb = lin2srgb(rgb); }
  return vec4f(rgb, a);
}
`
);
var DIFF_WGSL = (
  /* wgsl */
  `
struct DiffU { ox: f32, oy: f32, dw: f32, dh: f32, refW: f32, refH: f32, gain: f32, pad: f32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var refTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> U: DiffU;
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let c = clamp(textureLoad(src, vec2i(p.xy), 0).rgb, vec3f(0.0), vec3f(1.0));
  let u = (p.xy - vec2f(U.ox, U.oy)) / vec2f(U.dw, U.dh);
  var g = vec3f(0.0);
  if (u.x >= 0.0 && u.y >= 0.0 && u.x < 1.0 && u.y < 1.0) {
    g = textureLoad(refTex, vec2i(u * vec2f(U.refW, U.refH)), 0).rgb;
  }
  return vec4f(clamp(abs(c - g) * U.gain, vec3f(0.0), vec3f(1.0)), 1.0);
}
`
);
var DESK_OCC_WGSL = (
  /* wgsl */
  `
struct Occ { F: mat4x4f, zrow: vec4f, params: vec4f, extra: vec4f, dbg: vec4f };
@group(1) @binding(0) var<uniform> O: Occ;
@group(1) @binding(1) var depth_f32: texture_2d<f32>;
@group(1) @binding(2) var depth_u16: texture_2d<u32>;
@group(1) @binding(3) var depth_d: texture_depth_2d;
fn occluded(w: vec4f) -> bool {
  let kind = O.params.y;
  if (kind < 0.5) { return false; }
  let h = O.F * w;
  if (h.w <= 0.0) { return false; }
  let uv = h.xy / h.w;
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return false; }
  let t = vec2i(uv * O.params.zw);
  var raw: f32;
  if (kind > 2.5) { raw = textureLoad(depth_d, t, 0); }
  else if (kind > 1.5) { raw = f32(textureLoad(depth_u16, t, 0).r); }
  else { raw = textureLoad(depth_f32, t, 0).r; }
  let dist = -dot(O.zrow, w);
  if (O.extra.w > 0.5) {
    let n = O.extra.x; let f = O.extra.z;
    let d = select(raw, raw / 65535.0, kind > 1.5 && kind < 2.5);
    if (d <= 0.0 || d >= 1.0) { return false; }
    return n * f / (f - d * (f - n)) + O.extra.y < dist;
  }
  let meters = raw * O.params.x;
  return meters > 0.0 && meters + O.extra.y < dist - O.extra.x;
}
`
);
var DESK_WGSL = (
  /* wgsl */
  `
${DESK_OCC_WGSL}
struct DeskU { mvp: mat4x4f, model: mat4x4f, flags: vec4f };
@group(0) @binding(0) var<uniform> U: DeskU;
struct V { @builtin(position) pos: vec4f, @location(0) col: vec4f, @location(1) world: vec3f };
@vertex
fn vs(@location(0) p: vec3f, @location(1) col: vec4f) -> V {
  var o: V;
  o.pos = U.mvp * vec4f(p, 1.0);
  o.col = col;
  o.world = (U.model * vec4f(p, 1.0)).xyz;
  return o;
}
fn srgb2lin(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
@fragment
fn fs(v: V) -> @location(0) vec4f {
  let n = normalize(cross(dpdx(v.world), dpdy(v.world)));
  let shade = select(0.55 + 0.45 * abs(dot(n, normalize(vec3f(0.3, 1.0, 0.5)))), 1.0, U.flags.y > 0.5);
  var c = clamp(v.col.rgb * shade, vec3f(0.0), vec3f(1.0));
  if (occluded(vec4f(v.world, 1.0))) {
    if (O.dbg.x < 0.5) { discard; }
    c = vec3f(1.0, 0.0, 0.0);
  }
  if (U.flags.x > 0.5) { c = srgb2lin(c); }
  return vec4f(c, clamp(v.col.a, 0.0, 1.0));
}
`
);

// src/splq.ts
var SPLQ_HDR = 256;
var SPLQ_FLAG_SN = 2;
var SPLQ_FLAG_P16 = 4;
function splqSnWords(fr) {
  if (!fr.sn) return null;
  const out = new Uint8Array(fr.n * 48), m = fr.snM, st = fr.snStride, src = fr.sn;
  for (let i = 0; i < fr.n; i++) out.set(src.subarray(i * st, i * st + m), i * 48);
  return new Uint32Array(out.buffer);
}

// src/i18n.ts
var nav = typeof navigator === "undefined" ? void 0 : navigator;
var forced = typeof location === "undefined" ? null : new URLSearchParams(location.search).get("lang");
var LANG = forced === "de" || forced === "en" ? forced : (nav?.languages?.[0] ?? nav?.language ?? "en").toLowerCase().startsWith("de") ? "de" : "en";
var tr = (de, en) => LANG === "de" ? de : en;
function localizeDom(root = document) {
  document.documentElement.lang = LANG;
  if (LANG !== "de") return;
  root.querySelectorAll("[data-de]").forEach((el) => {
    el.textContent = el.dataset.de;
  });
  root.querySelectorAll("[data-de-title]").forEach((el) => {
    el.title = el.dataset.deTitle;
  });
  const t = document.querySelector("title");
  if (t?.dataset.de) document.title = t.dataset.de;
}

// src/recolor.ts
var NEUTRAL = { exposure: 0, temperature: 0, tint: 0, saturation: 1, contrast: 1, shadows: 0, midtones: 0, highlights: 0 };
var GRADE_FIELDS = [
  {
    field: "exposure",
    label: tr("Belichtung", "Exposure"),
    short: tr("Bel", "Exp"),
    min: -2,
    max: 2,
    step: 0.05,
    tip: tr("Helligkeit in Blendenstufen: +1 verdoppelt, \u22121 halbiert das Licht.", "Brightness in stops: +1 doubles the light, \u22121 halves it.")
  },
  {
    field: "temperature",
    label: tr("Temperatur", "Temperature"),
    short: "Temp",
    min: -1,
    max: 1,
    step: 0.01,
    tip: tr("Wei\xDFabgleich: positiv w\xE4rmer (orange), negativ k\xFChler (blau).", "White balance: positive warms toward orange, negative cools toward blue.")
  },
  {
    field: "tint",
    label: tr("T\xF6nung", "Tint"),
    short: "Tint",
    min: -1,
    max: 1,
    step: 0.01,
    tip: tr("Wei\xDFabgleich, andere Achse: positiv gr\xFCner, negativ magenta.", "White balance on the other axis: positive toward green, negative toward magenta.")
  },
  {
    field: "saturation",
    label: tr("S\xE4ttigung", "Saturation"),
    short: tr("S\xE4tt", "Sat"),
    min: 0,
    max: 2,
    step: 0.01,
    tip: tr("Farbintensit\xE4t: 1 unver\xE4ndert, 0 schwarzwei\xDF.", "Colour intensity: 1 is unchanged, 0 is black and white.")
  },
  {
    field: "contrast",
    label: tr("Kontrast", "Contrast"),
    short: tr("Kontr", "Contr"),
    min: 0.5,
    max: 1.5,
    step: 0.01,
    tip: tr("Abstand zwischen Dunkel und Hell: 1 unver\xE4ndert, kleiner flacher.", "Spread between darks and lights: 1 is unchanged, lower is flatter.")
  },
  {
    field: "shadows",
    label: tr("Tiefen", "Shadows"),
    short: tr("Tief", "Shad"),
    min: -0.5,
    max: 0.5,
    step: 0.01,
    tip: tr("Hebt (positiv) oder dr\xFCckt (negativ) nur die dunkelsten T\xF6ne.", "Lift (positive) or crush (negative) just the darkest tones.")
  },
  {
    field: "midtones",
    label: tr("Mitten", "Midtones"),
    short: tr("Mitt", "Mid"),
    min: -0.5,
    max: 0.5,
    step: 0.01,
    tip: tr("Hellt die Mitten (Haut, Kleidung) auf oder ab, ohne Schwarz und Wei\xDF.", "Brighten or darken the middle tones (skin, clothing) without moving black or white.")
  },
  {
    field: "highlights",
    label: tr("Lichter", "Highlights"),
    short: tr("Licht", "High"),
    min: -0.5,
    max: 0.5,
    step: 0.01,
    tip: tr("Hellt die hellsten T\xF6ne auf oder zieht sie herunter.", "Brighten or pull down just the brightest tones.")
  }
];
function sanitizeGrade(value) {
  const g = { ...NEUTRAL };
  if (!value || typeof value !== "object") return g;
  for (const s of GRADE_FIELDS) {
    const v = Number(value[s.field]);
    if (Number.isFinite(v)) g[s.field] = clamp(v, s.min, s.max);
  }
  return g;
}
function isNeutral(g) {
  return GRADE_FIELDS.every((s) => Math.abs(g[s.field] - NEUTRAL[s.field]) < 1e-4);
}
var LUMA = [0.2126, 0.7152, 0.0722];
var MINKOWSKI = 4;
var CLIPPED = 0.98;
var DARK = 0.03;
var WB_T = [0.35, 0, -0.4];
var WB_G = [-0.08, 0.22, -0.04];
var clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
var toLinear = (v) => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
function imageStats(rgba, scale = 1, premultiplied = false, step = 1) {
  let w = 0, wIllum = 0, logY = 0, chroma = 0, y1 = 0, y2 = 0;
  const p = [0, 0, 0];
  for (let i = 0; i + 3 < rgba.length; i += 4 * step) {
    const a = premultiplied ? rgba[i + 3] * scale : 1;
    if (a < 0.5) continue;
    const k = scale / a;
    const c = [clamp(rgba[i] * k, 0, 1), clamp(rgba[i + 1] * k, 0, 1), clamp(rgba[i + 2] * k, 0, 1)];
    const yd = c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2];
    chroma += a * (Math.abs(c[0] - yd) + Math.abs(c[1] - yd) + Math.abs(c[2] - yd)) / 3;
    y1 += a * yd;
    y2 += a * yd * yd;
    const l = [toLinear(c[0]), toLinear(c[1]), toLinear(c[2])];
    logY += a * Math.log(l[0] * LUMA[0] + l[1] * LUMA[1] + l[2] * LUMA[2] + 1e-3);
    w += a;
    if (Math.max(c[0], c[1], c[2]) >= CLIPPED || yd < DARK) continue;
    for (let j = 0; j < 3; j++) p[j] += a * l[j] ** MINKOWSKI;
    wIllum += a;
  }
  if (w <= 0) return null;
  const e = wIllum > 0 ? p.map((v) => (v / wIllum) ** (1 / MINKOWSKI)) : [1, 1, 1];
  const ey = e[0] * LUMA[0] + e[1] * LUMA[1] + e[2] * LUMA[2];
  const mean = y1 / w;
  return {
    illum: ey > 1e-6 ? e.map((v) => v / ey) : [1, 1, 1],
    logY: logY / w,
    chroma: chroma / w,
    spread: Math.sqrt(Math.max(0, y2 / w - mean * mean))
  };
}
function meanStats(all) {
  const n = all.length;
  const avg = (f) => all.reduce((a, s) => a + f(s), 0) / n;
  const illum = [0, 1, 2].map((c) => Math.exp(avg((s) => Math.log(Math.max(s.illum[c], 1e-4)))));
  const ey = illum[0] * LUMA[0] + illum[1] * LUMA[1] + illum[2] * LUMA[2];
  return { illum: illum.map((v) => v / ey), logY: avg((s) => s.logY), chroma: avg((s) => s.chroma), spread: avg((s) => s.spread) };
}
function solveBalance(a, b) {
  let t = 0, g = 0;
  const f = (t2, g2) => {
    const k = gradeGain({ ...NEUTRAL, temperature: t2, tint: g2 });
    return [Math.log(k[0] / k[1]) - a, Math.log(k[2] / k[1]) - b];
  };
  for (let it = 0; it < 8; it++) {
    const r = f(t, g);
    const h = 1e-4;
    const rt = f(t + h, g), rg = f(t, g + h);
    const j00 = (rt[0] - r[0]) / h, j10 = (rt[1] - r[1]) / h, j01 = (rg[0] - r[0]) / h, j11 = (rg[1] - r[1]) / h;
    const det = j00 * j11 - j01 * j10;
    if (Math.abs(det) < 1e-9) break;
    t = clamp(t - (j11 * r[0] - j01 * r[1]) / det, -1, 1);
    g = clamp(g - (j00 * r[1] - j10 * r[0]) / det, -1, 1);
  }
  return [t, g];
}
function estimateGrade(room, content, strength) {
  const ref = content ?? { illum: [1, 1, 1], logY: room.logY, chroma: room.chroma, spread: room.spread };
  const lr = room.illum.map((v) => Math.log(Math.max(v, 1e-4)));
  const [temperature, tint] = solveBalance(lr[0] - lr[1], lr[2] - lr[1]);
  const s = clamp(strength, 0, 1);
  const ratio = (num, den, lo, hi) => 1 + (clamp(den > 1e-4 ? Math.sqrt(num / den) : 1, lo, hi) - 1) * s;
  return {
    ...NEUTRAL,
    exposure: s * clamp((room.logY - ref.logY) / Math.LN2 / 2, -1, 1),
    temperature: s * temperature,
    tint: s * tint,
    saturation: ratio(room.chroma, ref.chroma, 0.75, 1.15),
    contrast: ratio(room.spread, ref.spread, 0.85, 1.1)
  };
}
function gradeGain(g) {
  const e = 2 ** g.exposure;
  const t = clamp(g.temperature, -1, 1);
  const n = clamp(g.tint, -1, 1);
  return [
    e * (1 + t * WB_T[0]) * (1 + n * WB_G[0]),
    e * (1 + n * WB_G[1]),
    e * (1 + t * WB_T[2]) * (1 + n * WB_G[2])
  ];
}
function gradeUniform(g, out = new Float32Array(12)) {
  out.fill(0);
  if (!g) return out;
  out.set(gradeGain(g), 0);
  out[3] = g.contrast;
  out[4] = g.saturation;
  out[5] = 1;
  out[8] = g.shadows;
  out[9] = g.midtones;
  out[10] = g.highlights;
  return out;
}
function formatField(s, v) {
  const d = s.step < 0.05 ? 2 : 1;
  return (s.min < 0 && v > 0 ? "+" : "") + v.toFixed(d);
}
function describeGrade(g) {
  return GRADE_FIELDS.filter((s) => Math.abs(g[s.field] - NEUTRAL[s.field]) >= 5e-3).map((s) => `${s.short} ${formatField(s, g[s.field])}`).join(" \xB7 ") || tr("neutral", "neutral");
}
var SETTLE_MS = 900;
var FRAMES = 6;
var FRAME_GAP_MS = 150;
var SAMPLE_W = 160;
var MIN_SPREAD = 0.02;
function withTimeout(p, ms, what, onLate) {
  return new Promise((resolve, reject) => {
    let done = false;
    const timer = setTimeout(() => {
      done = true;
      reject(new Error(`${what} timed out after ${ms}ms`));
    }, ms);
    p.then(
      (v) => {
        if (done) {
          onLate?.(v);
          return;
        }
        done = true;
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    );
  });
}
var stopStream = (s) => s.getTracks().forEach((t) => t.stop());
var sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function captureRoom(timeoutMs = 5e3) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("no camera access (getUserMedia)");
  const stream = await withTimeout(
    navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 640 } }, audio: false }),
    timeoutMs,
    "getUserMedia",
    stopStream
  );
  try {
    const track = stream.getVideoTracks()[0];
    console.info("MR: room camera", track?.label, JSON.stringify(track?.getSettings?.() ?? {}));
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await withTimeout(video.play(), timeoutMs, "camera play");
    await sleep(SETTLE_MS);
    const vw = video.videoWidth || 4, vh = video.videoHeight || 3;
    const w = SAMPLE_W, h = Math.max(1, Math.round(SAMPLE_W * vh / vw));
    const ctx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("no 2d context");
    const frames = [];
    for (let i = 0; i < FRAMES; i++) {
      if (i) await sleep(FRAME_GAP_MS);
      ctx.drawImage(video, 0, 0, w, h);
      const s = imageStats(ctx.getImageData(0, 0, w, h).data, 1 / 255);
      if (s && s.spread >= MIN_SPREAD) frames.push(s);
    }
    if (!frames.length) throw new Error("camera frame is blank");
    return meanStats(frames);
  } finally {
    stopStream(stream);
  }
}

// src/renderer.ts
var renderCfg = { near: 0.01 };
var isQ = (g) => g.q === true;
var ST = GPUBufferUsage.STORAGE;
var CD = GPUBufferUsage.COPY_DST;
var CS = GPUBufferUsage.COPY_SRC;
var PASSES = ["eval", "project", "sort", "raster", "blit"];
function nextPow2(n) {
  let p = 512;
  while (p < n) p *= 2;
  return p;
}
var CAM_OCCL_WGSL = (
  /* wgsl */
  `
struct VIn {
  @location(0) xy: vec2f,
  @location(1) uv: vec2f,
  @location(2) col: vec4f,
  @location(3) z: f32,
};
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) z: f32,
  @location(1) uv: vec2f,
  @location(2) col: vec4f,
};
struct PaintU { size: vec2f, pad: vec2f };
@group(0) @binding(0) var<uniform> U: PaintU;
@group(0) @binding(1) var bodyZ: texture_2d<f32>;
@group(0) @binding(2) var img: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@vertex
fn vs(v: VIn) -> VOut {
  var o: VOut;
  o.pos = vec4f(v.xy.x / U.size.x * 2.0 - 1.0, 1.0 - v.xy.y / U.size.y * 2.0, 0.0, 1.0);
  o.z = v.z;
  o.uv = v.uv;
  o.col = v.col;
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  let sz = textureLoad(bodyZ, vec2i(v.pos.xy), 0).r;
  if (sz + 0.05 < v.z) { discard; }
  let texel = textureSampleLevel(img, samp, v.uv, 0.0);
  let a = v.col.a * texel.a;
  if (a < 0.01) { discard; }
  return vec4f(texel.rgb * v.col.rgb, a);
}
`
);
var Renderer = class {
  device;
  format;
  bytes = /* @__PURE__ */ new Map();
  // live GPU allocations by name
  peakBytes = 0;
  evalPipe;
  evalqPipe;
  projPipe;
  dummySn;
  dummySnq;
  sortGlobal;
  sortLocal;
  rasterPipe;
  shadowPipe;
  composePipe;
  blitPipe;
  diffPipe = null;
  evalU;
  camU;
  shadowU;
  indirect;
  bgU;
  diffU = null;
  /** Planar floor shadow (projected soft discs). Off by default; cheap enough for Quest with stride > 1. */
  shadow = false;
  shadowFloorY = 0;
  shadowStrength = 0.45;
  shadowMaxRad = 0.12;
  shadowFadeH = 1.6;
  shadowSoft = 0.55;
  shadowStride = 1;
  /** Direction toward the light (content space). */
  shadowLight = [0.55, 1, 0.35];
  shadowScratch = new ArrayBuffer(48);
  shadowF = new Float32Array(this.shadowScratch);
  shadowI = new Uint32Array(this.shadowScratch);
  /** Max-blended shadow discs; composed into litTarget α before the bg blit. */
  shadowTarget = null;
  litTarget = null;
  shadowOffTex;
  sortU = null;
  sortSteps = [];
  cap = 0;
  n2 = 0;
  world;
  splat2d;
  keys;
  vals;
  target = null;
  targetSize = [0, 0];
  xrTargets = [];
  xrBlitPipe = null;
  xrBlitFormat = null;
  xrBlitU = null;
  xrBlitData = new Float32Array(24);
  /** P40 (Quest): sort only every `sortEvery`-th frame of the same section; frames in between reuse the last order
   *  (the order of slowly moving splats barely changes between 72 Hz frames). 1 = sort every frame (default). */
  sortEvery = 1;
  sortAge = 0;
  sortSrc = null;
  sortN = -1;
  reuseOrder(g) {
    const n = isQ(g) ? g.n : g.ps.n;
    if (this.sortEvery > 1 && this.sortSrc === g && this.sortN === n && this.sortAge < this.sortEvery - 1) {
      this.sortAge++;
      return true;
    }
    this.sortSrc = g;
    this.sortN = n;
    this.sortAge = 0;
    return false;
  }
  evalScratch = new ArrayBuffer(16);
  evalF = new Float32Array(this.evalScratch);
  evalI = new Uint32Array(this.evalScratch);
  camScratch = new ArrayBuffer(128);
  camF = new Float32Array(this.camScratch);
  camI = new Uint32Array(this.camScratch);
  indirectReset = new Uint32Array([4, 0, 0, 0]);
  views = /* @__PURE__ */ new WeakMap();
  blitBGs = /* @__PURE__ */ new Map();
  composeBGs = /* @__PURE__ */ new Map();
  occLayout;
  occFloat;
  occUint;
  occOff;
  occSlots = [];
  ts = null;
  lastTimings = null;
  timingsLog = [];
  visibleBuf;
  constructor(device, format, timestamps) {
    this.device = device;
    this.format = format;
    const mod = (code) => device.createShaderModule({ code });
    const cp = (code, entryPoint = "main") => device.createComputePipeline({ layout: "auto", compute: { module: mod(code), entryPoint } });
    this.evalPipe = cp(EVAL_WGSL);
    this.evalqPipe = cp(EVALQ_WGSL);
    this.projPipe = cp(PROJECT_WGSL);
    const sm = mod(SORT_WGSL);
    this.sortGlobal = device.createComputePipeline({ layout: "auto", compute: { module: sm, entryPoint: "global_step" } });
    this.sortLocal = device.createComputePipeline({ layout: "auto", compute: { module: sm, entryPoint: "local_sort" } });
    const rm = mod(RASTER_WGSL);
    const blend = { srcFactor: "one-minus-dst-alpha", dstFactor: "one", operation: "add" };
    const minBlend = { srcFactor: "one", dstFactor: "one", operation: "min" };
    const V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
    const rasterBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: V, buffer: { type: "uniform" } },
      { binding: 1, visibility: V, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: V, buffer: { type: "read-only-storage" } }
    ] });
    this.occLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: F, buffer: { type: "uniform" } },
      { binding: 1, visibility: F, texture: { sampleType: "unfilterable-float" } },
      { binding: 2, visibility: F, texture: { sampleType: "uint" } }
    ] });
    const tex1 = (format2) => device.createTexture({ size: [1, 1], format: format2, usage: GPUTextureUsage.TEXTURE_BINDING });
    this.occFloat = tex1("r32float");
    this.occUint = tex1("r16uint");
    this.occOff = this.occSlot(-1).bg;
    this.rasterPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [rasterBGL, this.occLayout] }),
      vertex: { module: rm, entryPoint: "vs" },
      fragment: { module: rm, entryPoint: "fs", targets: [
        { format: "rgba16float", blend: { color: blend, alpha: blend } },
        { format: "rgba16float", blend: { color: minBlend, alpha: minBlend } }
      ] },
      primitive: { topology: "triangle-strip" }
    });
    this.rasterPipeColor = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [rasterBGL, this.occLayout] }),
      vertex: { module: rm, entryPoint: "vs" },
      fragment: { module: rm, entryPoint: "fs_color", targets: [
        { format: "rgba16float", blend: { color: blend, alpha: blend } }
      ] },
      primitive: { topology: "triangle-strip" }
    });
    const smSh = mod(SHADOW_RASTER_WGSL);
    const shadowBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: V, buffer: { type: "uniform" } },
      { binding: 1, visibility: V, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: V | F, buffer: { type: "uniform" } }
    ] });
    const shadowBlend = { srcFactor: "one", dstFactor: "one", operation: "max" };
    this.shadowPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [shadowBGL, this.occLayout] }),
      vertex: { module: smSh, entryPoint: "vs" },
      fragment: { module: smSh, entryPoint: "fs", targets: [{ format: "rgba16float", blend: { color: shadowBlend, alpha: shadowBlend } }] },
      primitive: { topology: "triangle-strip" }
    });
    this.shadowOffTex = device.createTexture({
      size: [1, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });
    device.queue.writeTexture({ texture: this.shadowOffTex }, new Uint16Array(4), { bytesPerRow: 8 }, [1, 1]);
    const cm = mod(SHADOW_COMPOSE_WGSL);
    this.composePipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: cm, entryPoint: "vs" },
      fragment: { module: cm, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
      primitive: { topology: "triangle-list" }
    });
    const bm = mod(BLIT_WGSL);
    this.blitPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: bm, entryPoint: "vs" },
      fragment: { module: bm, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" }
    });
    this.bgU = this.buf("bgU", 64, GPUBufferUsage.UNIFORM | CD);
    this.evalU = this.buf("evalU", 16, GPUBufferUsage.UNIFORM | CD);
    this.camU = this.buf("camU", 128, GPUBufferUsage.UNIFORM | CD);
    this.shadowU = this.buf("shadowU", 48, GPUBufferUsage.UNIFORM | CD);
    this.indirect = this.buf("indirect", 16, GPUBufferUsage.INDIRECT | ST | CD | CS);
    this.visibleBuf = this.buf("visibleRead", 16, GPUBufferUsage.MAP_READ | CD);
    this.dummySn = this.buf("dummySn", 48, ST);
    this.dummySnq = this.buf("dummySnq", 360, ST);
    if (timestamps) {
      const n = PASSES.length * 2;
      this.ts = {
        set: device.createQuerySet({ type: "timestamp", count: n }),
        resolve: this.buf("tsResolve", n * 8, GPUBufferUsage.QUERY_RESOLVE | CS),
        read: [0, 1, 2].map((i) => this.buf(`tsRead${i}`, n * 8, GPUBufferUsage.MAP_READ | CD)),
        busy: [false, false, false],
        next: 0
      };
    }
  }
  /** desktop canvas background, linear 0..1 (the splats are composited over it) */
  setBackground(r, g, b) {
    this.device.queue.writeBuffer(this.bgU, 0, new Float32Array([r, g, b, 1]));
  }
  /** grade applied by the desktop and XR blits; null turns it off */
  setGrade(g) {
    const u = gradeUniform(g);
    this.xrBlitData.set(u, 4);
    this.device.queue.writeBuffer(this.bgU, 16, u);
  }
  /** P40: in-headset readout (fps, splats in thousands) drawn by the XR eye blit; px = digit height, 0 turns it off */
  setHud(fps, kSplats, px) {
    const d = this.xrBlitData;
    d[16] = fps;
    d[17] = kSplats;
    d[18] = px > 0 ? 1 : 0;
    d[19] = px;
  }
  /**
   * P40 bench/test hook: run the XR eye blit (HUD on) from a cleared w×h float target into an rgba8unorm texture and
   * return how many pixels are lit (green/amber/red) and dark (HUD background) — proves the WGSL compiles and draws.
   */
  async debugXrBlit(w, h, fps, kSplats, px) {
    const dev = this.device;
    const src = dev.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const dst = dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const { pipe, uni } = this.xrBlitPipeline("rgba8unorm");
    const keep = this.xrBlitData.slice();
    this.setHud(fps, kSplats, px);
    this.xrBlitData[0] = 0;
    this.xrBlitData[1] = 0;
    this.xrBlitData[2] = 1;
    this.xrBlitData[3] = 0;
    this.xrBlitData[22] = 0;
    dev.queue.writeBuffer(uni, 0, this.xrBlitData);
    this.xrBlitData.set(keep);
    const enc = dev.createCommandEncoder();
    const clr = enc.beginRenderPass({ colorAttachments: [{ view: src.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }] });
    clr.end();
    const bp = enc.beginRenderPass({ colorAttachments: [{ view: dst.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }] });
    bp.setPipeline(pipe);
    bp.setBindGroup(0, this.blitBindGroup(pipe, src, uni));
    bp.draw(3);
    bp.end();
    const bpr = Math.ceil(w * 4 / 256) * 256;
    const rb = dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: dst }, { buffer: rb, bytesPerRow: bpr }, [w, h]);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const px8 = new Uint8Array(rb.getMappedRange());
    let lit = 0, dark = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * bpr + x * 4;
      if (px8[i + 3] < 200) continue;
      if (Math.max(px8[i], px8[i + 1]) > 100) lit++;
      else dark++;
    }
    rb.unmap();
    rb.destroy();
    src.destroy();
    dst.destroy();
    return { lit, dark, total: w * h };
  }
  buf(name, size, usage) {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label: name });
    this.bytes.set(name, (this.bytes.get(name) ?? 0) + b.size);
    this.peakBytes = Math.max(this.peakBytes, this.gpuBytes());
    return b;
  }
  doomed = [];
  reapQueued = false;
  free(name, b) {
    this.bytes.set(name, (this.bytes.get(name) ?? 0) - b.size);
    this.doomed.push(b);
    if (this.reapQueued) return;
    this.reapQueued = true;
    this.device.queue.onSubmittedWorkDone().then(() => {
      this.reapQueued = false;
      const batch = this.doomed;
      this.doomed = [];
      for (const x of batch) x.destroy();
    });
  }
  gpuBytes() {
    let s = 0;
    for (const v of this.bytes.values()) s += v;
    if (this.target) s += this.targetSize[0] * this.targetSize[1] * 8;
    if (this.bodyDepths) for (const t of this.bodyDepths.values()) s += t.width * t.height * 8;
    for (const t of this.xrTargets) s += t.width * t.height * 8;
    return s;
  }
  /** Allocates the section's buffers; the data goes up through {@link pump}. */
  upload(ps) {
    const pending = [];
    const mk = (nm, data) => {
      const b = this.buf(`sec_${nm}`, data.byteLength, ST | CD);
      pending.push({ buf: b, data, done: 0 });
      return b;
    };
    const snq = new Float32Array(90);
    snq.set(ps.snOffset, 0);
    snq.set(ps.snScale, 45);
    const g = { ps, hdr: mk("hdr", ps.hdr), knots: mk("knots", ps.knots), sn: mk("sn", ps.sn), snq: mk("snq", snq), bytes: 0, pending };
    g.bytes = g.hdr.size + g.knots.size + g.sn.size + g.snq.size;
    this.ensureCapacity(ps.n);
    return g;
  }
  /** Issue up to `budget` bytes of a section's pending writes (Infinity finishes it). */
  pump(g, budget) {
    const q = this.device.queue;
    while (g.pending?.length && budget > 0) {
      const w = g.pending[0];
      const len = Math.min(w.data.byteLength - w.done, budget >= 2 ** 30 ? Infinity : Math.max(4, budget & ~3));
      q.writeBuffer(w.buf, w.done, w.data.buffer, w.data.byteOffset + w.done, len);
      w.done += len;
      budget -= len;
      if (w.done >= w.data.byteLength) g.pending.shift();
    }
    if (g.pending && !g.pending.length) g.pending = null;
  }
  release(g) {
    if (isQ(g)) {
      this.free("q_hdr", g.hdr);
      this.free("q_data", g.data);
      if (g.p16) this.free("q_p16", g.p16);
      if (g.sn) this.free("q_sn", g.sn);
      if (g.snq) this.free("q_snq", g.snq);
      return;
    }
    this.free("sec_hdr", g.hdr);
    this.free("sec_knots", g.knots);
    this.free("sec_sn", g.sn);
    this.free("sec_snq", g.snq);
  }
  /** upload a .splq frame into `slot` (reused when it is large enough and has the same optional blocks, else replaced). */
  uploadQ(fr, slot = null) {
    const opt = fr.flags & (SPLQ_FLAG_SN | SPLQ_FLAG_P16);
    if (!slot || slot.cap < fr.n || slot.flags !== opt) {
      if (slot) this.release(slot);
      const cap = Math.ceil(fr.n * 1.05) + 256;
      slot = {
        q: true,
        n: fr.n,
        cap,
        hdr: this.buf("q_hdr", SPLQ_HDR, GPUBufferUsage.UNIFORM | CD),
        data: this.buf("q_data", cap * 16, ST | CD),
        p16: fr.p16 ? this.buf("q_p16", cap * 8, ST | CD) : null,
        sn: fr.sn ? this.buf("q_sn", cap * 48, ST | CD) : null,
        snq: fr.sn ? this.buf("q_snq", 360, ST | CD) : null,
        flags: opt,
        bytes: 0
      };
      slot.bytes = slot.hdr.size + slot.data.size + (slot.p16?.size ?? 0) + (slot.sn?.size ?? 0) + (slot.snq?.size ?? 0);
    }
    slot.n = fr.n;
    this.device.queue.writeBuffer(slot.hdr, 0, fr.header.buffer, fr.header.byteOffset, SPLQ_HDR);
    this.device.queue.writeBuffer(slot.data, 0, fr.data.buffer, fr.data.byteOffset, fr.n * 16);
    if (fr.p16 && slot.p16) this.device.queue.writeBuffer(slot.p16, 0, fr.p16.buffer, fr.p16.byteOffset, fr.n * 8);
    const snw = splqSnWords(fr);
    if (snw && slot.sn && slot.snq) {
      this.device.queue.writeBuffer(slot.sn, 0, snw.buffer, 0, fr.n * 48);
      this.device.queue.writeBuffer(slot.snq, 0, fr.snq.buffer, fr.snq.byteOffset, 360);
    }
    this.ensureCapacity(fr.n);
    return slot;
  }
  ensureCapacity(n) {
    if (n <= this.cap) return;
    if (this.cap) {
      this.free("world", this.world);
      this.free("splat2d", this.splat2d);
      this.free("keys", this.keys);
      this.free("vals", this.vals);
    }
    this.cap = n;
    this.n2 = nextPow2(n);
    this.world = this.buf("world", n * 64, ST | CS);
    this.splat2d = this.buf("splat2d", n * 48, ST);
    this.keys = this.buf("keys", this.n2 * 4, ST | CS);
    this.vals = this.buf("vals", this.n2 * 4, ST | CS);
    const steps = [{ k: 512, j: 0, mode: 0 }];
    for (let k = 1024; k <= this.n2; k *= 2) {
      for (let j = k / 2; j >= 512; j /= 2) steps.push({ k, j, mode: 2 });
      steps.push({ k, j: 0, mode: 1 });
    }
    this.sortSteps = steps;
    if (this.sortU) this.free("sortU", this.sortU);
    this.sortU = this.buf("sortU", steps.length * 256, GPUBufferUsage.UNIFORM | CD);
    const u = new Uint32Array(steps.length * 64);
    steps.forEach((s, i) => {
      u[i * 64] = s.k;
      u[i * 64 + 1] = s.j;
      u[i * 64 + 2] = s.mode === 2 ? 0 : s.mode;
    });
    this.device.queue.writeBuffer(this.sortU, 0, u);
    this.sortBGs = steps.map((s, idx) => {
      const pipe = s.mode === 2 ? this.sortGlobal : this.sortLocal;
      return this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.sortU, offset: idx * 256, size: 16 } },
        { binding: 1, resource: { buffer: this.keys, size: this.n2 * 4 } },
        { binding: 2, resource: { buffer: this.vals, size: this.n2 * 4 } }
      ] });
    });
    this.capVersion++;
  }
  sortBGs = [];
  capVersion = 0;
  bindGroups(g) {
    if (g.bgs && g.bgs.ver === this.capVersion) return g.bgs;
    const dev = this.device;
    const q = isQ(g);
    g.bgs = {
      ver: this.capVersion,
      eval: q ? dev.createBindGroup({ layout: this.evalqPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: g.hdr } },
        { binding: 1, resource: { buffer: g.data } },
        { binding: 2, resource: { buffer: this.world } },
        { binding: 3, resource: { buffer: g.p16 ?? this.dummySn } }
      ] }) : dev.createBindGroup({ layout: this.evalPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.evalU } },
        { binding: 1, resource: { buffer: g.hdr } },
        { binding: 2, resource: { buffer: g.knots } },
        { binding: 3, resource: { buffer: this.world } }
      ] }),
      proj: dev.createBindGroup({ layout: this.projPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.camU } },
        { binding: 1, resource: { buffer: this.world } },
        { binding: 2, resource: { buffer: q ? g.sn ?? this.dummySn : g.sn } },
        { binding: 3, resource: { buffer: q ? g.snq ?? this.dummySnq : g.snq } },
        { binding: 4, resource: { buffer: this.splat2d } },
        { binding: 5, resource: { buffer: this.keys } },
        { binding: 6, resource: { buffer: this.vals } },
        { binding: 7, resource: { buffer: this.indirect } }
      ] }),
      raster: dev.createBindGroup({ layout: this.rasterPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.camU } },
        { binding: 1, resource: { buffer: this.splat2d } },
        { binding: 2, resource: { buffer: this.vals, size: this.n2 * 4 } }
      ] }),
      shadow: dev.createBindGroup({ layout: this.shadowPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.camU } },
        { binding: 1, resource: { buffer: this.world } },
        { binding: 2, resource: { buffer: this.shadowU } }
      ] })
    };
    return g.bgs;
  }
  sortDispatches() {
    return this.sortSteps.length;
  }
  ensureTarget(w, h) {
    if (this.target && this.targetSize[0] === w && this.targetSize[1] === h) return;
    this.target?.destroy();
    this.litTarget?.destroy();
    this.blitBGs.clear();
    this.composeBGs.clear();
    this.target = this.device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    });
    this.litTarget = this.device.createTexture({
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      label: "lit-compose"
    });
    this.targetSize = [w, h];
    this.ensureShadowTarget(w, h);
    this.peakBytes = Math.max(this.peakBytes, this.gpuBytes());
  }
  ensureShadowTarget(w, h) {
    if (this.shadowTarget && this.shadowTarget.width === w && this.shadowTarget.height === h) return;
    this.shadowTarget?.destroy();
    this.composeBGs.clear();
    this.shadowTarget = this.device.createTexture({
      size: [Math.max(1, w), Math.max(1, h)],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "floor-shadow"
    });
  }
  ensureBodyDepth(w, h) {
    if (!this.bodyDepths) this.bodyDepths = /* @__PURE__ */ new Map();
    const key = w + "x" + h;
    let tex = this.bodyDepths.get(key);
    if (!tex) {
      tex = this.device.createTexture({
        size: [Math.max(1, w), Math.max(1, h)],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        label: "body-depth-" + key
      });
      this.bodyDepths.set(key, tex);
    }
    this.bodyDepth = tex;
    return tex;
  }
  captureBodyDepth(enc, src) {
    const dev = this.device;
    const sw = src.width, sh = src.height;
    const cols = Math.max(1, Math.ceil(sw / 8));
    const dh = Math.max(1, Math.ceil(sh / 8));
    const bytesPerRow = Math.ceil(cols * 8 / 256) * 256;
    if (!this.depthRead || this.depthRead.sw !== sw || this.depthRead.sh !== sh) {
      if (this.depthRead?.busy.some(Boolean)) return;
      this.depthRead?.tex.destroy();
      this.depthRead?.bufs.forEach((b) => b.destroy());
      if (!this.depthPipe) {
        const sm = dev.createShaderModule({ code: `
@group(0) @binding(0) var src: texture_2d<f32>;
struct D { srcSize: vec2f, tile: vec2f };
@group(0) @binding(1) var<uniform> U: D;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let tile = max(U.tile, vec2f(1.0));
  let origin = vec2i(vec2f(p.xy) * tile);
  let sx = max(1, i32(tile.x));
  let sy = max(1, i32(tile.y));
  let step = max(1, min(sx, sy) / 4);
  var m = 1e3;
  let lim = vec2i(U.srcSize);
  for (var y = 0; y < sy; y += step) {
    for (var x = 0; x < sx; x += step) {
      let s = origin + vec2i(x, y);
      if (s.x >= 0 && s.y >= 0 && s.x < lim.x && s.y < lim.y) { m = min(m, textureLoad(src, s, 0).r); }
    }
  }
  return vec4f(m, 0.0, 0.0, 1.0);
}
` });
        this.depthPipe = dev.createRenderPipeline({
          layout: "auto",
          vertex: { module: sm, entryPoint: "vs" },
          fragment: { module: sm, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
          primitive: { topology: "triangle-list" }
        });
        this.depthU = this.buf("depthU", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      }
      const tex = dev.createTexture({
        size: [cols, dh], format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        label: "body-depth-small"
      });
      const bufs = [0, 1].map(() => dev.createBuffer({
        size: bytesPerRow * dh,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      }));
      this.depthRead = { tex, bufs, busy: [false, false], slot: 0, sw, sh, cols, dh, bytesPerRow };
      this.depthBG = null;
    }
    const r = this.depthRead;
    dev.queue.writeBuffer(this.depthU, 0, new Float32Array([sw, sh, sw / r.cols, sh / r.dh]));
    if (!this.depthBG || this.depthBG.src !== src) {
      this.depthBG = {
        src,
        bg: dev.createBindGroup({
          layout: this.depthPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.viewOf(src) },
            { binding: 1, resource: { buffer: this.depthU } }
          ]
        })
      };
    }
    const slot = r.slot;
    if (r.busy[slot]) return;
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: r.tex.createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 1e3, g: 0, b: 0, a: 1 }
      }]
    });
    pass.setPipeline(this.depthPipe);
    pass.setBindGroup(0, this.depthBG.bg);
    pass.draw(3);
    pass.end();
    r.slot = 1 - slot;
    enc.copyTextureToBuffer({ texture: r.tex }, { buffer: r.bufs[slot], bytesPerRow: r.bytesPerRow }, [r.cols, r.dh]);
    r.busy[slot] = true;
    this.depthPendingMap = {
      buf: r.bufs[slot], slot, r,
      w: r.cols, h: r.dh, srcW: r.sw, srcH: r.sh, stride16: r.bytesPerRow / 2
    };
  }
  flushDepthRead() {
    const job = this.depthPendingMap;
    if (!job) return;
    this.depthPendingMap = null;
    const { buf, slot, r, w, h, srcW, srcH, stride16 } = job;
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const u16 = new Uint16Array(buf.getMappedRange().slice(0));
      buf.unmap();
      r.busy[slot] = false;
      if (this.depthRead !== r) return;
      const data = new Float32Array(w * h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) data[y * w + x] = halfToFloat(u16[y * stride16 + x * 4]);
      }
      this.splatDepth = { w, h, srcW, srcH, stride: w, data };
    }).catch(() => {
      r.busy[slot] = false;
    });
  }
  paintCams() {
  }
  /** encode one frame at global frame time t (fractional) into `out` (canvas texture) and submit.
   *  opts.diff: show |render − ref| (difference view) instead of the render. */
  frame(g, t, cam, out, opts = {}) {
    const dev = this.device;
    const ts = opts.timing && this.ts && !this.ts.busy[this.ts.next] ? this.ts : null;
    const enc = dev.createCommandEncoder();
    this.record(
      enc,
      g,
      t,
      cam,
      out ? { kind: "canvas", texture: out, diff: opts.diff ?? null } : null,
      true,
      ts,
      null,
      this.reuseOrder(g) ? 2 : 0
    );
    this.camMesh = null;
    let slot = -1;
    if (ts) {
      slot = ts.next;
      ts.next = (ts.next + 1) % ts.read.length;
      ts.busy[slot] = true;
      const cnt = out ? PASSES.length * 2 : (PASSES.length - 1) * 2;
      enc.resolveQuerySet(ts.set, 0, cnt, ts.resolve, 0);
      enc.copyBufferToBuffer(ts.resolve, 0, ts.read[slot], 0, cnt * 8);
    }
    const readVis = opts.readVisible && this.visibleBuf.mapState === "unmapped";
    if (readVis) enc.copyBufferToBuffer(this.indirect, 0, this.visibleBuf, 0, 16);
    dev.queue.submit([enc.finish()]);
    this.flushDepthRead();
    if (ts && slot >= 0) {
      const rb = ts.read[slot], hasBlit = !!out;
      rb.mapAsync(GPUMapMode.READ).then(() => {
        const v = new BigUint64Array(rb.getMappedRange().slice(0));
        rb.unmap();
        ts.busy[slot] = false;
        const r = {};
        PASSES.forEach((nm, p) => {
          if (p < 4 || hasBlit) r[nm] = Number(v[2 * p + 1] - v[2 * p]) / 1e6;
        });
        r.gpu_total = Number(v[hasBlit ? 9 : 7] - v[0]) / 1e6;
        this.lastTimings = r;
        this.timingsLog.push(r);
      }).catch(() => {
        ts.busy[slot] = false;
      });
    }
    if (readVis) {
      this.visibleBuf.mapAsync(GPUMapMode.READ).then(() => {
        this.lastVisible = new Uint32Array(this.visibleBuf.getMappedRange().slice(0))[1];
        this.visibleBuf.unmap();
      }).catch(() => {
      });
    }
  }
  /** Both eyes of one immersive frame. Eval runs on the first eye; each eye is its own submit so its camera uniform is visible to that submit only. */
  frameEyes(g, t, eyes, overlay2) {
    const shared = eyes.length > 1;
    const reuse = this.reuseOrder(g);
    eyes.forEach((eye, i) => {
      const target = this.xrTarget(i, eye.cam.width, eye.cam.height);
      const enc = this.device.createCommandEncoder();
      const order = reuse ? 2 : shared ? i === 0 ? 1 : 2 : 0;
      this.record(enc, g, t, eye.cam, { kind: "xr", eye }, i === 0, null, target, order, this.occFor(i, eye.occ));
      this.device.queue.submit([enc.finish()]);
      if (!overlay2) return;
      const over = this.device.createCommandEncoder();
      overlay2(over, eye, i);
      this.device.queue.submit([over.finish()]);
    });
  }
  xrTarget(eye, w, h) {
    const cur = this.xrTargets[eye];
    if (cur && cur.width === w && cur.height === h) return cur;
    cur?.destroy();
    const t = this.device.createTexture({
      size: [Math.max(1, w), Math.max(1, h)],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: `xr-eye-${eye}`
    });
    this.xrTargets[eye] = t;
    this.peakBytes = Math.max(this.peakBytes, this.gpuBytes());
    return t;
  }
  occGroup(buf, tex, kind, desc) {
    return this.device.createBindGroup({ layout: this.occLayout, entries: [
      { binding: 0, resource: { buffer: buf } },
      { binding: 1, resource: (tex && kind === "float" ? tex : this.occFloat).createView(tex && kind === "float" ? desc : void 0) },
      { binding: 2, resource: (tex && kind === "uint" ? tex : this.occUint).createView(tex && kind === "uint" ? desc : void 0) }
    ] });
  }
  /** slot -1: the always-off group (zero uniform) */
  occSlot(i) {
    if (i >= 0 && this.occSlots[i]) return this.occSlots[i];
    const buf = this.buf("occU", 128, GPUBufferUsage.UNIFORM | CD);
    const s = { buf, bg: this.occGroup(buf, null, "float"), tex: null, layer: -1 };
    if (i >= 0) this.occSlots[i] = s;
    return s;
  }
  /** Real-world depth test for eye `i` (one uniform per eye, so both survive their submits). */
  occFor(i, occ) {
    if (!occ || occ.kind !== "float" && occ.kind !== "uint") return this.occOff;
    const s = this.occSlot(i);
    this.device.queue.writeBuffer(s.buf, 0, occ.data);
    const layer = occ.view?.baseArrayLayer ?? 0;
    if (s.tex !== occ.texture || s.layer !== layer) {
      s.tex = occ.texture;
      s.layer = layer;
      s.bg = this.occGroup(s.buf, occ.texture, occ.kind, occ.view);
    }
    return s.bg;
  }
  xrBlitPipeline(format) {
    if (!this.xrBlitU) this.xrBlitU = this.buf("xrBlitU", 96, GPUBufferUsage.UNIFORM | CD);
    if (!this.xrBlitPipe || this.xrBlitFormat !== format) {
      const bm = this.device.createShaderModule({ code: BLIT_XR_WGSL });
      this.xrBlitPipe = this.device.createRenderPipeline({
        label: "xr-blit",
        layout: "auto",
        vertex: { module: bm, entryPoint: "vs" },
        fragment: { module: bm, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" }
      });
      this.xrBlitFormat = format;
    }
    return { pipe: this.xrBlitPipe, uni: this.xrBlitU };
  }
  record(enc, g, t, cam, dest, doEval, ts, raster, order = 0, occ = this.occOff) {
    const dev = this.device, q = isQ(g), n = q ? g.n : g.ps.n;
    this.ensureCapacity(n);
    if (!raster) this.ensureTarget(cam.width, cam.height);
    const target = raster ?? this.target;
    if (doEval && !q) {
      const ps = g.ps, ef = this.evalF;
      ef[0] = t;
      ef[1] = ps.g0;
      ef[2] = Math.max(ps.nframes - 1, 1);
      this.evalI[3] = n;
      dev.queue.writeBuffer(this.evalU, 0, this.evalScratch);
    }
    const cf = this.camF, ci = this.camI;
    cf.set(cam.view, 0);
    cf[16] = cam.campos[0];
    cf[17] = cam.campos[1];
    cf[18] = cam.campos[2];
    cf[20] = cam.fx;
    cf[21] = cam.fy;
    cf[22] = cam.cx;
    cf[23] = cam.cy;
    cf[24] = cam.width;
    cf[25] = cam.height;
    cf[26] = renderCfg.near;
    cf[27] = 0;
    ci[28] = n;
    ci[29] = this.n2;
    ci[30] = q && !g.sn ? 1 : 0;
    ci[31] = order;
    dev.queue.writeBuffer(this.camU, 0, this.camScratch);
    if (order !== 2) dev.queue.writeBuffer(this.indirect, 0, this.indirectReset);
    const tw = (p) => ts ? { querySet: ts.set, beginningOfPassWriteIndex: 2 * p, endOfPassWriteIndex: 2 * p + 1 } : void 0;
    const bgs = this.bindGroups(g);
    if (doEval) {
      const pass2 = enc.beginComputePass({ timestampWrites: tw(0) });
      pass2.setPipeline(q ? this.evalqPipe : this.evalPipe);
      pass2.setBindGroup(0, bgs.eval);
      pass2.dispatchWorkgroups(Math.ceil(n / 256));
      pass2.end();
    }
    let pass = enc.beginComputePass({ timestampWrites: tw(1) });
    pass.setPipeline(this.projPipe);
    pass.setBindGroup(0, bgs.proj);
    pass.dispatchWorkgroups(Math.ceil(this.n2 / 256));
    pass.end();
    if (order !== 2) {
      pass = enc.beginComputePass({ timestampWrites: tw(2) });
      this.sortSteps.forEach((s, idx) => {
        pass.setPipeline(s.mode === 2 ? this.sortGlobal : this.sortLocal);
        pass.setBindGroup(0, this.sortBGs[idx]);
        pass.dispatchWorkgroups(s.mode === 2 ? Math.ceil(this.n2 / 2 / 256) : this.n2 / 512);
      });
      pass.end();
    }
    const targetView = this.viewOf(target);
    const bodyDepth = this.bodyDepthOn && dest?.kind === "canvas" ? this.ensureBodyDepth(target.width, target.height) : null;
    const color0 = {
      view: targetView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store"
    };
    const rp = enc.beginRenderPass({
      colorAttachments: bodyDepth ? [color0, {
        view: this.viewOf(bodyDepth),
        clearValue: { r: 1000, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store"
      }] : [color0],
      timestampWrites: tw(3)
    });
    rp.setPipeline(bodyDepth ? this.rasterPipe : this.rasterPipeColor);
    rp.setBindGroup(0, bgs.raster);
    rp.setBindGroup(1, occ);
    rp.drawIndirect(this.indirect, 0);
    rp.end();
    if (bodyDepth && (this.depthTick = (this.depthTick + 1) | 0) % 2 === 0) this.captureBodyDepth(enc, bodyDepth);
    if (this.shadow) {
      this.ensureShadowTarget(target.width, target.height);
      const shTex = this.shadowTarget;
      const L = this.shadowLight;
      const len = Math.hypot(L[0], L[1], L[2]) || 1;
      const sf = this.shadowF, si = this.shadowI;
      sf[0] = L[0] / len;
      sf[1] = L[1] / len;
      sf[2] = L[2] / len;
      sf[3] = this.shadowFloorY;
      sf[4] = this.shadowStrength;
      sf[5] = this.shadowMaxRad;
      sf[6] = this.shadowFadeH;
      sf[7] = this.shadowSoft;
      si[8] = n;
      si[9] = Math.max(1, this.shadowStride | 0);
      si[10] = 0;
      si[11] = 0;
      dev.queue.writeBuffer(this.shadowU, 0, this.shadowScratch);
      const sp = enc.beginRenderPass({
        colorAttachments: [{
          view: this.viewOf(shTex),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store"
        }]
      });
      sp.setPipeline(this.shadowPipe);
      sp.setBindGroup(0, bgs.shadow);
      sp.setBindGroup(1, occ);
      sp.draw(4, Math.ceil(n / Math.max(1, this.shadowStride)));
      sp.end();
    }
    if (dest?.kind === "canvas") {
      const bp = enc.beginRenderPass({
        colorAttachments: [{ view: dest.texture.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
        timestampWrites: tw(4)
      });
      const df = dest.diff;
      if (df) {
        if (!this.diffPipe) {
          const dm = dev.createShaderModule({ code: DIFF_WGSL });
          this.diffPipe = dev.createRenderPipeline({
            layout: "auto",
            vertex: { module: dm, entryPoint: "vs" },
            fragment: { module: dm, entryPoint: "fs", targets: [{ format: this.format }] },
            primitive: { topology: "triangle-list" }
          });
          this.diffU = this.buf("diffU", 32, GPUBufferUsage.UNIFORM | CD);
        }
        dev.queue.writeBuffer(this.diffU, 0, new Float32Array([df.ox, df.oy, df.dw, df.dh, df.tex.width, df.tex.height, df.gain, 0]));
        bp.setPipeline(this.diffPipe);
        bp.setBindGroup(0, dev.createBindGroup({ layout: this.diffPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: this.viewOf(target) },
          { binding: 1, resource: df.tex.createView() },
          { binding: 2, resource: { buffer: this.diffU } }
        ] }));
      } else {
        bp.setPipeline(this.blitPipe);
        bp.setBindGroup(0, this.blitBindGroup(this.blitPipe, target, this.bgU));
      }
      bp.draw(3);
      bp.end();
      if (bodyDepth && this.camMesh?.length && bodyDepth.width === dest.texture.width && bodyDepth.height === dest.texture.height)
        this.paintCams(enc, dest.texture);
    } else if (dest?.kind === "xr") {
      const eye = dest.eye;
      const { pipe, uni } = this.xrBlitPipeline(eye.format);
      this.xrBlitData[0] = eye.viewport.x;
      this.xrBlitData[1] = eye.viewport.y;
      this.xrBlitData[2] = eye.keepAlpha ? 1 : 0;
      this.xrBlitData[3] = eye.encodeSrgb ? 1 : 0;
      this.xrBlitData[20] = target.width / eye.viewport.width;
      this.xrBlitData[21] = target.height / eye.viewport.height;
      this.xrBlitData[22] = target.width !== eye.viewport.width || target.height !== eye.viewport.height ? 1 : 0;
      dev.queue.writeBuffer(uni, 0, this.xrBlitData);
      const bp = enc.beginRenderPass({
        colorAttachments: [{
          view: eye.color,
          clearValue: { r: 0, g: 0, b: 0, a: eye.keepAlpha ? 0 : 1 },
          loadOp: eye.clear ? "clear" : "load",
          storeOp: "store"
        }]
      });
      bp.setPipeline(pipe);
      bp.setBindGroup(0, this.blitBindGroup(pipe, target, uni));
      const full = eye.viewport.x === 0 && eye.viewport.y === 0 && eye.viewport.width === eye.attachmentWidth && eye.viewport.height === eye.attachmentHeight;
      if (!full) bp.setViewport(eye.viewport.x, eye.viewport.y, eye.viewport.width, eye.viewport.height, 0, 1);
      bp.draw(3);
      bp.end();
    }
  }
  lastVisible = 0;
  viewOf(tex) {
    let v = this.views.get(tex);
    if (!v) {
      v = tex.createView();
      this.views.set(tex, v);
    }
    return v;
  }
  composeBindGroup(color, shadow) {
    const key = `${color.width}x${color.height}`;
    const hit = this.composeBGs.get(key);
    if (hit) return hit;
    const bg = this.device.createBindGroup({
      layout: this.composePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.viewOf(color) },
        { binding: 1, resource: this.viewOf(shadow) }
      ]
    });
    this.composeBGs.set(key, bg);
    return bg;
  }
  blitBindGroup(pipe, src, uni) {
    const sh = this.shadow && this.shadowTarget ? this.shadowTarget : this.shadowOffTex;
    const hit = this.blitBGs.get(src);
    if (hit && hit.pipe === pipe && hit.sh === sh) return hit.bg;
    const entries = [
      { binding: 0, resource: this.viewOf(src) },
      { binding: 2, resource: this.viewOf(sh) }
    ];
    if (uni) entries.push({ binding: 1, resource: { buffer: uni } });
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    this.blitBGs.set(src, { pipe, bg, sh });
    return bg;
  }
  /** float render target -> 8-bit RGB (same clamp as the blit), for PSNR checks. */
  async readTarget() {
    const { w, h, bpr, src } = await this.readTargetHalf();
    const rgb = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      const v = half(src[y * bpr / 2 + x * 4 + c]);
      rgb[(y * w + x) * 3 + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
    return { w, h, rgb };
  }
  /** every `step`-th pixel of the last desktop frame, premultiplied RGBA; null before the first frame. */
  async readTargetRGBA(step = 1) {
    if (!this.target) return null;
    const { w, h, bpr, src } = await this.readTargetHalf();
    const out = new Float32Array(Math.ceil(w * h / step) * 4);
    let o = 0;
    for (let i = 0; i < w * h; i += step) {
      const base = Math.floor(i / w) * (bpr / 2) + i % w * 4;
      for (let c = 0; c < 4; c++) out[o++] = half(src[base + c]);
    }
    return out;
  }
  async readTargetHalf() {
    const [w, h] = this.targetSize;
    const bpr = Math.ceil(w * 8 / 256) * 256;
    const rb = this.device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.target }, { buffer: rb, bytesPerRow: bpr }, [w, h]);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const src = new Uint16Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return { w, h, bpr, src };
  }
  /** evaluated world attributes of splats `idx` (GPU eval pass output of the last frame). */
  async readWorld(idx) {
    const rb = this.device.createBuffer({ size: this.world.size, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.world, 0, rb, 0, this.world.size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return idx.map((i) => all.slice(i * 16, i * 16 + 16));
  }
  /** P40 (WB-1341): the whole evaluated world buffer of the last frame (16 floats per splat), raw bytes. */
  async readWorldAll() {
    const rb = this.device.createBuffer({ size: this.world.size, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.world, 0, rb, 0, this.world.size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const all = rb.getMappedRange().slice(0);
    rb.destroy();
    return all;
  }
};
function half(h) {
  const s = h & 32768 ? -1 : 1, e = h >> 10 & 31, f = h & 1023;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

// src/bspline.ts
var BSPLINE_DEGREE = 3;
function bsplineWeights(K, uIn) {
  const p = Math.min(BSPLINE_DEGREE, K - 1);
  if (p <= 0) return [1];
  const nInt = K - p - 1;
  const kn = new Array(p + 1).fill(0);
  for (let i = 0; i < nInt; i++) kn.push((i + 1) / (nInt + 1));
  for (let i = 0; i < p + 1; i++) kn.push(1);
  const u = Math.min(Math.max(uIn, 0), 1);
  const m = kn.length - 1;
  let N = [];
  for (let i = 0; i < m; i++) N.push(kn[i] <= u && u < kn[i + 1] || u === 1 && kn[i] < kn[i + 1] && kn[i + 1] === 1 ? 1 : 0);
  for (let d = 1; d <= p; d++) {
    const M = [];
    for (let i = 0; i < m - d; i++) {
      const a = kn[i + d] > kn[i] ? (u - kn[i]) / (kn[i + d] - kn[i]) * N[i] : 0;
      const b = kn[i + d + 1] > kn[i + 1] ? (kn[i + d + 1] - u) / (kn[i + d + 1] - kn[i + 1]) * N[i + 1] : 0;
      M.push(a + b);
    }
    N = M;
  }
  return N.slice(0, K);
}

// src/pack.ts
var GROUP_DIMS = [3, 4, 3, 1, 3];
var HDR_U32 = 8;
function evalSplat(ps, i, t) {
  const h = i * HDR_U32;
  const fi = Math.floor(t);
  const alive = ps.hdr[h] <= fi && fi < ps.hdr[h + 1];
  const span = Math.max(ps.nframes - 1, 1);
  const vals = [];
  for (let g = 0; g < 5; g++) {
    const K = ps.hdr[h + 2] >>> 6 * g & 63, d = GROUP_DIMS[g], off = ps.hdr[h + 3 + g];
    const v = [];
    if (ps.hdr[h + 2] >>> 31) {
      const u = Math.min(Math.max((t - ps.g0) / span, 0), 1);
      const wts = bsplineWeights(K, u);
      for (let c = 0; c < d; c++) {
        let acc = 0;
        for (let j = 0; j < K; j++) if (wts[j] !== 0) acc += wts[j] * ps.knots[off + j * d + c];
        v.push(acc);
      }
    } else {
      let j = 0, w = 0;
      if (K > 1) {
        const tau = Math.min(Math.max((t - ps.g0) * (K - 1) / span, 0), K - 1);
        j = Math.min(Math.floor(tau), K - 2);
        w = tau - j;
      }
      for (let c = 0; c < d; c++) {
        const a = ps.knots[off + j * d + c];
        const b = K > 1 ? ps.knots[off + (j + 1) * d + c] : a;
        v.push(a + (b - a) * w);
      }
    }
    if (g === 1) {
      const nrm = Math.hypot(v[0], v[1], v[2], v[3]) || 1;
      for (let c = 0; c < 4; c++) v[c] /= nrm;
    }
    vals.push(v);
  }
  return { alive, vals };
}

// src/clips.ts
function sectionsReady(nsec, gpu) {
  if (nsec <= 0) return false;
  for (let i = 0; i < nsec; i++) {
    const g = gpu[i];
    if (!g || g.pending) return false;
  }
  return true;
}
function stepName(list, current, dir) {
  if (list.length < 2) return null;
  const i = list.indexOf(current);
  if (i < 0) return null;
  return list[(i + dir + list.length) % list.length];
}
function retainClips(current, next) {
  const keep = /* @__PURE__ */ new Set([current]);
  if (next) keep.add(next);
  return keep;
}

// src/hands.ts
var JOINTS = [
  "wrist",
  "thumb-metacarpal",
  "thumb-phalanx-proximal",
  "thumb-phalanx-distal",
  "thumb-tip",
  "index-finger-metacarpal",
  "index-finger-phalanx-proximal",
  "index-finger-phalanx-intermediate",
  "index-finger-phalanx-distal",
  "index-finger-tip",
  "middle-finger-metacarpal",
  "middle-finger-phalanx-proximal",
  "middle-finger-phalanx-intermediate",
  "middle-finger-phalanx-distal",
  "middle-finger-tip",
  "ring-finger-metacarpal",
  "ring-finger-phalanx-proximal",
  "ring-finger-phalanx-intermediate",
  "ring-finger-phalanx-distal",
  "ring-finger-tip",
  "pinky-finger-metacarpal",
  "pinky-finger-phalanx-proximal",
  "pinky-finger-phalanx-intermediate",
  "pinky-finger-phalanx-distal",
  "pinky-finger-tip"
];
var JOINT_INDEX = new Map(JOINTS.map((j, i) => [j, i]));
var HAND_INFLATE = 2e-3;
var FOREARM = { length: 0.32, overlap: 0.03, width: 0.036, thickness: 0.03 };
function parseHandMesh(buf) {
  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
  const binStart = 20 + jsonLen + 8;
  const accessor = (i, Ctor, comps) => {
    const a = json.accessors[i];
    const view = json.bufferViews[a.bufferView];
    const offset = binStart + (view.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const bytes = a.count * comps * Ctor.BYTES_PER_ELEMENT;
    return new Ctor(buf.slice(offset, offset + bytes), 0, a.count * comps);
  };
  const prim = json.meshes[0].primitives[0];
  const skin = json.skins[0];
  const jointMap = new Uint8Array(skin.joints.map((node) => {
    const idx = JOINT_INDEX.get(json.nodes[node].name);
    if (idx === void 0) throw new Error(`hand joint ${json.nodes[node].name} not in WebXR set`);
    return idx;
  }));
  return {
    position: accessor(prim.attributes.POSITION, Float32Array, 3),
    normal: accessor(prim.attributes.NORMAL, Float32Array, 3),
    joints: accessor(prim.attributes.JOINTS_0, Uint8Array, 4),
    weights: accessor(prim.attributes.WEIGHTS_0, Float32Array, 4),
    indices: accessor(prim.indices, Uint16Array, 1),
    jointMap,
    inverseBind: accessor(skin.inverseBindMatrices, Float32Array, 16)
  };
}
async function loadHandMesh(url) {
  return parseHandMesh(await (await fetch(url)).arrayBuffer());
}
function mulAt(out, o, a, ao, b, bo) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[ao + k * 4 + r] * b[bo + c * 4 + k];
    out[o + c * 4 + r] = s;
  }
}
function writeSkin(out, mesh, poses, hand) {
  for (let s = 0; s < mesh.jointMap.length; s++) {
    mulAt(out, (hand * 25 + s) * 16, poses, mesh.jointMap[s] * 16, mesh.inverseBind, s * 16);
  }
}
function forearmMatrix(poses, out, o = 0) {
  const at = (j, k) => poses[j * 16 + 12 + k];
  const wrist = [at(0, 0), at(0, 1), at(0, 2)];
  const a = [0, 1, 2].map((k) => wrist[k] - at(11, k));
  const al = Math.hypot(a[0], a[1], a[2]);
  if (al < 1e-4) return false;
  for (let k = 0; k < 3; k++) a[k] /= al;
  const s = [0, 1, 2].map((k) => at(20, k) - at(5, k));
  const sa = s[0] * a[0] + s[1] * a[1] + s[2] * a[2];
  for (let k = 0; k < 3; k++) s[k] -= sa * a[k];
  const sl = Math.hypot(s[0], s[1], s[2]);
  if (sl < 1e-4) return false;
  for (let k = 0; k < 3; k++) s[k] /= sl;
  const n = [a[1] * s[2] - a[2] * s[1], a[2] * s[0] - a[0] * s[2], a[0] * s[1] - a[1] * s[0]];
  const c = FOREARM.length / 2 - FOREARM.overlap;
  out.fill(0, o, o + 16);
  for (let k = 0; k < 3; k++) {
    out[o + k] = s[k] * 2 * FOREARM.width;
    out[o + 4 + k] = n[k] * 2 * FOREARM.thickness;
    out[o + 8 + k] = a[k] * FOREARM.length;
    out[o + 12 + k] = wrist[k] + a[k] * c;
  }
  out[o + 15] = 1;
  return true;
}
function forearmSphere() {
  const lat = 8, lon = 12, r = 0.5;
  const pos = [];
  for (let i = 0; i <= lat; i++) {
    const th = i / lat * Math.PI;
    const st = Math.sin(th), ct = Math.cos(th);
    for (let j = 0; j <= lon; j++) {
      const ph = j / lon * Math.PI * 2;
      pos.push(st * Math.cos(ph) * r, ct * r, st * Math.sin(ph) * r);
    }
  }
  const row = lon + 1;
  const tri = [];
  for (let i = 0; i < lat; i++) for (let j = 0; j < lon; j++) {
    const a = i * row + j;
    const p = (k) => {
      tri.push(pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2]);
    };
    p(a);
    p(a + row);
    p(a + 1);
    p(a + 1);
    p(a + row);
    p(a + row + 1);
  }
  const position = new Float32Array(new ArrayBuffer(tri.length * 4));
  position.set(tri);
  return { position, count: position.length / 3 };
}

// src/occlusion.ts
var KIND_CODE = { float: 1, uint: 2, depth: 3 };
function depthKind(format) {
  if (format === "r32float") return "float";
  if (format === "r16uint") return "uint";
  if (format.startsWith("depth")) return "depth";
  return null;
}
var OccBinder = class {
  layout;
  off;
  device;
  blank;
  slots = [];
  constructor(device) {
    this.device = device;
    const F = GPUShaderStage.FRAGMENT;
    this.layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: F, buffer: { type: "uniform" } },
      { binding: 1, visibility: F, texture: { sampleType: "unfilterable-float" } },
      { binding: 2, visibility: F, texture: { sampleType: "uint" } },
      { binding: 3, visibility: F, texture: { sampleType: "depth" } }
    ] });
    const tex1 = (format) => device.createTexture({ size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING });
    this.blank = { float: tex1("r32float"), uint: tex1("r16uint"), depth: tex1("depth16unorm") };
    this.off = this.slot(-1).bg;
  }
  /** slot -1: the always-off group (zero uniform) */
  slot(i) {
    if (i >= 0 && this.slots[i]) return this.slots[i];
    const buf = this.device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "occU" });
    const s = { buf, bg: this.group(buf, null, "float"), tex: null, layer: -1 };
    if (i >= 0) this.slots[i] = s;
    return s;
  }
  group(buf, tex, kind, desc) {
    const view = (k) => tex && kind === k ? tex.createView(k === "depth" ? { ...desc, aspect: "depth-only" } : desc) : this.blank[k].createView();
    return this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: buf } },
      { binding: 1, resource: view("float") },
      { binding: 2, resource: view("uint") },
      { binding: 3, resource: view("depth") }
    ] });
  }
  /** Real-world depth test for eye `i` (one uniform per eye, so both survive their submits). */
  forEye(i, data, tex, kind, desc) {
    if (!data || !tex) return this.off;
    const s = this.slot(i);
    this.device.queue.writeBuffer(s.buf, 0, data);
    const layer = desc?.baseArrayLayer ?? 0;
    if (s.tex !== tex || s.layer !== layer) {
      s.tex = tex;
      s.layer = layer;
      s.bg = this.group(s.buf, tex, kind, desc);
    }
    return s.bg;
  }
};

// src/desk.ts
var TRAVEL = 0.11;
var HALF_W = 0.013;
var BODY_H = 0.026;
var KNOB_H = 0.01;
var FLOOR = 0.012;
var GRAB_Y = 0.09;
var BTN = 0.016;
var TAKE_ROW = 0.05;
var SPACING = 0.054;
var TILT = 30 * Math.PI / 180;
var MIN_SCALE = 0.4;
var MAX_SCALE = 3;
var TRAY = [0.1, 0.105, 0.115, 1];
var METAL = [0.22, 0.23, 0.25, 1];
var SLOT = [0.05, 0.052, 0.06, 1];
var MARK = [0.93, 0.94, 0.96, 1];
var HELD = [0.55, 0.92, 1, 1];
var ON = [0.2, 0.5, 0.62, 1];
var HINT_COL = [0.62, 0.64, 0.68, 1];
var HINT_H = 0.014;
var HINT = [
  () => tr("Regler kneifen und schieben \xB7 Brett kneifen zum Tragen", "Pinch a fader to slide it \xB7 pinch the tray to carry it"),
  () => tr(
    "Zwei H\xE4nde: am Brett Gr\xF6\xDFe \xE4ndern, in der Luft Szene bewegen, drehen, skalieren",
    "Two hands: on the tray to resize it, in the air to move, turn and scale the scene"
  )
];
var CAROUSEL_HINT = [
  () => tr("Pause kneifen, um diesen Clip zu halten \xB7 Pfeile wechseln", "Pinch pause to hold this clip \xB7 arrows change take")
];
var ACCENT = {
  exposure: [0.95, 0.9, 0.62, 1],
  temperature: [0.98, 0.6, 0.25, 1],
  tint: [0.85, 0.4, 0.85, 1],
  saturation: [0.35, 0.85, 0.45, 1],
  contrast: [0.75, 0.78, 0.82, 1],
  shadows: [0.4, 0.5, 0.95, 1],
  midtones: [0.55, 0.75, 0.95, 1],
  highlights: [0.95, 0.95, 0.8, 1]
};
var SHADOW_FADE_MIN = 0.4;
var SHADOW_FADE_MAX = 3.5;
var SHADOW_FADE_STEP = 0.05;
var SHADOW_FADE_DEFAULT = 1.8;
var SHADOW_ACCENT = [0.22, 0.22, 0.24, 1];
var SHADOW_SHORT = () => tr("Schatten", "Shadow");
var glyphs = /* @__PURE__ */ new Map();
function head(ctx, x, y, angle, size) {
  ctx.beginPath();
  ctx.moveTo(x + Math.cos(angle) * size, y + Math.sin(angle) * size);
  ctx.lineTo(x + Math.cos(angle + 2.4) * size, y + Math.sin(angle + 2.4) * size);
  ctx.lineTo(x + Math.cos(angle - 2.4) * size, y + Math.sin(angle - 2.4) * size);
  ctx.closePath();
  ctx.fill();
}
function arcArrow(ctx, cx, cy, r, a0, a1, ccw) {
  ctx.beginPath();
  ctx.arc(cx, cy, r, a0, a1, ccw);
  ctx.stroke();
  head(ctx, cx + Math.cos(a1) * r, cy + Math.sin(a1) * r, a1 + (ccw ? -Math.PI / 2 : Math.PI / 2), 7);
}
function drawIcon(ctx, icon) {
  const c = 20;
  ctx.lineWidth = 3.5;
  ctx.lineCap = "round";
  switch (icon) {
    case "auto": {
      const star = (x, y, r) => {
        ctx.beginPath();
        for (let i = 0; i < 8; i++) {
          const a = i * Math.PI / 4 - Math.PI / 2, k = i % 2 ? r * 0.3 : r;
          ctx.lineTo(x + Math.cos(a) * k, y + Math.sin(a) * k);
        }
        ctx.closePath();
        ctx.fill();
      };
      star(17, 22, 14);
      star(31, 9, 6);
      break;
    }
    case "reset":
      arcArrow(ctx, c, c, 12, Math.PI * 0.05, Math.PI * 1.55, false);
      break;
    case "rotate":
      arcArrow(ctx, c, c, 13, Math.PI * 1.1, Math.PI * 1.9, false);
      arcArrow(ctx, c, c, 13, Math.PI * 0.1, Math.PI * 0.9, false);
      ctx.beginPath();
      ctx.arc(c, c, 3, 0, Math.PI * 2);
      ctx.fill();
      break;
    case "move":
      ctx.beginPath();
      ctx.moveTo(c, 8);
      ctx.lineTo(c, 32);
      ctx.moveTo(8, c);
      ctx.lineTo(32, c);
      ctx.stroke();
      for (let i = 0; i < 4; i++) {
        const a = i * Math.PI / 2;
        head(ctx, c + Math.cos(a) * 15, c + Math.sin(a) * 15, a, 7);
      }
      break;
    case "scale":
      ctx.beginPath();
      ctx.moveTo(12, 28);
      ctx.lineTo(28, 12);
      ctx.moveTo(6, 22);
      ctx.lineTo(6, 34);
      ctx.lineTo(18, 34);
      ctx.moveTo(22, 6);
      ctx.lineTo(34, 6);
      ctx.lineTo(34, 18);
      ctx.stroke();
      head(ctx, 10, 30, Math.PI * 3 / 4, 6);
      head(ctx, 30, 10, -Math.PI / 4, 6);
      break;
    case "prev":
    case "next": {
      const s = icon === "next" ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(c + s * 11, c);
      ctx.lineTo(c - s * 9, c - 12);
      ctx.lineTo(c - s * 9, c + 12);
      ctx.closePath();
      ctx.fill();
      break;
    }
    case "hold":
      ctx.fillRect(13, 10, 5, 20);
      ctx.fillRect(22, 10, 5, 20);
      break;
    case "resume":
      ctx.beginPath();
      ctx.moveTo(13, 8);
      ctx.lineTo(13, 32);
      ctx.lineTo(30, 20);
      ctx.closePath();
      ctx.fill();
      break;
  }
}
function glyph(label) {
  const cached = glyphs.get(label);
  if (cached) return cached;
  if (glyphs.size > 600) glyphs.clear();
  const h = 40;
  const icon = label.startsWith("icon:") ? label.slice(5) : null;
  const font = `700 ${h * 0.62}px ui-sans-serif, system-ui, sans-serif`;
  let w = 40;
  if (typeof OffscreenCanvas === "undefined") return { w, h, runs: [] };
  if (!icon) {
    const m = new OffscreenCanvas(1, 1).getContext("2d");
    if (m) {
      m.font = font;
      w = Math.max(40, Math.ceil(m.measureText(label).width + h * 0.3));
    }
  }
  const g = { w, h, runs: [] };
  glyphs.set(label, g);
  const ctx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true });
  if (!ctx) return g;
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#fff";
  ctx.lineJoin = "round";
  if (icon) drawIcon(ctx, icon);
  else {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = font;
    ctx.lineWidth = h * 0.06;
    ctx.strokeText(label, w / 2, h / 2 + 1);
    ctx.fillText(label, w / 2, h / 2 + 1);
  }
  const data = ctx.getImageData(0, 0, w, h).data;
  for (let y = 0; y < h; y++) {
    let x = 0;
    while (x < w) {
      if (data[(y * w + x) * 4 + 3] < 64) {
        x++;
        continue;
      }
      let rw = 1;
      while (x + rw < w && data[(y * w + x + rw) * 4 + 3] >= 64) rw++;
      g.runs.push({ x, y, w: rw });
      x += rw;
    }
  }
  return g;
}
var GradeDesk = class {
  model = new Float32Array(16);
  dirty = true;
  onChange = null;
  /** Floor-shadow height falloff (metres); live while the desk is up. */
  onShadowChange = null;
  onButton = null;
  lit = /* @__PURE__ */ new Set();
  placed = false;
  specs;
  extra;
  buttons;
  z0 = TRAVEL / 2;
  down = /* @__PURE__ */ new Map();
  owner = null;
  /** hands holding the tray, at their last point; two of them scale the desk */
  carriers = /* @__PURE__ */ new Map();
  origin = [0, 0, 0];
  axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  back = [0, 0, 1];
  scale = 1;
  /** viewer position in room space; a carried desk turns about world up to face it */
  head = null;
  /** name of the playing take; null hides the take row (prev / name / next) */
  takeName = null;
  /** colour faders only while holding a clip; carousel still shows prev / name / next / hold */
  editingOn = false;
  scratch = new Float32Array(1 << 16);
  count = 0;
  /** Height (m) above the floor that still casts; longer = softer, more extended shadow. */
  shadowFade = SHADOW_FADE_DEFAULT;
  grade;
  constructor(grade) {
    this.grade = grade;
    const n = GRADE_FIELDS.length;
    this.specs = GRADE_FIELDS.map((s, i) => ({ field: s.field, x: (i - (n - 1) / 2) * SPACING, min: s.min, max: s.max }));
    this.extra = {
      id: "shadowfade",
      x: this.specs[n - 1].x + SPACING,
      min: SHADOW_FADE_MIN,
      max: SHADOW_FADE_MAX,
      step: SHADOW_FADE_STEP,
      def: SHADOW_FADE_DEFAULT
    };
    const zb = this.z0 + 0.06;
    this.buttons = [
      { id: "auto", x: -0.17, z: zb, label: "icon:auto" },
      { id: "reset", x: -0.085, z: zb, label: "icon:reset" },
      { id: "move", x: 0.015, z: zb, label: "icon:move" },
      { id: "rotate", x: 0.1, z: zb, label: "icon:rotate" },
      { id: "scale", x: 0.185, z: zb, label: "icon:scale" },
      { id: "prev", x: -0.17, z: zb + TAKE_ROW, label: "icon:prev" },
      { id: "hold", x: 0, z: zb, label: "icon:hold" },
      { id: "next", x: 0.185, z: zb + TAKE_ROW, label: "icon:next" }
    ];
  }
  /** hands carrying the tray; two means they are resizing it */
  get carrierCount() {
    return this.carriers.size;
  }
  hintZ() {
    return this.z0 + 0.06 + (this.takeName !== null ? TAKE_ROW : 0) + BTN + 0.018;
  }
  get take() {
    return this.takeName;
  }
  set take(name) {
    if (name === this.takeName) return;
    this.takeName = name;
    this.dirty = true;
  }
  get editing() {
    return this.editingOn;
  }
  set editing(on) {
    if (on === this.editingOn) return;
    this.editingOn = on;
    this.dirty = true;
  }
  shown(id) {
    if (id === "prev" || id === "next" || id === "hold") return this.takeName !== null;
    if (id === "move" || id === "rotate" || id === "scale") return false;
    return this.editingOn;
  }
  /** In reach, right of the line of sight and below the head, facing the user. */
  place(head2, forward) {
    const fl = Math.hypot(forward[0], forward[2]) || 1;
    const f = [forward[0] / fl, 0, forward[2] / fl];
    const r = [-f[2], 0, f[0]];
    this.origin = [head2[0] + f[0] * 0.4 + r[0] * 0.2, head2[1] - 0.4, head2[2] + f[2] * 0.4 + r[2] * 0.2];
    this.setAxes(r, [-f[0], 0, -f[2]]);
    this.placed = true;
  }
  setAxes(r, back) {
    const c = Math.cos(TILT), s = Math.sin(TILT);
    const ay = [back[0] * s, c, back[2] * s];
    const az = [back[0] * c, -s, back[2] * c];
    this.back = back;
    this.axes = [r, ay, az];
    this.updateModel();
  }
  updateModel() {
    const m = this.model;
    const [ax, ay, az] = this.axes;
    const k = this.scale;
    m.set([
      ax[0] * k,
      ax[1] * k,
      ax[2] * k,
      0,
      ay[0] * k,
      ay[1] * k,
      ay[2] * k,
      0,
      az[0] * k,
      az[1] * k,
      az[2] * k,
      0,
      this.origin[0],
      this.origin[1],
      this.origin[2],
      1
    ]);
  }
  toLocal(p) {
    const d = [p[0] - this.origin[0], p[1] - this.origin[1], p[2] - this.origin[2]];
    const [ax, ay, az] = this.axes;
    const k = 1 / this.scale;
    return [(d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2]) * k, (d[0] * ay[0] + d[1] * ay[1] + d[2] * ay[2]) * k, (d[0] * az[0] + d[1] * az[1] + d[2] * az[2]) * k];
  }
  /** Turn about world up through `pivot` until the near edge faces the head. */
  faceHead(pivot) {
    if (!this.head) return;
    const dx = this.head[0] - pivot[0], dz = this.head[2] - pivot[2];
    const l = Math.hypot(dx, dz);
    if (l < 0.15) return;
    const nb = [dx / l, 0, dz / l];
    const ob = this.back;
    const or = [ob[2], 0, -ob[0]];
    const nr = [nb[2], 0, -nb[0]];
    const vx = this.origin[0] - pivot[0], vz = this.origin[2] - pivot[2];
    const a = vx * ob[0] + vz * ob[2], b = vx * or[0] + vz * or[2];
    this.origin = [pivot[0] + a * nb[0] + b * nr[0], this.origin[1], pivot[2] + a * nb[2] + b * nr[2]];
    this.setAxes(nr, nb);
  }
  /** One carrier moved to `p`: one hand translates, two translate with their midpoint and scale by their spread. */
  carry(hand, p) {
    const prev = new Map(this.carriers);
    this.carriers.set(hand, [p[0], p[1], p[2]]);
    if (this.carriers.size === 1) {
      const last = prev.get(hand);
      for (let i = 0; i < 3; i++) this.origin[i] += p[i] - last[i];
      this.faceHead(p);
      return;
    }
    const [a0, b0] = [...prev.values()];
    const [a1, b1] = [...this.carriers.values()];
    const mid0 = [0, 1, 2].map((i) => (a0[i] + b0[i]) / 2);
    const mid1 = [0, 1, 2].map((i) => (a1[i] + b1[i]) / 2);
    const d0 = Math.hypot(a0[0] - b0[0], a0[1] - b0[1], a0[2] - b0[2]);
    const d1 = Math.hypot(a1[0] - b1[0], a1[1] - b1[1], a1[2] - b1[2]);
    let k = d0 > 0.03 && d1 > 0.03 ? d1 / d0 : 1;
    k = Math.min(MAX_SCALE, Math.max(MIN_SCALE, this.scale * k)) / this.scale;
    this.origin = [0, 1, 2].map((i) => mid1[i] + (this.origin[i] - mid0[i]) * k);
    this.scale *= k;
    this.faceHead(mid1);
  }
  tick(dt) {
    for (const [id, v] of [...this.down]) {
      const n = Math.max(0, v - dt * 4);
      if (n === 0) this.down.delete(id);
      else this.down.set(id, n);
      this.dirty = true;
    }
  }
  update(hand, point, down, rising) {
    if (!this.placed) return false;
    if (this.carriers.has(hand)) {
      if (!down || !point) {
        this.carriers.delete(hand);
        return false;
      }
      this.carry(hand, point);
      this.updateModel();
      return true;
    }
    const o = this.owner;
    if (o && o.hand === hand) {
      if (!down || !point) {
        this.owner = null;
        this.dirty = true;
        return false;
      }
      const lz = this.toLocal(point)[2];
      if (o.kind === "slider") this.setFromZ(o.index, lz);
      else if (o.kind === "extra") this.setExtraFromZ(lz);
      return true;
    }
    if (!rising || !point) return false;
    const [x, y, z] = this.toLocal(point);
    if (y < -0.02 || y > GRAB_Y) return false;
    const [lo, hi] = this.trayBounds();
    const onTray = x >= lo[0] && x <= hi[0] && z >= lo[2] && z <= hi[2];
    if (this.carriers.size && onTray) {
      this.carriers.set(hand, [point[0], point[1], point[2]]);
      return true;
    }
    if (o) return false;
    const button = this.buttonAt(x, z);
    if (button) {
      this.owner = { hand, kind: "press" };
      this.down.set(button, 1);
      this.dirty = true;
      this.onButton?.(button);
      return true;
    }
    if (this.hitExtra(x, z)) {
      this.owner = { hand, kind: "extra" };
      this.setExtraFromZ(z);
      this.dirty = true;
      return true;
    }
    const index = this.hit(x, z);
    if (index >= 0) {
      this.owner = { hand, kind: "slider", index };
      this.setFromZ(index, z);
      this.dirty = true;
      return true;
    }
    if (onTray) {
      this.carriers.set(hand, [point[0], point[1], point[2]]);
      return true;
    }
    return false;
  }
  /** Slider value at 0..1 along its slot. */
  norm(s) {
    return Math.min(1, Math.max(0, (this.grade[s.field] - s.min) / (s.max - s.min)));
  }
  setFromZ(index, z) {
    const s = this.specs[index];
    const step = GRADE_FIELDS[index].step;
    const v = Math.min(1, Math.max(0, (this.z0 - z) / TRAVEL));
    let value = Math.round((s.min + v * (s.max - s.min)) / step) * step;
    const neutral = NEUTRAL[s.field];
    if (Math.abs(value - neutral) < (s.max - s.min) * 0.02) value = neutral;
    if (Math.abs(this.grade[s.field] - value) < 1e-6) return;
    this.grade[s.field] = value;
    this.dirty = true;
    this.onChange?.(s.field);
  }
  setExtraFromZ(z) {
    const s = this.extra;
    const t = Math.min(1, Math.max(0, (this.z0 - z) / TRAVEL));
    let value = Math.round((s.min + t * (s.max - s.min)) / s.step) * s.step;
    if (Math.abs(value - s.def) < (s.max - s.min) * 0.02) value = s.def;
    if (Math.abs(this.shadowFade - value) < 1e-6) return;
    this.shadowFade = value;
    this.dirty = true;
    this.onShadowChange?.(value);
  }
  slotHit(x, z, sx) {
    return Math.abs(x - sx) <= 0.032 && z <= this.z0 + 0.03 && z >= this.z0 - TRAVEL - 0.03;
  }
  hit(x, z) {
    if (!this.editingOn) return -1;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < this.specs.length; i++) {
      const s = this.specs[i];
      if (!this.slotHit(x, z, s.x)) continue;
      const d = Math.abs(x - s.x);
      if (d < bestD) {
        best = i;
        bestD = d;
      }
    }
    return best;
  }
  hitExtra(x, z) {
    return this.slotHit(x, z, this.extra.x);
  }
  buttonAt(x, z) {
    for (const b of this.buttons) {
      if (this.shown(b.id) && Math.abs(x - b.x) <= BTN + 0.01 && Math.abs(z - b.z) <= BTN + 0.01) return b.id;
    }
    return null;
  }
  trayBounds() {
    const right = Math.max(this.specs[this.specs.length - 1].x, this.extra.x);
    const left = Math.min(this.specs[0].x, this.extra.x);
    const hx = Math.max(Math.abs(left), Math.abs(right)) + 0.02;
    const front = this.hintZ() + HINT_H * (HINT.length - 0.5) + 0.01;
    return [[-hx, 0, this.z0 - TRAVEL - 0.05], [hx, FLOOR, front]];
  }
  /** Rebuilds the mesh if needed; returns position + colour floats (7 per vertex). */
  mesh() {
    if (this.dirty) {
      const w = new MeshWriter(this.scratch);
      this.build(w);
      this.scratch = w.out;
      this.count = w.n;
      this.dirty = false;
    }
    return this.scratch.subarray(0, this.count);
  }
  build(w) {
    const [lo, hi] = this.trayBounds();
    box(w, 0, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2, hi[0], (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2, TRAY);
    if (this.editingOn) this.specs.forEach((s, i) => this.drawOne(w, s, i));
    this.drawExtra(w);
    if (this.takeName !== null) stamp(w, 75e-4, FLOOR + 1e-3, this.z0 + 0.06 + TAKE_ROW, 0.022, this.takeName);
    for (const b of this.buttons) {
      if (!this.shown(b.id)) continue;
      const press = this.down.get(b.id) ?? 0;
      const top = 0.014 - press * 5e-3;
      const col = press > 0.05 ? HELD : this.lit.has(b.id) ? ON : METAL;
      const label = b.id === "hold" ? this.editingOn ? "icon:resume" : "icon:hold" : b.label;
      box(w, b.x, FLOOR + top / 2, b.z, BTN * 1.6, top / 2, BTN, col);
      stamp(w, b.x, FLOOR + top + 1e-3, b.z, 0.022, label);
    }
    const hz = this.hintZ();
    const hints = this.editingOn ? HINT : CAROUSEL_HINT;
    hints.forEach((line, i) => stamp(w, 0, FLOOR + 1e-3, hz + i * HINT_H, HINT_H * 0.8, line(), HINT_COL));
  }
  drawOne(w, s, index) {
    const z0 = this.z0, z1 = z0 - TRAVEL, mid = (z0 + z1) / 2;
    const v = this.norm(s);
    const knob = z0 - v * TRAVEL;
    const held = this.owner?.kind === "slider" && this.owner.index === index;
    const accent = ACCENT[s.field];
    const top = FLOOR + BODY_H;
    box(w, s.x, FLOOR + BODY_H / 2, mid, HALF_W + 4e-3, BODY_H / 2, TRAVEL / 2 + 8e-3, METAL);
    box(w, s.x, top + 1e-3, mid, 35e-4, 12e-4, TRAVEL / 2, SLOT);
    const nz = z0 - (NEUTRAL[s.field] - s.min) / (s.max - s.min) * TRAVEL;
    box(w, s.x - HALF_W, top + 1e-3, nz, 3e-3, 12e-4, 12e-4, MARK);
    if (Math.abs(knob - nz) > 3e-3) box(w, s.x, top + 16e-4, (nz + knob) / 2, 22e-4, 14e-4, Math.abs(nz - knob) / 2, accent);
    box(w, s.x, top + KNOB_H / 2, knob, 0.011, KNOB_H / 2, 8e-3, held ? HELD : accent);
    const spec = GRADE_FIELDS[index];
    stamp(w, s.x, top + 1e-3, z0 + 0.02, 0.018, spec.short);
    stamp(w, s.x, top + 1e-3, z1 - 0.018, 0.015, formatField(spec, this.grade[s.field]));
  }
  /** Always-on fader: how far up the body still darkens the floor. */
  drawExtra(w) {
    const s = this.extra;
    const z0 = this.z0, z1 = z0 - TRAVEL, mid = (z0 + z1) / 2;
    const v = Math.min(1, Math.max(0, (this.shadowFade - s.min) / (s.max - s.min)));
    const knob = z0 - v * TRAVEL;
    const held = this.owner?.kind === "extra";
    const top = FLOOR + BODY_H;
    box(w, s.x, FLOOR + BODY_H / 2, mid, HALF_W + 4e-3, BODY_H / 2, TRAVEL / 2 + 8e-3, METAL);
    box(w, s.x, top + 1e-3, mid, 35e-4, 12e-4, TRAVEL / 2, SLOT);
    const nz = z0 - (s.def - s.min) / (s.max - s.min) * TRAVEL;
    box(w, s.x - HALF_W, top + 1e-3, nz, 3e-3, 12e-4, 12e-4, MARK);
    if (Math.abs(knob - nz) > 3e-3) box(w, s.x, top + 16e-4, (nz + knob) / 2, 22e-4, 14e-4, Math.abs(nz - knob) / 2, SHADOW_ACCENT);
    box(w, s.x, top + KNOB_H / 2, knob, 0.011, KNOB_H / 2, 8e-3, held ? HELD : SHADOW_ACCENT);
    stamp(w, s.x, top + 1e-3, z0 + 0.02, 0.018, SHADOW_SHORT());
    stamp(w, s.x, top + 1e-3, z1 - 0.018, 0.015, this.shadowFade.toFixed(2));
  }
};
function stamp(w, x, y, z, height, label, color = MARK) {
  stampPlane(w, [x, y, z], [1, 0, 0], [0, 0, 1], height, label, color);
}
function stampPlane(w, c, right, down, height, label, color) {
  const g = glyph(label);
  const cell = height / g.h;
  const at = (u, v) => {
    const a = (u - g.w / 2) * cell, b = (v - g.h / 2) * cell;
    return [c[0] + right[0] * a + down[0] * b, c[1] + right[1] * a + down[1] * b, c[2] + right[2] * a + down[2] * b];
  };
  for (const run of g.runs) {
    const p00 = at(run.x, run.y), p10 = at(run.x + run.w, run.y), p01 = at(run.x, run.y + 1), p11 = at(run.x + run.w, run.y + 1);
    w.tri(p01, p11, p10, color);
    w.tri(p01, p10, p00, color);
  }
}
var MeshWriter = class {
  n = 0;
  out;
  constructor(out) {
    this.out = out;
  }
  vert(p, c) {
    if (this.n + 7 > this.out.length) {
      const grown = new Float32Array(this.out.length * 2);
      grown.set(this.out);
      this.out = grown;
    }
    const o = this.n;
    this.out[o] = p[0];
    this.out[o + 1] = p[1];
    this.out[o + 2] = p[2];
    this.out[o + 3] = c[0];
    this.out[o + 4] = c[1];
    this.out[o + 5] = c[2];
    this.out[o + 6] = c[3] ?? 1;
    this.n += 7;
  }
  tri(a, b, c, color) {
    this.vert(a, color);
    this.vert(b, color);
    this.vert(c, color);
  }
};
function box(w, x, y, z, hx, hy, hz, color) {
  const x0 = x - hx, x1 = x + hx, y0 = y - hy, y1 = y + hy, z0 = z - hz, z1 = z + hz;
  w.tri([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], color);
  w.tri([x0, y0, z1], [x1, y1, z1], [x0, y1, z1], color);
  w.tri([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], color);
  w.tri([x1, y0, z0], [x0, y1, z0], [x1, y1, z0], color);
  w.tri([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], color);
  w.tri([x1, y0, z1], [x1, y1, z0], [x1, y1, z1], color);
  w.tri([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], color);
  w.tri([x0, y0, z0], [x0, y1, z1], [x0, y1, z0], color);
  w.tri([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], color);
  w.tri([x0, y1, z1], [x1, y1, z0], [x0, y1, z0], color);
  w.tri([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], color);
  w.tri([x0, y0, z0], [x1, y0, z1], [x0, y0, z1], color);
}
function mul4(a, b, out) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = s;
  }
  return out;
}
var HAND_WGSL = (
  /* wgsl */
  `
struct HandU { mvp: mat4x4f, fore: array<mat4x4f, 2> };
@group(0) @binding(0) var<uniform> U: HandU;
@group(0) @binding(1) var<uniform> skin: array<mat4x4f, 50>;

struct HandIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) joints: vec4u,
  @location(3) weights: vec4f,
};

@vertex fn vs_hand(in: HandIn, @builtin(instance_index) hand: u32) -> @builtin(position) vec4f {
  let b = hand * 25u;
  let m = skin[b + in.joints.x] * in.weights.x
    + skin[b + in.joints.y] * in.weights.y
    + skin[b + in.joints.z] * in.weights.z
    + skin[b + in.joints.w] * in.weights.w;
  return U.mvp * (m * vec4f(in.position + normalize(in.normal) * ${HAND_INFLATE}, 1.0));
}

@vertex fn vs_forearm(@location(0) p: vec3f, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  return U.mvp * (U.fore[ii] * vec4f(p, 1.0));
}

@fragment fn fs_occluder() -> @location(0) vec4f { return vec4f(0.0); }
`
);
var DeskGPU = class {
  pipes = /* @__PURE__ */ new Map();
  handPipes = /* @__PURE__ */ new Map();
  forePipes = /* @__PURE__ */ new Map();
  vbuf = null;
  vcount = 0;
  slots = [];
  layout;
  handLayout;
  data = new Float32Array(36);
  tmp = new Float32Array(16);
  skinCpu = new Float32Array(50 * 16);
  skinBuf;
  fore = new Float32Array(32);
  meshes = [null, null];
  gpuHands = [null, null];
  handOn = [false, false];
  foreOn = [false, false];
  sphere = null;
  sphereCount = 0;
  device;
  unlit;
  /** false: never hidden by real-world depth or the hands (markers that sit on the hands) */
  occlude;
  occ;
  constructor(device, unlit = false, occlude = true) {
    this.device = device;
    this.unlit = unlit;
    this.occlude = occlude;
    this.occ = new OccBinder(device);
    this.layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }] });
    const V = GPUShaderStage.VERTEX;
    this.handLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: V, buffer: { type: "uniform" } },
      { binding: 1, visibility: V, buffer: { type: "uniform" } }
    ] });
    this.skinBuf = device.createBuffer({ size: this.skinCpu.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "handSkin" });
  }
  /** WebXR input-profile meshes. Until these arrive the desk is unoccluded by hands. */
  setHandMeshes(left, right) {
    this.meshes[0] = left;
    this.meshes[1] = right;
    this.gpuHands[0] = this.uploadHand(left, "left");
    this.gpuHands[1] = this.uploadHand(right, "right");
    if (!this.sphere) {
      const sphere = forearmSphere();
      this.sphereCount = sphere.count;
      const src = sphere.position;
      this.sphere = this.device.createBuffer({ size: src.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: "forearm" });
      this.device.queue.writeBuffer(this.sphere, 0, src.buffer.slice(src.byteOffset, src.byteOffset + src.byteLength));
    }
  }
  /** Joint poses in room space from {@link readHandPoses}. Null hides that hand. */
  setHandPoses(left, right) {
    const poses = [left, right];
    for (let h = 0; h < 2; h++) {
      const p = poses[h], mesh = this.meshes[h];
      this.handOn[h] = !!(p && mesh);
      this.foreOn[h] = !!(p && forearmMatrix(p, this.fore, h * 16));
      if (p && mesh) writeSkin(this.skinCpu, mesh, p, h);
    }
    if (this.handOn[0] || this.handOn[1]) this.device.queue.writeBuffer(this.skinBuf, 0, this.skinCpu);
  }
  upload(mesh) {
    this.vcount = mesh.length / 7;
    if (!this.vcount) return;
    if (!this.vbuf || this.vbuf.size < mesh.byteLength) {
      this.vbuf?.destroy();
      this.vbuf = this.device.createBuffer({ size: Math.ceil(mesh.byteLength * 1.5 / 4) * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, label: "desk" });
    }
    this.device.queue.writeBuffer(this.vbuf, 0, mesh.buffer, mesh.byteOffset, mesh.byteLength);
  }
  pipe(format) {
    let p = this.pipes.get(format);
    if (p) return p;
    const module = this.device.createShaderModule({ code: DESK_WGSL });
    p = this.device.createRenderPipeline({
      label: "desk",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout, this.occ.layout] }),
      vertex: { module, entryPoint: "vs", buffers: [{ arrayStride: 28, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x4" }
      ] }] },
      fragment: { module, entryPoint: "fs", targets: [{ format, blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
      } }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });
    this.pipes.set(format, p);
    return p;
  }
  uploadHand(mesh, label) {
    const buf = (data, usage, name) => {
      const b = this.device.createBuffer({ size: Math.max(4, data.byteLength), usage, label: name });
      this.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
      return b;
    };
    const V = GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
    return {
      vb: [
        buf(mesh.position, V, `${label}-pos`),
        buf(mesh.normal, V, `${label}-nrm`),
        buf(mesh.joints, V, `${label}-joints`),
        buf(mesh.weights, V, `${label}-weights`)
      ],
      ib: buf(mesh.indices, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, `${label}-idx`),
      count: mesh.indices.length
    };
  }
  handPipe(format) {
    let p = this.handPipes.get(format);
    if (p) return p;
    const module = this.device.createShaderModule({ code: HAND_WGSL });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.handLayout] });
    const depthStencil = { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" };
    const fragment = { module, entryPoint: "fs_occluder", targets: [{ format, writeMask: 0 }] };
    p = this.device.createRenderPipeline({
      label: "hand-occ",
      layout,
      vertex: { module, entryPoint: "vs_hand", buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
        { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
        { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: "uint8x4" }] },
        { arrayStride: 16, attributes: [{ shaderLocation: 3, offset: 0, format: "float32x4" }] }
      ] },
      fragment,
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil
    });
    this.handPipes.set(format, p);
    this.forePipes.set(format, this.device.createRenderPipeline({
      label: "forearm-occ",
      layout,
      vertex: { module, entryPoint: "vs_forearm", buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
      fragment,
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil
    }));
    return p;
  }
  slot(i, w, h) {
    let s = this.slots[i];
    if (!s) {
      const buf = this.device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `deskU${i}` });
      const handBuf = this.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `handU${i}` });
      s = this.slots[i] = {
        buf,
        bg: this.device.createBindGroup({ layout: this.layout, entries: [{ binding: 0, resource: { buffer: buf } }] }),
        handBuf,
        handBg: this.device.createBindGroup({ layout: this.handLayout, entries: [
          { binding: 0, resource: { buffer: handBuf } },
          { binding: 1, resource: { buffer: this.skinBuf } }
        ] }),
        depth: null
      };
    }
    if (!s.depth || s.depth.width !== w || s.depth.height !== h) {
      s.depth?.destroy();
      s.depth = this.device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT, label: `deskDepth${i}` });
    }
    return s;
  }
  draw(enc, eye, i, model) {
    if (!this.vbuf || !this.vcount) return;
    const s = this.slot(i, eye.attachmentWidth, eye.attachmentHeight);
    mul4(eye.proj, mul4(eye.roomToEye, model, this.tmp), this.data);
    this.data.set(model, 16);
    this.data[32] = eye.format.endsWith("-srgb") ? 1 : 0;
    this.data[33] = this.unlit ? 1 : 0;
    this.device.queue.writeBuffer(s.buf, 0, this.data);
    const hands = this.occlude && (this.handOn[0] || this.handOn[1] || this.foreOn[0] || this.foreOn[1]);
    if (hands) {
      mul4(eye.proj, eye.roomToEye, this.tmp);
      this.device.queue.writeBuffer(s.handBuf, 0, this.tmp);
      this.device.queue.writeBuffer(s.handBuf, 64, this.fore);
    }
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: eye.color, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: { view: s.depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" }
    });
    const vp = eye.viewport;
    if (vp.x !== 0 || vp.y !== 0 || vp.width !== eye.attachmentWidth || vp.height !== eye.attachmentHeight) rp.setViewport(vp.x, vp.y, vp.width, vp.height, 0, 1);
    if (hands && this.sphere) {
      this.handPipe(eye.format);
      const fore = this.forePipes.get(eye.format);
      rp.setPipeline(fore);
      rp.setBindGroup(0, s.handBg);
      rp.setVertexBuffer(0, this.sphere);
      for (let h = 0; h < 2; h++) if (this.foreOn[h]) rp.draw(this.sphereCount, 1, 0, h);
      rp.setPipeline(this.handPipes.get(eye.format));
      for (let h = 0; h < 2; h++) {
        const mesh = this.gpuHands[h];
        if (!mesh || !this.handOn[h]) continue;
        for (let k = 0; k < 4; k++) rp.setVertexBuffer(k, mesh.vb[k]);
        rp.setIndexBuffer(mesh.ib, "uint16");
        rp.drawIndexed(mesh.count, 1, 0, 0, h);
      }
    }
    rp.setPipeline(this.pipe(eye.format));
    rp.setBindGroup(0, s.bg);
    rp.setBindGroup(1, eye.occ && this.occlude ? this.occ.forEye(i, eye.occ.room, eye.occ.texture, eye.occ.kind, eye.occ.view) : this.occ.off);
    rp.setVertexBuffer(0, this.vbuf);
    rp.draw(this.vcount);
    rp.end();
  }
};

// src/xr.ts
var OCC_MARGIN = 0.08;
function mul42(a, b, out = new Float32Array(16)) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = s;
  }
  return out;
}
function projNear(p) {
  if (Math.abs(p[10]) < 1e-8) return 0;
  const near = p[14] / p[10];
  return near > 0 && near < 10 ? near : 0;
}
var NDC_TO_UV = new Float32Array([0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 1]);
function occlusionData(cam, scale, eyePose, depthPoseInv, depthProj, uvFromView, rawValueToMeters, kind, width, height, windowDepth = null, out = new Float32Array(32), room = null) {
  const s = scale;
  const E = new Float32Array([s / cam.fx, 0, 0, 0, 0, -s / cam.fy, 0, 0, -s * cam.cx / cam.fx, s * cam.cy / cam.fy, -s, 0, 0, 0, 0, 1]);
  const A = mul42(mul42(depthPoseInv, eyePose), E);
  const toUv = mul42(uvFromView, NDC_TO_UV);
  out.set(mul42(toUv, mul42(depthProj, A)), 0);
  out[16] = A[2];
  out[17] = A[6];
  out[18] = A[10];
  out[19] = A[14];
  out[20] = rawValueToMeters;
  out[21] = KIND_CODE[kind];
  out[22] = width;
  out[23] = height;
  out[24] = windowDepth ? windowDepth.near : projNear(depthProj);
  out[25] = OCC_MARGIN;
  out[26] = windowDepth ? windowDepth.far : 0;
  out[27] = windowDepth ? 1 : 0;
  if (room) {
    room.set(out);
    room.set(mul42(toUv, mul42(depthProj, depthPoseInv)), 0);
    room[16] = depthPoseInv[2];
    room[17] = depthPoseInv[6];
    room[18] = depthPoseInv[10];
    room[19] = depthPoseInv[14];
  }
  return out;
}
var dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
var cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
var sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
var add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
var mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (l < 1e-8) return null;
  return [a[0] / l, a[1] / l, a[2] / l];
}
function horizontal(v, up) {
  return norm(sub(v, mul(up, dot(v, up))));
}
function frameOf(forward, upIn) {
  const up = norm(upIn);
  if (!up) return null;
  let fwd = horizontal(forward, up);
  if (!fwd) {
    const hint = Math.abs(up[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    fwd = norm(cross(hint, up));
    if (!fwd) return null;
  }
  const right = norm(cross(fwd, up));
  if (!right) return null;
  return [right, up, fwd];
}
function m3(cols) {
  const m = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[c * 3 + r] = cols[c][r];
  return m;
}
function mulM3(a, b) {
  const m = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) {
    m[c * 3 + r] = a[r] * b[c * 3] + a[3 + r] * b[c * 3 + 1] + a[6 + r] * b[c * 3 + 2];
  }
  return m;
}
function transposeM3(a) {
  const m = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[c * 3 + r] = a[r * 3 + c];
  return m;
}
function mulM3V(a, v) {
  return [
    a[0] * v[0] + a[3] * v[1] + a[6] * v[2],
    a[1] * v[0] + a[4] * v[1] + a[7] * v[2],
    a[2] * v[0] + a[5] * v[1] + a[8] * v[2]
  ];
}
function place(cam, center, up, head2, headForward, meters) {
  const scene = frameOf(sub(center, cam), up);
  const room = frameOf(headForward, [0, 1, 0]);
  if (!scene || !room) return null;
  const dist = Math.hypot(center[0] - cam[0], center[1] - cam[1], center[2] - cam[2]);
  const scale = dist > 1e-8 && meters > 0 ? meters / dist : 1;
  return {
    R: mulM3(m3(room), transposeM3(m3(scene))),
    scale,
    scale0: scale,
    cam: [cam[0], cam[1], cam[2]],
    anchor: [head2[0], head2[1], head2[2]],
    locked: false,
    pivot: [center[0], center[1], center[2]]
  };
}
function intrinsics(p, width, height) {
  return {
    fx: p[0] * 0.5 * width,
    fy: p[5] * 0.5 * height,
    cx: width * (1 - p[8]) * 0.5,
    cy: height * (1 + p[9]) * 0.5
  };
}
function eyeCamera(p, proj, pose, width, height) {
  const Rt = transposeM3(p.R);
  const c0 = mulM3V(Rt, [pose[0], pose[1], pose[2]]);
  const c1 = mulM3V(Rt, [pose[4], pose[5], pose[6]]);
  const c2 = mulM3V(Rt, [pose[8], pose[9], pose[10]]);
  const r0 = c0;
  const r1 = mul(c1, -1);
  const r2 = mul(c2, -1);
  const eye = [pose[12], pose[13], pose[14]];
  const campos = add(p.cam, mul(mulM3V(Rt, sub(eye, p.anchor)), 1 / p.scale));
  const view = new Float32Array(16);
  const rows = [r0, r1, r2];
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) view[c * 4 + r] = rows[r][c];
  view[12] = -(r0[0] * campos[0] + r0[1] * campos[1] + r0[2] * campos[2]);
  view[13] = -(r1[0] * campos[0] + r1[1] * campos[1] + r1[2] * campos[2]);
  view[14] = -(r2[0] * campos[0] + r2[1] * campos[1] + r2[2] * campos[2]);
  view[15] = 1;
  const k = intrinsics(proj, width, height);
  return { view, campos, ...k, width, height };
}
var PLACE_ARM_MS = 400;
var PLACE_ARM_SPAN = 0.12;
var PLACE_YAW_SPAN = 0.1;
var PLACE_JUMP = 0.05;
var PLACE_YAW_JUMP = 0.18;
var SCALE_STEP = 0.08;
var PINCH_ON = 0.016;
var PINCH_OFF = 0.04;
var PINCH_ON_BONE = 0.032;
var PINCH_OFF_BONE = 0.055;
function grabState() {
  return { armAt: null, placing: false, last: null };
}
function pinchClosed(was, gap, radii) {
  const on = radii > 0 ? PINCH_ON : PINCH_ON_BONE;
  const off = radii > 0 ? PINCH_OFF : PINCH_OFF_BONE;
  return was ? gap < off : gap < on;
}
var DOUBLE_PINCH_MS = 450;
var PINCH_CLICK_MS = 320;
var DoublePinch = class {
  lastClickAt = 0;
  lastHand = null;
  downAt = [0, 0];
  down = [false, false];
  feed(hand, held, rising, now, claimed, otherHeld = false) {
    if (claimed || otherHeld) {
      this.down[hand] = held;
      if (rising || claimed) {
        this.lastClickAt = 0;
        this.lastHand = null;
      }
      return false;
    }
    if (rising) {
      const dbl = this.lastHand === hand && now - this.lastClickAt > 0 && now - this.lastClickAt <= DOUBLE_PINCH_MS;
      this.downAt[hand] = now;
      this.down[hand] = true;
      if (dbl) {
        this.lastClickAt = 0;
        this.lastHand = null;
      }
      return dbl;
    }
    if (!held && this.down[hand]) {
      this.down[hand] = false;
      if (now - this.downAt[hand] <= PINCH_CLICK_MS) {
        this.lastClickAt = now;
        this.lastHand = hand;
      } else {
        this.lastClickAt = 0;
        this.lastHand = null;
      }
    }
    return false;
  }
};
function copy3(v) {
  return [v[0], v[1], v[2]];
}
function yawAboutY(radians) {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  const m = new Float32Array(9);
  m[0] = c;
  m[2] = -s;
  m[4] = 1;
  m[6] = s;
  m[8] = c;
  return m;
}
function applyGrab(p, from, to) {
  const [la, lb] = from;
  const [a, b] = to;
  if (Math.hypot(a[0] - la[0], a[1] - la[1], a[2] - la[2]) > PLACE_JUMP) return false;
  if (Math.hypot(b[0] - lb[0], b[1] - lb[1], b[2] - lb[2]) > PLACE_JUMP) return false;
  const lastMid = mul(add(la, lb), 0.5);
  const mid = mul(add(a, b), 0.5);
  let k = 1;
  const lastDist = Math.hypot(lb[0] - la[0], lb[1] - la[1], lb[2] - la[2]);
  const dist = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  if (dist >= 0.06 && lastDist >= 0.06) {
    const next = dist / lastDist;
    if (Math.abs(next - 1) <= SCALE_STEP) k = next;
  }
  const lo = p.scale0 * 0.05;
  const hi = p.scale0 * 20;
  if (p.scale > 1e-8) {
    if (p.scale * k < lo) k = lo / p.scale;
    else if (p.scale * k > hi) k = hi / p.scale;
  }
  let yaw = 0;
  const spanH = Math.hypot(b[0] - a[0], b[2] - a[2]);
  const lastH = Math.hypot(lb[0] - la[0], lb[2] - la[2]);
  if (spanH >= PLACE_YAW_SPAN && lastH >= PLACE_YAW_SPAN) {
    const ang = Math.atan2(b[0] - a[0], b[2] - a[2]);
    const prev = Math.atan2(lb[0] - la[0], lb[2] - la[2]);
    const d = Math.atan2(Math.sin(ang - prev), Math.cos(ang - prev));
    if (Math.abs(d) <= PLACE_YAW_JUMP) yaw = d;
  }
  const spun = mulM3V(yawAboutY(yaw), sub(p.anchor, lastMid));
  p.anchor = add(mid, mul(spun, k));
  p.scale *= k;
  if (yaw !== 0) p.R = mulM3(yawAboutY(yaw), p.R);
  return true;
}
function stepGrab(g, a, b, now, placement) {
  const span = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const spanning = span >= (g.armAt !== null && !g.placing ? PLACE_ARM_SPAN * 0.65 : PLACE_ARM_SPAN);
  if (g.placing || spanning) {
    if (!g.placing) {
      if (g.armAt === null) g.armAt = now;
      if (now - g.armAt >= PLACE_ARM_MS) g.placing = true;
    }
    if (g.placing) {
      if (g.last) applyGrab(placement, g.last, [a, b]);
      g.last = [copy3(a), copy3(b)];
    }
  } else {
    g.armAt = null;
    g.placing = false;
    g.last = null;
  }
}
function jointAt(frame, space, ref) {
  if (!space) return null;
  const withJoint = frame;
  const pose = withJoint.getJointPose?.(space, ref) ?? frame.getPose(space, ref);
  if (!pose) return null;
  const q = pose.transform.position;
  return { p: [q.x, q.y, q.z], r: pose.radius ?? 0 };
}
var poseHold = [new Float32Array(JOINTS.length * 16), new Float32Array(JOINTS.length * 16)];
function readHandPoses(frame, session, ref) {
  const out = [null, null];
  const get = frame.getJointPose;
  for (let i = 0; i < session.inputSources.length; i++) {
    const source = session.inputSources[i];
    const hand = source.hand;
    const handed = source.handedness;
    if (!hand || handed !== "left" && handed !== "right") continue;
    const side = handed === "left" ? 0 : 1;
    const dst = poseHold[side];
    let wrist = null;
    let n = 0;
    for (let j = 0; j < JOINTS.length; j++) {
      let pose;
      try {
        const space = hand.get(JOINTS[j]);
        pose = space ? get?.call(frame, space, ref) ?? frame.getPose(space, ref) ?? void 0 : void 0;
      } catch {
        pose = void 0;
      }
      if (pose) {
        dst.set(pose.transform.matrix, j * 16);
        if (j === 0) wrist = dst.subarray(0, 16);
        n++;
      } else if (wrist) {
        dst.set(wrist, j * 16);
      } else {
        n = 0;
        break;
      }
    }
    if (n > 0 && wrist) out[side] = dst;
  }
  return out;
}
function readTips(frame, hand, ref) {
  const thumb = jointAt(frame, hand.get("thumb-tip"), ref);
  const index = jointAt(frame, hand.get("index-finger-tip"), ref);
  if (!thumb || !index) return null;
  return {
    gap: Math.hypot(thumb.p[0] - index.p[0], thumb.p[1] - index.p[1], thumb.p[2] - index.p[2]) - thumb.r - index.r,
    radii: thumb.r + index.r,
    point: [(thumb.p[0] + index.p[0]) * 0.5, (thumb.p[1] + index.p[1]) * 0.5, (thumb.p[2] + index.p[2]) * 0.5],
    index: index.p
  };
}
var Hands = class {
  pinch = /* @__PURE__ */ new Map();
  gate = grabState();
  clicks = new DoublePinch();
  onDoublePinch = null;
  tips = {};
  update(frame, session, ref, placement, now, ui) {
    const pts = /* @__PURE__ */ new Map();
    const seen = /* @__PURE__ */ new Set();
    const tips = {};
    this.tips = tips;
    for (let i = 0; i < session.inputSources.length; i++) {
      const source = session.inputSources[i];
      const side = source.handedness;
      if (side !== "left" && side !== "right") continue;
      const was = this.pinch.get(side) ?? false;
      let point = null;
      let tip = null;
      let held = false;
      let ray = null;
      try {
        const aim = frame.getPose(source.targetRaySpace, ref);
        if (aim) {
          const m = aim.transform.matrix;
          ray = { o: [m[12], m[13], m[14]], d: [-m[8], -m[9], -m[10]] };
        }
        if (source.hand) {
          const tips2 = readTips(frame, source.hand, ref);
          if (tips2) {
            point = tips2.point;
            tip = tips2.index;
            held = pinchClosed(was, tips2.gap, tips2.radii);
          }
          const wrist = jointAt(frame, source.hand.get("wrist"), ref);
          if (ray && wrist) ray.o = wrist.p;
        } else if (source.gripSpace) {
          const grip = frame.getPose(source.gripSpace, ref);
          if (grip) {
            const q = grip.transform.position;
            point = tip = [q.x, q.y, q.z];
            held = !!source.gamepad?.buttons[0]?.pressed;
          }
        } else continue;
      } catch {
        point = null;
      }
      held &&= !!point;
      seen.add(side);
      this.pinch.set(side, held);
      if (held && tip) tips[side] = tip;
      const hand = side === "left" ? 0 : 1;
      const rising = held && !was;
      const claimed = ui?.update(hand, point, held, rising && !this.gate.placing, ray) ?? false;
      const other = this.pinch.get(side === "left" ? "right" : "left") ?? false;
      if (this.clicks.feed(hand, held, rising, now, claimed || this.gate.placing, other)) this.onDoublePinch?.();
      if (held && point && !claimed) pts.set(side, point);
    }
    for (const side of ["left", "right"]) {
      if (seen.has(side)) continue;
      this.pinch.delete(side);
      ui?.update(side === "left" ? 0 : 1, null, false, false, null);
    }
    const a = pts.get("left");
    const b = pts.get("right");
    if (!a || !b) {
      this.gate.armAt = null;
      this.gate.placing = false;
      this.gate.last = null;
      return;
    }
    stepGrab(this.gate, a, b, now, placement);
  }
};
function depthApi(binding, session) {
  const names = (o) => {
    const out = /* @__PURE__ */ new Set();
    for (let p = o; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
      for (const k of Object.getOwnPropertyNames(p)) if (/depth/i.test(k)) out.add(k);
    }
    return [...out].join(",") || "-";
  };
  const g = globalThis;
  const globals = Object.getOwnPropertyNames(globalThis).filter((k) => /^XR.*Depth/i.test(k)).join(",") || "-";
  return `binding[${names(binding)}] session[${names(session)}] frame[${names(g.XRFrame?.prototype)}] webgl[${names(g.XRWebGLBinding?.prototype)}] globals[${globals}]`;
}
var ImmersiveAR = class _ImmersiveAR {
  session;
  binding;
  layer;
  refSpace;
  colorFormat;
  /** The real world shows through wherever alpha is 0. */
  passthrough;
  /** how depth reaches the occlusion test: the binding's GPU texture, or XRFrame CPU data uploaded each frame */
  depthMode;
  get depthSensing() {
    return this.depthMode !== null;
  }
  device;
  cpuDepth = [];
  cpuDataFormat;
  /** turns the real-world depth test off without ending the session */
  occlude = true;
  /** tint occluded splat fragments red instead of dropping them */
  occDebug = false;
  /** P40 (Quest): splats are rendered at this fraction of the eye viewport and upscaled by the eye blit. Independent of
   *  the layer scaleFactor, which Quest Browser 152 ignores for WebGPU layers (fixed 1680×1760, Meta investigation
   *  2144687486168018). 1 = native. */
  renderScale = 1;
  occLogged = false;
  hands = new Hands();
  onDoublePinch = null;
  occData = [];
  occRoom = [];
  constructor(session, binding, layer, refSpace, colorFormat, passthrough, depthMode, device, cpuDataFormat) {
    this.session = session;
    this.binding = binding;
    this.layer = layer;
    this.refSpace = refSpace;
    this.colorFormat = colorFormat;
    this.passthrough = passthrough;
    this.depthMode = depthMode;
    this.device = device;
    this.cpuDataFormat = cpuDataFormat;
  }
  static async diagnose(mode = "immersive-ar") {
    if (!navigator.xr) return "no WebXR (navigator.xr)";
    const ok = await navigator.xr.isSessionSupported(mode).catch(() => false);
    if (!ok) return `${mode} not supported`;
    if (!("XRGPUBinding" in globalThis)) return "no WebXR/WebGPU binding (XRGPUBinding)";
    return null;
  }
  static async start(device, scaleFactor = 1, opts = {}) {
    const gpuDepthApi = typeof XRGPUBinding.prototype.getDepthInformation === "function";
    const init = {
      requiredFeatures: ["webgpu"],
      optionalFeatures: ["local-floor", "hand-tracking", "depth-sensing"],
      depthSensing: {
        usagePreference: gpuDepthApi ? ["gpu-optimized", "cpu-optimized"] : ["cpu-optimized", "gpu-optimized"],
        dataFormatPreference: ["float32", "unsigned-short", "luminance-alpha"],
        matchDepthView: false
      }
    };
    const session = await requestWithTimeout("immersive-ar", init, 6e3);
    try {
      const binding = new XRGPUBinding(session, device);
      const colorFormat = binding.getPreferredColorFormat?.() ?? navigator.gpu.getPreferredCanvasFormat();
      const layer = binding.createProjectionLayer({ colorFormat, scaleFactor });
      if (opts.foveation !== void 0) {
        try {
          layer.fixedFoveation = opts.foveation;
        } catch {
        }
      }
      if (opts.frameRate) {
        const s = session;
        if (s.updateTargetFrameRate && (!s.supportedFrameRates || [...s.supportedFrameRates].includes(opts.frameRate))) {
          void s.updateTargetFrameRate(opts.frameRate).catch(() => {
          });
        }
      }
      session.updateRenderState({ layers: [layer], depthNear: 0.01, depthFar: 50 });
      const features = session.enabledFeatures ?? [];
      const floor = features.includes("local-floor");
      const refSpace = await session.requestReferenceSpace(floor ? "local-floor" : "local");
      const blend = session.environmentBlendMode;
      const passthrough = blend === "alpha-blend" || blend === "additive";
      const { mode, format } = _ImmersiveAR.depthMode(session, binding);
      console.info(`MR: ${colorFormat} blend=${blend} floor=${floor} hands=${features.includes("hand-tracking")} depth=${mode ?? "off"}/${format} features=${features.join(",")} getDepthInformation binding=${typeof binding.getDepthInformation} frame=${typeof XRFrame.prototype.getDepthInformation} usage=${session.depthUsage} scale=${scaleFactor} foveation=${layer.fixedFoveation ?? "n/a"}`);
      if (features.includes("depth-sensing")) console.info(`MR depth api: ${depthApi(binding, session)}`);
      return new _ImmersiveAR(session, binding, layer, refSpace, colorFormat, passthrough, mode, device, format);
    } catch (error) {
      void session.end().catch(() => {
      });
      throw error;
    }
  }
  /** Granted usage decides the path; for GPU depth the texture format decides per frame whether the shaders can read it. */
  static depthMode(session, binding) {
    if (!session.enabledFeatures?.includes("depth-sensing")) return { mode: null, format: "-" };
    const s = session;
    let usage = "", format = "unknown";
    try {
      usage = s.depthUsage ?? "";
      format = s.depthDataFormat ?? "unknown";
    } catch {
    }
    const frameApi = typeof XRFrame.prototype.getDepthInformation === "function";
    if (usage === "cpu-optimized") return { mode: frameApi ? "cpu" : null, format };
    return { mode: binding.getDepthInformation ? "gpu" : null, format };
  }
  probed = false;
  /** Logs once what XRFrame.getDepthInformation yields, since the WebGPU binding path may be missing. */
  probeDepth(frame, pose) {
    if (this.probed || !this.session.enabledFeatures?.includes("depth-sensing")) return;
    this.probed = true;
    const get = frame.getDepthInformation;
    if (typeof get !== "function") {
      console.info("MR depth probe: XRFrame.getDepthInformation missing");
      return;
    }
    try {
      const d = get.call(frame, pose.views[0]);
      if (!d) {
        console.info("MR depth probe: frame depth null");
        return;
      }
      const fields = ["width", "height", "rawValueToMeters", "depthNear", "depthFar"].map((k) => `${k}=${String(d[k])}`).join(" ");
      const data = d.data;
      console.info(`MR depth probe: ${Object.prototype.toString.call(d)} ${fields} data=${data ? data.byteLength : "-"} texture=${d.texture ? String(d.texture) : "-"}`);
    } catch (e) {
      console.warn("MR depth probe: frame.getDepthInformation threw", e instanceof Error ? e.message : String(e));
    }
  }
  /** CPU depth for `view` copied into this eye's r32float / r16uint texture (luminance-alpha is the same 16-bit layout). */
  cpuOcclusion(frame, view, cam, scale, slot) {
    let info = null;
    try {
      info = frame.getDepthInformation(view);
    } catch (e) {
      if (!this.occLogged) {
        this.occLogged = true;
        console.warn("MR depth (cpu): getDepthInformation threw", e instanceof Error ? e.message : String(e));
      }
      return null;
    }
    if (!info?.data) return null;
    const format = this.cpuDataFormat === "float32" ? "r32float" : "r16uint";
    const { width: w, height: h } = info;
    let tex = this.cpuDepth[slot];
    if (!tex || tex.width !== w || tex.height !== h || tex.format !== format) {
      tex?.destroy();
      tex = this.cpuDepth[slot] = this.device.createTexture({ size: [w, h], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: `cpuDepth${slot}` });
    }
    this.device.queue.writeTexture({ texture: tex }, info.data, { bytesPerRow: w * (format === "r32float" ? 4 : 2) }, [w, h]);
    const n = info.depthNear, f = info.depthFar;
    const windowDepth = typeof n === "number" && typeof f === "number" && n > 0 ? { near: n, far: f > n ? f : 1e6 } : null;
    const proj = info.projectionMatrix ?? view.projectionMatrix;
    const poseInv = (info.transform ?? view.transform).inverse.matrix;
    if (!this.occLogged) {
      this.occLogged = true;
      console.info(`MR depth (cpu): ${this.cpuDataFormat} -> ${format} ${w}x${h} bytes ${info.data.byteLength}, rawValueToMeters ${info.rawValueToMeters}, near ${n} far ${f} (${windowDepth ? "hardware depth" : "linear"}), own projection ${!!info.projectionMatrix}, own transform ${!!info.transform}, uvFromView ${Array.from(info.normDepthBufferFromNormView.matrix).map((v) => v.toFixed(2)).join(",")}`);
    }
    const kind = depthKind(format);
    const data = this.occData[slot] ?? (this.occData[slot] = new Float32Array(32));
    const room = this.occRoom[slot] ?? (this.occRoom[slot] = new Float32Array(32));
    occlusionData(
      cam,
      scale,
      view.transform.matrix,
      poseInv,
      proj,
      info.normDepthBufferFromNormView.matrix,
      info.rawValueToMeters,
      kind,
      w,
      h,
      windowDepth,
      data,
      room
    );
    data[28] = room[28] = this.occDebug ? 1 : 0;
    return { data, room, texture: tex, kind, view: {} };
  }
  occlusionFor(frame, view, cam, scale, slot) {
    if (!this.depthSensing || !this.occlude) return null;
    if (this.depthMode === "cpu") return this.cpuOcclusion(frame, view, cam, scale, slot);
    let info = null;
    try {
      info = this.binding.getDepthInformation(view);
    } catch (e) {
      if (!this.occLogged) {
        this.occLogged = true;
        console.warn("MR depth: getDepthInformation threw", e instanceof Error ? e.message : String(e));
      }
      return null;
    }
    if (!info?.texture) {
      if (!this.occLogged) {
        this.occLogged = true;
        console.info(`MR depth: no depth information for this frame (${info ? "no texture" : "null"})`);
      }
      return null;
    }
    const fmt = info.texture.format;
    const kind = depthKind(fmt);
    const n = info.depthNear, f = info.depthFar;
    const windowDepth = typeof n === "number" && typeof f === "number" && n > 0 ? { near: n, far: f > n ? f : 1e6 } : null;
    const layers = info.texture.depthOrArrayLayers;
    const src = info.getViewDescriptor?.();
    if (!this.occLogged) {
      this.occLogged = true;
      console.info(`MR depth: ${fmt} ${kind ? "" : "(unsupported) "}${info.width}x${info.height}x${layers}, usage ${info.texture.usage}, rawValueToMeters ${info.rawValueToMeters}, near ${n} far ${f} (${windowDepth ? "hardware depth" : "linear"}), view descriptor ${src ? JSON.stringify(src) : "none"}, proj[10] ${info.projectionMatrix[10].toFixed(4)} proj[14] ${info.projectionMatrix[14].toFixed(4)}, uvFromView ${Array.from(info.normDepthBufferFromNormView.matrix).map((v) => v.toFixed(2)).join(",")}`);
    }
    if (!kind) return null;
    const data = this.occData[slot] ?? (this.occData[slot] = new Float32Array(32));
    const room = this.occRoom[slot] ?? (this.occRoom[slot] = new Float32Array(32));
    occlusionData(
      cam,
      scale,
      view.transform.matrix,
      info.transform.inverse.matrix,
      info.projectionMatrix,
      info.normDepthBufferFromNormView.matrix,
      info.rawValueToMeters,
      kind,
      info.width,
      info.height,
      windowDepth,
      data,
      room
    );
    data[28] = room[28] = this.occDebug ? 1 : 0;
    const desc = src ? { ...src } : layers > 1 ? { baseArrayLayer: slot } : {};
    delete desc.format;
    if (layers > 1) {
      desc.dimension = "2d";
      desc.arrayLayerCount = 1;
    }
    return { data, room, texture: info.texture, kind, view: desc };
  }
  /** Feed `ui` the pinch points, then move, yaw and scale the placement when both free hands are pinched. */
  pinch(frame, placement, now, ui = null) {
    this.hands.onDoublePinch = this.onDoublePinch;
    this.hands.update(frame, this.session, this.refSpace, placement, now, ui);
  }
  /** pinching hands of the last {@link pinch} */
  get pinchTips() {
    return this.hands.tips;
  }
  /** Per-eye cameras and projection-layer views for this pose. */
  eyes(frame, pose, placement) {
    const encodeSrgb = !this.colorFormat.endsWith("-srgb");
    const keepAlpha = this.passthrough;
    const drawn = [];
    const out = [];
    for (const view of pose.views) {
      const sub2 = this.binding.getViewSubImage(this.layer, view);
      const vp = sub2.viewport;
      if (vp.width < 1 || vp.height < 1) continue;
      const tex = sub2.colorTexture;
      const srcDesc = sub2.getViewDescriptor?.();
      const layerIndex = srcDesc?.baseArrayLayer ?? 0;
      const mip = srcDesc?.baseMipLevel ?? 0;
      const desc = { ...srcDesc ?? {} };
      delete desc.format;
      const clear = !drawn.some((d) => d.tex === tex && d.layer === layerIndex);
      drawn.push({ tex, layer: layerIndex });
      const rs = this.renderScale > 0 && this.renderScale < 1 ? this.renderScale : 1;
      const cam = eyeCamera(
        placement,
        view.projectionMatrix,
        view.transform.matrix,
        Math.max(1, Math.round(vp.width * rs)),
        Math.max(1, Math.round(vp.height * rs))
      );
      out.push({
        cam,
        occ: this.occlusionFor(frame, view, cam, placement.scale, out.length),
        color: tex.createView(desc),
        format: this.colorFormat,
        attachmentWidth: Math.max(1, tex.width >> mip),
        attachmentHeight: Math.max(1, tex.height >> mip),
        viewport: { x: vp.x, y: vp.y, width: vp.width, height: vp.height },
        clear,
        keepAlpha,
        encodeSrgb,
        proj: view.projectionMatrix,
        roomToEye: view.transform.inverse.matrix
      });
    }
    return out;
  }
};
function requestWithTimeout(mode, init, ms) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error(`requestSession timed out after ${ms}ms`));
    }, ms);
    navigator.xr.requestSession(mode, init).then(
      (session) => {
        if (settled) {
          void session.end();
          return;
        }
        settled = true;
        window.clearTimeout(timer);
        resolve(session);
      },
      (error) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    );
  });
}

// src/camViz.ts
var camToWorld = (R, t, p) => {
  const q = [p[0] - t[0], p[1] - t[1], p[2] - t[2]];
  return [
    R[0][0] * q[0] + R[1][0] * q[1] + R[2][0] * q[2],
    R[0][1] * q[0] + R[1][1] * q[1] + R[2][1] * q[2],
    R[0][2] * q[0] + R[1][2] * q[1] + R[2][2] * q[2]
  ];
};
var camposOf = (R, t) => [
  -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
  -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
  -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2])
];
function project(view, p) {
  const v = view.view;
  const x = v[0] * p[0] + v[4] * p[1] + v[8] * p[2] + v[12];
  const y = v[1] * p[0] + v[5] * p[1] + v[9] * p[2] + v[13];
  const z = v[2] * p[0] + v[6] * p[1] + v[10] * p[2] + v[14];
  if (z < 1e-4) return null;
  return {
    x: view.fx * (x / z) + view.cx,
    y: view.fy * (y / z) + view.cy,
    z
  };
}
var FRUSTUM_Z = 0.22;
var REAL_CAM = "c24";
var REAL_LABEL = "Real view";
var SYNTH_LABEL = "synthetic";
var REAL_FRUSTUM_Z = 0.7;
var SYNTH_FRAMES = 64;
function stripFrame(imgW, frame, frames = SYNTH_FRAMES) {
  const n = Math.max(1, frames);
  const w = imgW / n;
  const i = Math.min(n - 1, Math.max(0, Math.floor(frame)));
  return { x: i * w, w };
}
function halfToFloat(h) {
  const s = (h & 32768) >> 15;
  const e = (h & 31744) >> 10;
  const f = h & 1023;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}
function behindActor(eye, point, actor) {
  const dx = point[0] - eye[0], dy = point[1] - eye[1], dz = point[2] - eye[2];
  const ox = eye[0] - actor.center[0], oz = eye[2] - actor.center[2];
  const a = dx * dx + dz * dz;
  const c = ox * ox + oz * oz - actor.radius * actor.radius;
  const y0 = actor.center[1] - actor.halfHeight, y1 = actor.center[1] + actor.halfHeight;
  if (a < 1e-8) return c <= 0 && eye[1] >= y0 && eye[1] <= y1 && dy * dy > 1e-8;
  const b = 2 * (ox * dx + oz * dz);
  const disc = b * b - 4 * a * c;
  if (disc < 0) return false;
  const s = Math.sqrt(disc);
  const t0 = (-b - s) / (2 * a);
  const t1 = (-b + s) / (2 * a);
  const enter = t0 > 0 ? t0 : t1 > 0 ? 0 : -1;
  if (enter < 0 || enter >= 1) return false;
  const y = eye[1] + dy * enter;
  return y >= y0 && y <= y1;
}
function drawImageQuad(ctx, img, sw, sh, p, behind) {
  const paint = (d0, d1, d2, s0x, s0y, s1x, s1y, s2x, s2y) => {
    const denom = s0x * (s1y - s2y) + s1x * (s2y - s0y) + s2x * (s0y - s1y);
    if (!(Math.abs(denom) > 1e-8)) return;
    const m11 = (d0.x * (s1y - s2y) + d1.x * (s2y - s0y) + d2.x * (s0y - s1y)) / denom;
    const m12 = (d0.y * (s1y - s2y) + d1.y * (s2y - s0y) + d2.y * (s0y - s1y)) / denom;
    const m21 = (d0.x * (s2x - s1x) + d1.x * (s0x - s2x) + d2.x * (s1x - s0x)) / denom;
    const m22 = (d0.y * (s2x - s1x) + d1.y * (s0x - s2x) + d2.y * (s1x - s0x)) / denom;
    const dx = (d0.x * (s1x * s2y - s2x * s1y) + d1.x * (s2x * s0y - s0x * s2y) + d2.x * (s0x * s1y - s1x * s0y)) / denom;
    const dy = (d0.y * (s1x * s2y - s2x * s1y) + d1.y * (s2x * s0y - s0x * s2y) + d2.y * (s0x * s1y - s1x * s0y)) / denom;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(d0.x, d0.y);
    ctx.lineTo(d1.x, d1.y);
    ctx.lineTo(d2.x, d2.y);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(m11, m12, m21, m22, dx, dy);
    ctx.drawImage(img, 0, 0, sw, sh);
    ctx.restore();
  };
  const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: (a.z ?? 0) + ((b.z ?? 0) - (a.z ?? 0)) * t });
  const at = (u, v) => lerp(lerp(p[0], p[1], u), lerp(p[3], p[2], u), v);
  const N = behind ? 8 : 1;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const u0 = i / N, u1 = (i + 1) / N, v0 = j / N, v1 = (j + 1) / N;
      if (behind) {
        const m = at((u0 + u1) / 2, (v0 + v1) / 2);
        if (behind(m.x, m.y, m.z)) continue;
      }
      const q00 = at(u0, v0), q10 = at(u1, v0), q11 = at(u1, v1), q01 = at(u0, v1);
      const sx0 = u0 * sw, sx1 = u1 * sw, sy0 = v0 * sh, sy1 = v1 * sh;
      paint(q00, q10, q01, sx0, sy0, sx1, sy0, sx0, sy1);
      paint(q10, q11, q01, sx1, sy0, sx1, sy1, sx0, sy1);
    }
  }
}
var CamVizOverlay = class {
  canvas;
  ctx;
  strips = /* @__PURE__ */ new Map();
  labels = /* @__PURE__ */ new Map();
  mesh = [];
  bakeKey = "";
  stripToken = 0;
  stripMiss = false;
  /** Live iPhone clip for c24. The rig around it plays the synthetic training views. */
  realVideo = null;
  realActive = false;
  enabled = false;
  constructor(host) {
    this.canvas = document.createElement("canvas");
    this.canvas.id = "cam-viz";
    Object.assign(this.canvas.style, {
      position: "fixed",
      inset: "0",
      width: "100%",
      height: "100%",
      pointerEvents: "none",
      zIndex: "4"
    });
    host.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
  }
  /** Load one frame-strip per rig camera. c24 plays the shared real clip on every take. */
  bind(key, take, cams) {
    if (this.simple) {
      if (key === this.bakeKey) return;
      this.bakeKey = key;
      this.posterLabel = take;
      const img = new Image();
      img.decoding = "async";
      img.src = new URL(`synth/${take}/c00.jpg`, import.meta.url).href;
      this.poster = img;
      return;
    }
    if (key === this.bakeKey) return;
    this.strips.clear();
    this.bakeKey = key;
    this.stripMiss = false;
    const token = ++this.stripToken;
    cams.forEach((cam, i) => {
      if (cam.name === REAL_CAM) return;
      const idx = cam.idx ?? i;
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        if (this.stripToken === token) this.strips.set(cam.name, img);
      };
      img.onerror = () => {
        if (this.stripToken !== token || this.stripMiss) return;
        this.stripMiss = true;
        console.warn("synthetic training views missing for", take);
      };
      img.src = new URL(`synth/${take}/c${String(idx).padStart(2, "0")}.jpg`, import.meta.url).href;
    });
  }
  resize(cssW, cssH, pixelW, pixelH) {
    if (this.canvas.width === pixelW && this.canvas.height === pixelH) return;
    this.canvas.width = pixelW;
    this.canvas.height = pixelH;
    this.canvas.style.width = cssW + "px";
    this.canvas.style.height = cssH + "px";
  }
  draw(view, cams, actor, frame = 0) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (this.simple) {
      const img = this.poster;
      if (!this.enabled || !img || !img.naturalWidth) return;
      const sw = img.naturalWidth / SYNTH_FRAMES;
      const sh = img.naturalHeight;
      const maxH = Math.min(H * 0.32, 420);
      const w = maxH * (sw / sh);
      const x = Math.max(16, W - w - 28);
      const y = Math.max(16, H * 0.22);
      ctx.drawImage(img, 0, 0, sw, sh, x, y, w, maxH);
      ctx.font = `600 ${Math.max(13, Math.round(W * 0.013))}px "IBM Plex Mono", ui-monospace, monospace`;
      ctx.fillStyle = "#c8f135";
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillText(this.posterLabel || "", x + w, y - 6);
      return;
    }
    if (!this.enabled || !cams.length) return;
    const sx = W / view.width, sy = H / view.height;
    const eye = [view.campos[0], view.campos[1], view.campos[2]];
    const rest = this.realActive ? cams.filter((c) => c.name !== REAL_CAM) : cams;
    const real = this.realActive ? cams.filter((c) => c.name === REAL_CAM) : [];
    for (const cam of rest) this.drawCam(ctx, view, cam, actor, eye, sx, sy, W, false, frame);
    for (const cam of real) this.drawCam(ctx, view, cam, actor, eye, sx, sy, W, true, frame);
  }
  drawCam(ctx, view, cam, actor, eye, sx, sy, W, real, frame) {
    const origin = camposOf(cam.R, cam.t);
    const z = real ? REAL_FRUSTUM_Z : FRUSTUM_Z;
    const hx = 0.5 * cam.w / cam.fx * z;
    const hy = 0.5 * cam.h / cam.fy * z;
    const corners = [
      camToWorld(cam.R, cam.t, [-hx, -hy, z]),
      camToWorld(cam.R, cam.t, [hx, -hy, z]),
      camToWorld(cam.R, cam.t, [hx, hy, z]),
      camToWorld(cam.R, cam.t, [-hx, hy, z])
    ];
    const po = project(view, origin);
    const pc = corners.map((p) => project(view, p));
    if (!po || pc.some((p) => !p)) return;
    const o = { x: po.x * sx, y: po.y * sy, z: po.z };
    const c = pc.map((p) => ({ x: p.x * sx, y: p.y * sy, z: p.z }));
    const depth = this.splatDepth;
    const behind = (x, y, zz, bias = 0.06) => {
      if (!depth) return false;
      const ix = Math.min(depth.w - 1, Math.max(0, (x / depth.srcW * depth.w) | 0));
      const iy = Math.min(depth.h - 1, Math.max(0, (y / depth.srcH * depth.h) | 0));
      const sz = depth.data[iy * depth.stride + ix];
      return sz < 800 && zz > sz + bias;
    };
    let reveal = 1;
    if (real) {
      const shown = this.realScale ?? 1;
      const inside = !depth || (o.x >= 0 && o.y >= 0 && o.x < depth.srcW && o.y < depth.srcH);
      const hidden = inside && behind(o.x, o.y, o.z, shown < 0.5 ? 0.02 : 0.12);
      const now = performance.now();
      const dt = Math.min(0.05, this.realScaleAt ? (now - this.realScaleAt) / 1000 : 0);
      this.realScaleAt = now;
      const k = dt ? 1 - Math.exp(-dt / 0.12) : 1;
      reveal = shown + ((hidden ? 0 : 1) - shown) * k;
      this.realScale = reveal;
      if (reveal < 0.02) return;
    }
    if (real) {
      const H = this.canvas.height;
      const labelH = Math.max(22, Math.round(W * 0.022)) + 24;
      const pad = 20;
      const minXBound = pad, maxXBound = W - pad;
      const minYBound = pad + labelH, maxYBound = H * 0.68;
      const availW = Math.max(80, maxXBound - minXBound);
      const availH = Math.max(80, maxYBound - minYBound);
      let minX2 = Math.min(c[0].x, c[1].x, c[2].x, c[3].x);
      let maxX2 = Math.max(c[0].x, c[1].x, c[2].x, c[3].x);
      let minY2 = Math.min(c[0].y, c[1].y, c[2].y, c[3].y);
      let maxY2 = Math.max(c[0].y, c[1].y, c[2].y, c[3].y);
      const bw0 = Math.max(maxX2 - minX2, 1);
      const bh0 = Math.max(maxY2 - minY2, 1);
      const target = Math.min(Math.max(280, H * 0.34), availW, availH);
      const s = Math.min(target / Math.max(bw0, bh0), availW / bw0, availH / bh0) * reveal;
      const cx = (minX2 + maxX2) / 2, cy = (minY2 + maxY2) / 2;
      for (const p of c) {
        p.x = cx + (p.x - cx) * s;
        p.y = cy + (p.y - cy) * s;
      }
      o.x = cx + (o.x - cx) * s;
      o.y = cy + (o.y - cy) * s;
      minX2 = Math.min(c[0].x, c[1].x, c[2].x, c[3].x);
      maxX2 = Math.max(c[0].x, c[1].x, c[2].x, c[3].x);
      minY2 = Math.min(c[0].y, c[1].y, c[2].y, c[3].y);
      maxY2 = Math.max(c[0].y, c[1].y, c[2].y, c[3].y);
      let dx = 0, dy = 0;
      if (minX2 < minXBound) dx = minXBound - minX2;
      if (maxX2 + dx > maxXBound) dx += maxXBound - (maxX2 + dx);
      if (minY2 < minYBound) dy = minYBound - minY2;
      if (maxY2 + dy > maxYBound) dy += maxYBound - (maxY2 + dy);
      if (dx || dy) {
        for (const p of c) {
          p.x += dx;
          p.y += dy;
        }
        o.x += dx;
        o.y += dy;
      }
    }
    const strokeFrustum = (width, style, occlude) => {
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      const seg = (a, b) => {
        const N = occlude ? 6 : 1;
        ctx.beginPath();
        let on = false;
        for (let i = 0; i <= N; i++) {
          const t = i / N;
          const x = a.x + (b.x - a.x) * t;
          const y = a.y + (b.y - a.y) * t;
          const zz = a.z + (b.z - a.z) * t;
          if (occlude && behind(x, y, zz)) { on = false; continue; }
          if (!on) { ctx.moveTo(x, y); on = true; }
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      };
      seg(c[0], c[1]); seg(c[1], c[2]); seg(c[2], c[3]); seg(c[3], c[0]);
      for (const q of c) seg(o, q);
    };
    const fillQuad = (occlude) => {
      const N = occlude ? 4 : 1;
      const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t });
      const at = (u, v) => lerp(lerp(c[0], c[1], u), lerp(c[3], c[2], u), v);
      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          const u0 = i / N, u1 = (i + 1) / N, v0 = j / N, v1 = (j + 1) / N;
          const m = at((u0 + u1) / 2, (v0 + v1) / 2);
          if (occlude && behind(m.x, m.y, m.z)) continue;
          const q00 = at(u0, v0), q10 = at(u1, v0), q11 = at(u1, v1), q01 = at(u0, v1);
          ctx.beginPath();
          ctx.moveTo(q00.x, q00.y);
          ctx.lineTo(q10.x, q10.y);
          ctx.lineTo(q11.x, q11.y);
          ctx.lineTo(q01.x, q01.y);
          ctx.closePath();
          ctx.fill();
        }
      }
    };
    if (real) {
      ctx.fillStyle = "rgba(200, 241, 53, 0.16)";
      fillQuad(false);
      strokeFrustum(Math.max(10, W * 7e-3), "rgba(7, 8, 9, 0.92)", false);
      strokeFrustum(Math.max(5, W * 36e-4), "#c8f135", false);
    } else {
      ctx.fillStyle = "rgba(58, 74, 14, 0.2)";
      fillQuad(true);
      strokeFrustum(Math.max(1, W * 12e-4), "rgba(58, 74, 14, 0.95)", true);
    }
    const strip = this.strips.get(cam.name);
    const live = real && this.realVideo && this.realVideo.readyState >= 2 ? this.realVideo : null;
    const mid = {
      x: (c[0].x + c[1].x + c[2].x + c[3].x) / 4,
      y: (c[0].y + c[1].y + c[2].y + c[3].y) / 4
    };
    const midz = (c[0].z + c[1].z + c[2].z + c[3].z) / 4;
    const bw = Math.hypot(c[1].x - c[0].x, c[1].y - c[0].y);
    const bh = Math.hypot(c[3].x - c[0].x, c[3].y - c[0].y);
    const minX = Math.min(c[0].x, c[1].x, c[2].x, c[3].x);
    const maxX = Math.max(c[0].x, c[1].x, c[2].x, c[3].x);
    const minY = Math.min(c[0].y, c[1].y, c[2].y, c[3].y);
    const maxY = Math.max(c[0].y, c[1].y, c[2].y, c[3].y);
    if (live && live.videoWidth > 0 && bw > 4 && bh > 4) {
      drawImageQuad(ctx, live, live.videoWidth, live.videoHeight, c);
      strokeFrustum(Math.max(10, W * 7e-3), "rgba(7, 8, 9, 0.92)", false);
      strokeFrustum(Math.max(5, W * 36e-4), "#c8f135", false);
    } else if (!real && strip && strip.naturalWidth > 0 && bw > 4 && bh > 4 && !behind(mid.x, mid.y, midz)) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(c[0].x, c[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(c[i].x, c[i].y);
      ctx.closePath();
      ctx.clip();
      const src = stripFrame(strip.naturalWidth, frame);
      ctx.drawImage(strip, src.x, 0, src.w, strip.naturalHeight, minX, minY, maxX - minX, maxY - minY);
      ctx.restore();
      ctx.lineWidth = Math.max(1, W * 1e-3);
      ctx.strokeStyle = "rgba(232, 228, 220, 0.7)";
      ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
    } else if (!real && !behind(mid.x, mid.y, midz)) {
      const tw = Math.max(28, bw * 0.9);
      const th = tw * (cam.h / cam.w);
      ctx.fillStyle = "rgba(12, 14, 16, 0.75)";
      ctx.fillRect(mid.x - tw / 2, mid.y - th / 2, tw, th);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(58, 74, 14, 0.9)";
      ctx.strokeRect(mid.x - tw / 2, mid.y - th / 2, tw, th);
    }
    const label = real ? REAL_LABEL : SYNTH_LABEL;
    const size = real ? Math.max(22, Math.round(W * 0.022)) * reveal : Math.max(9, Math.round(W * 9e-3));
    ctx.font = `${real ? 700 : 500} ${size}px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const top = Math.min(c[0].y, c[1].y, c[2].y, c[3].y);
    const bottom = Math.max(c[0].y, c[1].y, c[2].y, c[3].y);
    const ly = real ? Math.max(8, top - size - 16) : bottom + 4;
    if (!real && behind(mid.x, ly, midz)) return;
    if (real) {
      const tw = ctx.measureText(label).width;
      const padX = 12, padY = 6;
      const half = tw / 2 + padX;
      const lx = Math.min(W - half - 8, Math.max(half + 8, mid.x));
      ctx.fillStyle = "#c8f135";
      ctx.fillRect(lx - half, ly - 2, tw + padX * 2, size + padY * 2);
      ctx.fillStyle = "#12140e";
      ctx.fillText(label, lx, ly);
      return;
    } else {
      ctx.save();
      ctx.shadowColor = "rgba(8, 10, 12, 0.8)";
      ctx.shadowBlur = 2;
      ctx.fillStyle = "rgba(232, 228, 220, 0.72)";
      ctx.fillText(label, mid.x, ly);
      ctx.restore();
    }
  }
};

// src/main.ts
var Q = new URLSearchParams(location.search);
var questQuery = Q.get("quest");
var QUEST = questQuery === "1" || (questQuery !== "0" && /OculusBrowser|\bQuest\b/i.test(navigator.userAgent));
var LIGHT = QUEST || /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && matchMedia("(pointer: coarse)").matches);
var qnum = (k, quest, dflt) => {
  const v = Number(Q.get(k) ?? (QUEST ? quest : dflt));
  return Number.isFinite(v) ? v : dflt;
};
var T_NAV = performance.now();
var LOOKAHEAD = Math.max(1, Math.round(qnum("lookahead", 1, 2)));
var UPLOAD_SLICE = 8 * 2 ** 20;
var marks = { script_start_ms: T_NAV };
var overlay = document.getElementById("stats");
var canvas = document.getElementById("c");
localizeDom();
var setting = (k) => {
  try {
    return localStorage.getItem("p5." + k);
  } catch {
    return null;
  }
};
var saveSetting = (k, v) => {
  try {
    localStorage.setItem("p5." + k, v);
  } catch {
  }
};
var sceneLabel = (n) => n.replace(/^scene_?/, "").replace(/\.json$/, "").replace(/_s2$/, "") || "scene";
var codecKnown = /* @__PURE__ */ new Map();
async function sceneList() {
  let names;
  try {
    names = await (await fetch("scenes.json", { cache: "no-store" })).json();
  } catch {
    return [];
  }
  if (!Array.isArray(names)) return [];
  await Promise.all(names.map(async (n) => {
    if (typeof n !== "string" || codecKnown.has(n)) return;
    try {
      const doc = await (await fetch(n, { cache: "no-store" })).json();
      codecKnown.set(n, doc?.codec === "s2");
    } catch {
      codecKnown.set(n, false);
    }
  }));
  return names.filter((n) => typeof n === "string" && codecKnown.get(n) === true).sort((a, b) => sceneLabel(a).localeCompare(sceneLabel(b)));
}
var onScenePicked = () => {
};
function fillScenes(names, current) {
  const sel = document.getElementById("scene");
  if (!names.length) return;
  const cur = names.includes(current) ? current : names[0];
  sel.replaceChildren();
  for (const n of names) sel.add(new Option(sceneLabel(n), n, false, n === cur));
  sel.hidden = false;
  sel.onchange = () => onScenePicked(sel.value);
}
function fail(msg) {
  overlay.textContent = tr("Fehler: ", "Error: ") + msg;
  window.__p5 = { error: msg };
  throw new Error(msg);
}
function lookAt(pos, target, up) {
  const sub2 = (a, b) => a.map((v, i) => v - b[i]);
  const cross2 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm2 = (a) => {
    const l = Math.hypot(...a);
    return a.map((v) => v / l);
  };
  const z = norm2(sub2(target, pos));
  const x = norm2(cross2(z, up));
  const y = cross2(z, x);
  return [x, y, z];
}
function camFromRt(R, t, fx, fy, cx, cy, w, h) {
  const view = new Float32Array(16);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) view[c * 4 + r] = R[r][c];
    view[12 + r] = t[r];
  }
  view[15] = 1;
  const campos = [0, 1, 2].map((c) => -(R[0][c] * t[0] + R[1][c] * t[1] + R[2][c] * t[2]));
  return { view, campos, fx, fy, cx, cy, width: w, height: h };
}
function orbitCam(sc, az, elev, distScale, w, h) {
  const o = sc.orbit, up = o.up, f = o.front;
  const s = [up[1] * f[2] - up[2] * f[1], up[2] * f[0] - up[0] * f[2], up[0] * f[1] - up[1] * f[0]];
  const ce = Math.cos(elev), se = Math.sin(elev), ca = Math.cos(az), sa = Math.sin(az);
  const d = o.dist * distScale;
  const pos = [0, 1, 2].map((i) => o.center[i] + d * (ce * (ca * f[i] + sa * s[i]) + se * up[i]));
  const R = lookAt(pos, o.center, up);
  const t = R.map((row) => -(row[0] * pos[0] + row[1] * pos[1] + row[2] * pos[2]));
  const fy = 0.5 * h / Math.tan((Q.has("fovy") ? Number(Q.get("fovy")) : o.fovy_deg) * Math.PI / 360);
  return camFromRt(R, t, fy, fy, w / 2, h / 2, w, h);
}
function questHint() {
  return tr(
    " \u2014 Quest: Quest Browser ab Version 146 n\xF6tig; in chrome://flags \u201EWebXR experiments\u201C (WebXR experimental features) aktivieren und den Browser neu starten. Die Seite muss \xFCber HTTPS kommen.",
    ' \u2014 Quest: needs Quest Browser 146 or newer; enable "WebXR experiments" (WebXR experimental features) in chrome://flags and restart the browser. The page must be served over HTTPS.'
  );
}
async function main() {
  if (!navigator.gpu) fail(tr("kein WebGPU (navigator.gpu fehlt)", "no WebGPU (navigator.gpu missing)") + (QUEST ? questHint() : ""));
  const xrProblem = await ImmersiveAR.diagnose();
  if (xrProblem) console.info(`MR: ${xrProblem}`);
  const xrReady = xrProblem === null;
  if (xrProblem && navigator.xr && (QUEST || xrProblem.includes("XRGPUBinding"))) {
    const hint = document.getElementById("xr-hint");
    if (hint) {
      hint.textContent = "MR: " + xrProblem + questHint();
      hint.hidden = false;
    }
  }
  const ask = (xr2) => {
    const p = navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
      ...xr2 ? { xrCompatible: true } : {}
    });
    return xr2 ? p.catch(() => null) : p;
  };
  let adapter = null;
  for (let i = 0; i < 10 && !adapter; i++) {
    if (xrReady) adapter = await ask(true);
    for (const pref of ["high-performance", "default", "low-power"]) {
      if (adapter) break;
      adapter = await navigator.gpu.requestAdapter(
        { ...pref === "default" ? {} : { powerPreference: pref }, ...xrReady ? { xrCompatible: true } : {} }
      ).catch(() => null);
    }
    if (!adapter) await new Promise((r) => setTimeout(r, 200));
  }
  if (!adapter) fail(tr("kein WebGPU-Adapter", "no WebGPU adapter"));
  const hasTs = !LIGHT && adapter.features.has("timestamp-query") && Q.get("ts") !== "0";
  const device = await adapter.requestDevice({
    requiredFeatures: hasTs ? ["timestamp-query"] : [],
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize }
  });
  let gpuError = null;
  device.lost.then((i) => {
    gpuError = `device lost (${i.reason}): ${i.message}`;
    console.error(gpuError);
  });
  device.addEventListener("uncapturederror", (e) => {
    const msg = e.error.message;
    console.error("WebGPU: " + msg);
    gpuError ??= msg;
  });
  marks.device_ms = performance.now();
  console.info(`p5: device ${marks.device_ms.toFixed(0)} ms`);
  const info = adapter.info ?? {};
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  const offscreen = Q.get("offscreen") === "1";
  if (!offscreen) ctx.configure({ device, format, alphaMode: "opaque" });
  let offTex = null;
  const outTexture = () => {
    if (!offscreen) return ctx.getCurrentTexture();
    if (!offTex || offTex.width !== canvas.width || offTex.height !== canvas.height) {
      offTex?.destroy();
      offTex = device.createTexture({ size: [canvas.width, canvas.height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    return offTex;
  };
  const dprCap = LIGHT ? (QUEST ? 1 : 1.25) : 2;
  const dpr = Number(Q.get("dpr") ?? Math.min(devicePixelRatio || 1, dprCap));
  const resize = () => {
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  };
  resize();
  addEventListener("resize", resize);
  const renderer = new Renderer(device, format, hasTs);
  renderer.sortEvery = Math.max(1, Math.round(Number(Q.get("sortevery") ?? (LIGHT ? 3 : 2))));
  renderer.bodyDepthOn = false;
  const camViz = new CamVizOverlay(document.body);
  camViz.simple = LIGHT;
  const camVizBox = document.getElementById("camviz");
  const setCamViz = (on) => {
    camViz.enabled = on;
    if (camVizBox) camVizBox.checked = on;
  };
  setCamViz(!!camVizBox?.checked);
  camVizBox?.addEventListener("change", () => setCamViz(camVizBox.checked));
  const REAL_SRC_FPS = 24400 / 841;
  const realVideo = document.createElement("video");
  realVideo.muted = true;
  realVideo.playsInline = true;
  realVideo.preload = LIGHT ? "none" : "auto";
  realVideo.hidden = true;
  if (!LIGHT) {
    realVideo.src = new URL("./real/angry_j.mp4", import.meta.url).href;
    document.body.appendChild(realVideo);
    camViz.realVideo = realVideo;
  }
  const shadowOn = (Q.get("shadow") ?? "1") !== "0";
  renderer.shadow = shadowOn;
  renderer.shadowStride = Math.max(1, Math.round(Number(Q.get("shadowstride") ?? (LIGHT ? 4 : 2))));
  renderer.shadowStrength = qnum("shadowstr", 0.9, 0.85);
  renderer.shadowFloorY = Number(Q.get("floory") ?? 0);
  const storedRaw = setting("shadowfade");
  const storedFade = storedRaw == null || storedRaw === "" ? NaN : Number(storedRaw);
  renderer.shadowFadeH = Number(Q.get("shadowfade") ?? (Number.isFinite(storedFade) ? storedFade : 1.15));
  renderer.shadowMaxRad = qnum("shadowrad", 0.034, 0.026);
  renderer.shadowSoft = qnum("shadowsoft", 0.55, 0.48);
  renderer.shadowLight = [0.55, 1, 0.35];
  const bgInput = document.getElementById("bg");
  const setBg = (hex) => {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return;
    const [r, g, b] = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16) / 255);
    renderer.setBackground(r, g, b);
    bgInput.value = "#" + m[1].toLowerCase();
  };
  setBg(Q.get("bg") ?? setting("bg") ?? (shadowOn ? "#d9d4cc" : "#000000"));
  bgInput.addEventListener("input", () => {
    setBg(bgInput.value);
    saveSetting("bg", bgInput.value);
  });
  const scenes = await sceneList();
  if (!scenes.length) fail(tr("keine S2-Szenen", "no S2 scenes"));
  const asked = Q.get("scene") ?? setting("scene");
  let sceneName = asked && scenes.includes(asked) ? asked : scenes[0];
  let sceneUrl = new URL(sceneName, location.href);
  fillScenes(scenes, sceneName);
  let scene = await (await fetch(sceneUrl)).json();
  marks.scene_ms = performance.now();
  let nsec = scene.sections.length;
  let total = scene.sections.reduce((a, s) => a + s.nframes, 0);
  const targetN = Number(Q.get("n") ?? 0);
  let worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const prefetchWorker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const decoded = /* @__PURE__ */ new Map();
  const gpu = /* @__PURE__ */ new Map();
  const requested = /* @__PURE__ */ new Set();
  const timings = [];
  const waiters = /* @__PURE__ */ new Map();
  const sceneDocs = /* @__PURE__ */ new Map();
  const clips = /* @__PURE__ */ new Map();
  let carousel = scenes;
  let prefetchName = null;
  const readyChecks = [];
  const ensureClip = (name, sc) => {
    let c = clips.get(name);
    if (!c) {
      c = { scene: sc, decoded: [], gpu: [] };
      clips.set(name, c);
    } else c.scene = sc;
    return c;
  };
  const clipReady = (name) => {
    const c = clips.get(name);
    return !!c && sectionsReady(c.scene.sections.length, c.gpu);
  };
  const warmClip = (c) => {
    for (let i = 0; i < c.scene.sections.length; i++) {
      const ps = c.decoded[i];
      if (!ps) continue;
      if (!c.gpu[i]) c.gpu[i] = renderer.upload(ps);
      if (c.gpu[i].pending) renderer.pump(c.gpu[i], UPLOAD_SLICE);
    }
  };
  const pumpClips = () => {
    for (const [name, c] of clips) if (name !== sceneName) warmClip(c);
    const q = readyChecks.splice(0);
    for (const f of q) f();
  };
  sceneDocs.set(sceneName, scene);
  ensureClip(sceneName, scene);
  const onWorker = (e) => {
    const { seq, ps, timing, error, clip } = e.data;
    if (error) fail(`${tr("Abschnitt", "Section")} ${seq}: ${error}`);
    if (clip && clip !== sceneName) return;
    timing.seq = seq;
    timing.index = seq % nsec;
    timing.recv_ms = performance.now();
    timings.push(timing);
    decoded.set(seq, { ps, timing });
    ensureClip(sceneName, scene).decoded[seq % nsec] = ps;
    maxDecoded = Math.max(maxDecoded, seq);
    (waiters.get(seq) ?? []).forEach((f) => f());
    waiters.delete(seq);
  };
  worker.onmessage = onWorker;
  prefetchWorker.onmessage = (e) => {
    const { seq, ps, error, clip } = e.data;
    if (error) {
      console.warn("p5: prefetch", clip, error);
      return;
    }
    if (!ps || typeof clip !== "string") return;
    const c = clips.get(clip);
    if (!c || c.decoded[seq]) return;
    c.decoded[seq] = ps;
  };
  const residentMiB = Number(Q.get("resident") ?? (QUEST ? 384 : LIGHT ? 512 : 1024));
  let resident = residentMiB > 0 ? /* @__PURE__ */ new Map() : null;
  let residentBytes = 0;
  const residentFull = () => !!resident && resident.size === nsec;
  const isResident = (seq, g) => !!resident && resident.get(seq % nsec) === g;
  const keep = (seq, g) => {
    ensureClip(sceneName, scene).gpu[seq % nsec] = g;
    if (!resident || resident.has(seq % nsec)) return;
    resident.set(seq % nsec, g);
    residentBytes += g.bytes;
    if (residentBytes <= residentMiB * 2 ** 20) return;
    const live2 = new Set(gpu.values());
    for (const r of resident.values()) if (!live2.has(r)) renderer.release(r);
    resident = null;
    console.info(`p5: clip exceeds ${residentMiB} MiB, streaming`);
  };
  let maxRequested = -1;
  const request = (seq) => {
    if (requested.has(seq) || residentFull()) return;
    for (let s = maxRequested + 1; s < seq; s++) request(s);
    maxRequested = Math.max(maxRequested, seq);
    requested.add(seq);
    worker.postMessage({ seq, url: new URL(scene.sections[seq % nsec].url, sceneUrl).href, targetN, clip: sceneName });
  };
  let maxDecoded = -1, curSeq = 0;
  const whenDecoded = (seq) => new Promise((res) => {
    request(seq);
    if (seq <= maxDecoded || residentFull()) res();
    else waiters.set(seq, [...waiters.get(seq) ?? [], res]);
  });
  const ensureGpu = (seq, now = true) => {
    let g = gpu.get(seq) ?? resident?.get(seq % nsec) ?? null;
    if (!g) {
      const d = decoded.get(seq);
      if (!d) return null;
      console.info(`p5: upload seq ${seq} n=${d.ps.n} at ${performance.now().toFixed(0)} ms`);
      g = renderer.upload(d.ps);
    }
    gpu.set(seq, g);
    if (g.pending) {
      const t0 = performance.now();
      renderer.pump(g, now ? Infinity : UPLOAD_SLICE);
      const d = decoded.get(seq);
      if (d) d.timing.upload_ms = (d.timing.upload_ms ?? 0) + performance.now() - t0;
      if (g.pending) return null;
      keep(seq, g);
    }
    return g;
  };
  const locate = (playFrames2) => {
    const loop = Math.floor(playFrames2 / total);
    let f = playFrames2 - loop * total;
    for (let i = 0; i < nsec; i++) {
      const s = scene.sections[i];
      if (f < s.nframes || i === nsec - 1) return { seq: loop * nsec + i, t: s.g0 + Math.min(f, s.nframes - 1e-4) };
      f -= s.nframes;
    }
    throw new Error("unreachable");
  };
  const nextClip = () => stepName(carousel, sceneName, 1);
  const startPrefetch = (name) => {
    if (!name || name === prefetchName) return;
    prefetchName = name;
    void (async () => {
      const sc = await loadDoc(name);
      if (prefetchName !== name) return;
      const c = ensureClip(name, sc);
      if (sectionsReady(sc.sections.length, c.gpu)) return;
      const base = new URL(name, location.href);
      for (let i = 0; i < sc.sections.length; i++) {
        if (c.decoded[i]) continue;
        prefetchWorker.postMessage({ seq: i, url: new URL(sc.sections[i].url, base).href, targetN, clip: name });
      }
    })();
  };
  const loadDoc = async (name) => {
    const hit = sceneDocs.get(name);
    if (hit) return hit;
    const sc = await (await fetch(new URL(name, location.href), { cache: "no-store" })).json();
    sceneDocs.set(name, sc);
    return sc;
  };
  const housekeeping = (seq) => {
    curSeq = seq;
    for (let s = seq; s <= seq + LOOKAHEAD; s++) request(s);
    ensureGpu(seq + 1, false);
    for (const [k, g] of gpu) if (k < seq) {
      if (!isResident(k, g)) renderer.release(g);
      gpu.delete(k);
    }
    for (const k of decoded.keys()) if (k < seq) decoded.delete(k);
    pumpClips();
    if (residentFull()) startPrefetch(nextClip());
  };
  const gradeKey = "grades";
  const storedGrades = () => {
    try {
      const v = JSON.parse(setting(gradeKey) ?? "{}");
      return v && typeof v === "object" ? v : {};
    } catch {
      return {};
    }
  };
  const gradeFor = (name) => sanitizeGrade(storedGrades()[name] ?? NEUTRAL);
  const saveGrade = (name, g) => {
    const all = storedGrades();
    all[name] = { ...g };
    saveSetting(gradeKey, JSON.stringify(all));
  };
  const live = gradeFor(sceneName);
  let holding = false;
  let switching = false;
  const gradeBox = document.getElementById("grade-box");
  const holdBtn = document.getElementById("hold");
  const recolorBox = document.getElementById("recolor");
  const recolorInfo = document.getElementById("recolor-info");
  const gradeRows = document.getElementById("grade-rows");
  const gradeInputs = /* @__PURE__ */ new Map();
  const showGrade = () => {
    renderer.setGrade(isNeutral(live) ? null : live);
    for (const s of GRADE_FIELDS) {
      const io = gradeInputs.get(s.field);
      if (!io) continue;
      io.range.value = String(live[s.field]);
      io.num.value = String(live[s.field]);
    }
    if (holding && recolorInfo) recolorInfo.textContent = describeGrade(live);
  };
  const rememberGrade = () => {
    saveGrade(sceneName, live);
    showGrade();
    if (desk) desk.dirty = true;
  };
  for (const s of GRADE_FIELDS) {
    const row = document.createElement("label");
    row.title = s.tip;
    const name = document.createElement("span");
    name.textContent = s.label;
    const range = document.createElement("input");
    range.type = "range";
    range.min = String(s.min);
    range.max = String(s.max);
    range.step = String(s.step);
    const num = document.createElement("input");
    num.type = "number";
    num.min = String(s.min);
    num.max = String(s.max);
    num.step = String(s.step);
    const apply = (v) => {
      if (!Number.isFinite(v)) return;
      live[s.field] = Math.min(s.max, Math.max(s.min, v));
      if (recolorBox) recolorBox.checked = false;
      rememberGrade();
    };
    range.addEventListener("input", () => apply(Number(range.value)));
    num.addEventListener("change", () => apply(Number(num.value)));
    row.append(name, range, num);
    gradeRows.append(row);
    gradeInputs.set(s.field, { range, num });
  }
  const showHoldUi = () => {
    gradeBox.hidden = !holding;
    gradeBox.open = holding;
    holdBtn.setAttribute("aria-expanded", holding ? "true" : "false");
    if (desk) {
      desk.take = sceneLabel(sceneName);
      desk.editing = holding;
      desk.dirty = true;
    }
    if (!holding && recolorInfo) recolorInfo.textContent = "";
    showGrade();
  };
  let desk = null;
  let deskGpu = null;
  const ensureDesk = () => {
    if (desk) return desk;
    desk = new GradeDesk(live);
    desk.onChange = () => {
      recolorBox && (recolorBox.checked = false);
      rememberGrade();
    };
    desk.onButton = (id) => {
      if (id === "reset") {
        Object.assign(live, NEUTRAL);
        if (recolorBox) recolorBox.checked = false;
        rememberGrade();
      } else if (id === "auto") {
        if (recolorBox) recolorBox.checked = true;
        if (xr) void autoGrade(xr);
      } else if (id === "prev" || id === "next") void stepVariant(id === "next" ? 1 : -1);
      else if (id === "hold") {
        holding = !holding;
        showHoldUi();
      }
    };
    desk.shadowFade = renderer.shadowFadeH;
    desk.onShadowChange = (v) => {
      renderer.shadowFadeH = v;
      saveSetting("shadowfade", String(v));
    };
    deskGpu = new DeskGPU(device);
    if (handMeshes) deskGpu.setHandMeshes(handMeshes[0], handMeshes[1]);
    return desk;
  };
  let handMeshes = null;
  void Promise.all([loadHandMesh("hands/left.glb"), loadHandMesh("hands/right.glb")]).then(([l, r]) => {
    handMeshes = [l, r];
    deskGpu?.setHandMeshes(l, r);
    console.info("MR: hand occluder meshes ready");
  }).catch((e) => console.error("MR: hand meshes failed \u2014 desk will not be occluded by hands", e));
  const dropGpu = (c) => {
    for (let i = 0; i < c.gpu.length; i++) {
      if (!c.gpu[i]) continue;
      renderer.release(c.gpu[i]);
      c.gpu[i] = void 0;
    }
  };
  const adoptClip = (name, hold) => {
    const c = clips.get(name);
    if (!c || !sectionsReady(c.scene.sections.length, c.gpu)) return false;
    sceneName = name;
    scene = c.scene;
    sceneUrl = new URL(name, location.href);
    nsec = scene.sections.length;
    total = scene.sections.reduce((a, s) => a + s.nframes, 0);
    playFrames = 0;
    lastNow = 0;
    gpu.clear();
    decoded.clear();
    requested.clear();
    waiters.clear();
    resident = residentMiB > 0 ? /* @__PURE__ */ new Map() : null;
    residentBytes = 0;
    for (let i = 0; i < nsec; i++) {
      const g = c.gpu[i];
      gpu.set(i, g);
      decoded.set(i, { ps: c.decoded[i], timing: {} });
      requested.add(i);
      if (resident) {
        resident.set(i, g);
        residentBytes += g.bytes;
      }
    }
    maxRequested = nsec - 1;
    maxDecoded = nsec - 1;
    curSeq = 0;
    Object.assign(live, gradeFor(name));
    holding = hold;
    const sel = document.getElementById("scene");
    if (Array.from(sel.options).every((o) => o.value !== name)) sel.add(new Option(sceneLabel(name), name));
    sel.value = name;
    saveSetting("scene", name);
    showHoldUi();
    const next = nextClip();
    const keepNames = retainClips(name, next);
    for (const [k, cc] of clips) if (!keepNames.has(k)) dropGpu(cc);
    if (next && next !== name) startPrefetch(next);
    return true;
  };
  const loadScene = async (name, hold) => {
    if (name === sceneName && clipReady(name)) {
      holding = hold;
      showHoldUi();
      return;
    }
    saveGrade(sceneName, live);
    const next = await loadDoc(name);
    if (!next.sections?.length) return;
    ensureClip(name, next);
    if (adoptClip(name, hold)) return;
    startPrefetch(name);
    await new Promise((res) => {
      const check = () => {
        if (clipReady(name)) res();
        else readyChecks.push(check);
      };
      check();
    });
    adoptClip(name, hold);
  };
  const stepVariant = async (dir) => {
    const name = stepName(carousel, sceneName, dir);
    if (!name) return;
    await loadScene(name, holding);
  };
  const advance = async () => {
    if (switching || holding) return;
    switching = true;
    try {
      await stepVariant(1);
    } finally {
      switching = false;
    }
  };
  onScenePicked = (name) => {
    void loadScene(name, true);
  };
  holdBtn.addEventListener("click", () => {
    holding = !holding;
    showHoldUi();
  });
  showHoldUi();
  const bumpPlay = (dt) => {
    if (!playing || switching || total <= 0) return;
    const prev = Math.floor(playFrames / total);
    playFrames += dt * scene.fps;
    if (!holding && Math.floor(playFrames / total) !== prev) void advance();
  };
  const fixedT = Q.has("t") ? Number(Q.get("t")) : null;
  let playing = !Q.has("pause") && fixedT === null;
  let playFrames = fixedT ?? 0;
  let lastNow = 0, stalls = 0, frames = 0, firstFrameDone = false, inflight = 0, gpuBound = 0;
  const ORBIT_ZOOM = 1.6;
  let az = Number(Q.get("az") ?? 0) * Math.PI / 180, elev = scene.orbit.elev_deg * Math.PI / 180;
  let distScale = Number(Q.get("zoom") ?? (document.documentElement.dataset.orbit === "1" ? ORBIT_ZOOM : 1));
  const ORBIT_PERIOD = 16, ORBIT_ELEV = 8 * Math.PI / 180, ORBIT_IDLE_MS = 15e3;
  let elevCenter = elev, nodPhase = 0;
  let orbiting = document.documentElement.dataset.orbit === "1" && !Q.has("az") && !Q.has("elev") && !Q.has("zoom") && !Q.has("testcam");
  let resumeAfterIdle = false;
  let lastInput = performance.now();
  let drag = null;
  const orbitBox = document.getElementById("orbit");
  const noteInput = () => {
    lastInput = performance.now();
  };
  const stopOrbit = (resume = false) => {
    orbiting = false;
    resumeAfterIdle = resume;
    if (orbitBox) orbitBox.checked = false;
    if (resume) noteInput();
  };
  const startOrbit = () => {
    elevCenter = elev - ORBIT_ELEV * Math.sin(nodPhase / 2);
    orbiting = true;
    resumeAfterIdle = false;
    if (orbitBox) orbitBox.checked = true;
  };
  if (orbitBox) orbitBox.checked = orbiting;
  orbitBox?.addEventListener("change", () => {
    if (orbitBox.checked) startOrbit();
    else {
      orbiting = false;
      resumeAfterIdle = false;
    }
  });
  window.addEventListener("pointerdown", noteInput);
  window.addEventListener("pointerup", noteInput);
  window.addEventListener("pointermove", (e) => {
    if (e.buttons) noteInput();
  });
  window.addEventListener("wheel", noteInput, { passive: true });
  window.addEventListener("keydown", noteInput);
  const fpsWin = [];
  const testIdx = Q.has("testcam") ? Number(Q.get("testcam")) : -1;
  const currentCam = () => {
    if (testIdx >= 0) {
      const c = scene.testcams[testIdx];
      return camFromRt(c.R, c.t, c.fx, c.fy, c.cx, c.cy, c.w, c.h);
    }
    return orbitCam(scene, az, elev, distScale, canvas.width, canvas.height);
  };
  let rafOn = true;
  let hud = () => {
  };
  const markFps = (now) => {
    fpsWin.push(now);
    while (fpsWin.length && fpsWin[0] < now - 1e3) fpsWin.shift();
  };
  const statusText = (seq, t, n) => {
    const tm = renderer.lastTimings;
    const mode = holding ? tr("halten", "hold") : tr("Karussell", "carousel");
    return `${sceneLabel(sceneName)}${mode ? ` \xB7 ${mode}` : ""} \xB7 ${fpsWin.length} fps \xB7 ${tr("Abschnitt", "section")} ${seq % nsec} t=${t.toFixed(2)} \xB7 ${n} splats (${renderer.lastVisible} ${tr("sichtbar", "visible")})` + (tm ? ` \xB7 GPU ${tm.gpu_total.toFixed(2)} ms` : "") + ` \xB7 ${(renderer.gpuBytes() / 2 ** 20).toFixed(0)} MiB`;
  };
  const tick = (now) => {
    if (gpuError) {
      overlay.textContent = tr("GPU-Fehler: ", "GPU error: ") + gpuError;
      window.__p5.error = gpuError;
      return;
    }
    if (!rafOn) return;
    const dt = lastNow ? (now - lastNow) / 1e3 : 0;
    lastNow = now;
    if (!orbiting && resumeAfterIdle && testIdx < 0 && !drag && now - lastInput >= ORBIT_IDLE_MS) startOrbit();
    if (orbiting) {
      const dPhase = 2 * Math.PI * Math.min(dt, 0.1) / ORBIT_PERIOD;
      az += dPhase;
      nodPhase += dPhase;
      elev = elevCenter + ORBIT_ELEV * Math.sin(nodPhase / 2);
    }
    let { seq, t } = locate(playFrames);
    housekeeping(seq);
    const g = ensureGpu(seq);
    if (g && inflight >= 2) {
      gpuBound++;
      bumpPlay(dt);
    } else if (g) {
      renderer.bodyDepthOn = !LIGHT && camViz.enabled;
      renderer.frame(g, t, currentCam(), outTexture(), { timing: hasTs && frames % 20 === 0, readVisible: frames % (LIGHT ? 30 : 10) === 0 });
      inflight++;
      device.queue.onSubmittedWorkDone().then(() => {
        inflight--;
      });
      frames++;
      if (!firstFrameDone) {
        firstFrameDone = true;
        console.info(`p5: first frame submitted ${performance.now().toFixed(0)} ms`);
        device.queue.onSubmittedWorkDone().then(() => {
          marks.first_frame_ms = performance.now();
          console.info(`p5: first frame done ${marks.first_frame_ms.toFixed(0)} ms`);
        });
      }
      bumpPlay(dt);
    } else if (playing) {
      stalls++;
    }
    markFps(now);
    hud = () => {
      overlay.textContent = statusText(seq, t, g ? g.ps.n : 0);
    };
    if (frames % 15 === 0) hud();
    camViz.resize(canvas.clientWidth, canvas.clientHeight, canvas.width, canvas.height);
    const take = sceneLabel(sceneName);
    camViz.realActive = !LIGHT;
    camViz.bind(sceneName, take, scene.testcams);
    if (!camViz.realActive || !camViz.enabled) {
      if (!realVideo.paused) realVideo.pause();
    } else if (realVideo.readyState >= 1) {
      const dur = realVideo.duration;
      const target = Math.max(0, Math.min(Number.isFinite(dur) ? Math.max(0, dur - 0.04) : 1e9, t / REAL_SRC_FPS));
      if (!Number.isFinite(realVideo.currentTime) || Math.abs(realVideo.currentTime - target) > 0.08 || !playing)
        realVideo.currentTime = target;
      if (playing) {
        realVideo.playbackRate = scene.fps / REAL_SRC_FPS;
        if (realVideo.paused) void realVideo.play().catch(() => {
        });
      } else if (!realVideo.paused) realVideo.pause();
    }
    if (camViz.enabled) {
      camViz.splatDepth = renderer.splatDepth ?? null;
      const oc = scene.orbit.center;
      camViz.draw(currentCam(), scene.testcams, { center: [oc[0], oc[1], oc[2]], radius: 0.45, halfHeight: 0.95 }, t);
    } else {
      camViz.draw(currentCam(), []);
    }
    void t;
    void seq;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  const touches = new Map();
  let pinch = 0;
  const touchSpan = () => {
    const p = [...touches.values()];
    return Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic or already released */ }
    if (touches.size >= 2) {
      drag = null;
      pinch = touchSpan();
    } else {
      drag = { x: e.clientX, y: e.clientY };
    }
  });
  const endTouch = (e) => {
    touches.delete(e.pointerId);
    if (touches.size >= 2) {
      drag = null;
      pinch = touchSpan();
    } else if (touches.size === 1) {
      const p = [...touches.values()][0];
      drag = { x: p.x, y: p.y };
      pinch = 0;
    } else {
      drag = null;
      pinch = 0;
    }
  };
  canvas.addEventListener("pointerup", endTouch);
  canvas.addEventListener("pointercancel", endTouch);
  canvas.addEventListener("pointermove", (e) => {
    if (!touches.has(e.pointerId)) return;
    touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (touches.size >= 2) {
      const d = touchSpan();
      if (pinch > 0 && d > 0) distScale *= pinch / d;
      pinch = d;
      drag = null;
      return;
    }
    if (!drag) return;
    stopOrbit(true);
    az -= (e.clientX - drag.x) * 0.01;
    elev = Math.max(-1.4, Math.min(1.4, elev + (e.clientY - drag.y) * 0.01));
    drag = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener("wheel", (e) => {
    distScale *= Math.exp(e.deltaY * 1e-3);
    e.preventDefault();
  }, { passive: false });
  document.getElementById("play").addEventListener("click", () => {
    playing = !playing;
  });
  const xrScaleQ = qnum("xrscale", 1, 1);
  const xrRenderScale = Math.min(1, Math.max(0.3, qnum("xrrs", 0.7, 1)));
  const xrLayerScale = Number.isFinite(xrScaleQ) && xrScaleQ > 0 ? xrScaleQ : 1;
  const xrFovQ = Q.get("foveation");
  const xrOpts = {
    foveation: xrFovQ !== null && Number.isFinite(Number(xrFovQ)) ? Math.min(1, Math.max(0, Number(xrFovQ))) : void 0,
    frameRate: qnum("hz", 72, 0) || void 0
  };
  const hudOn = qnum("hud", 0, 0) > 0;
  const xrDistQ = Number(Q.get("xrdist") ?? 1.1);
  const xrDist = Number.isFinite(xrDistQ) && xrDistQ > 0 ? xrDistQ : 1.1;
  const xrButton = document.getElementById("xr-ar");
  const recolorQ = Number(Q.get("recolor") ?? 0.6);
  const recolorStrength = Number.isFinite(recolorQ) && recolorQ >= 0 ? recolorQ : 0.6;
  let xr = null;
  const autoGrade = async (stage) => {
    if (recolorInfo) recolorInfo.textContent = tr("Kamera \u2026", "Camera \u2026");
    let room;
    try {
      room = await captureRoom();
    } catch (e) {
      console.warn("MR: recolor", e);
      if (recolorInfo) recolorInfo.textContent = tr("Kamera: ", "Camera: ") + (e instanceof Error ? e.message : e);
      return;
    }
    if (xr !== stage || !recolorBox?.checked) {
      showGrade();
      return;
    }
    const px = await renderer.readTargetRGBA(4);
    Object.assign(live, estimateGrade(room, px ? imageStats(px, 1, true) : null, recolorStrength));
    rememberGrade();
    console.info(`MR: recolor ${describeGrade(live)}`);
  };
  recolorBox?.addEventListener("change", () => {
    if (recolorBox.checked) {
      if (xr) void autoGrade(xr);
      return;
    }
    showGrade();
  });
  let placed = null;
  let xrFrames = 0;
  let xrDrawError = false;
  const onXR = (time, frame) => {
    const stage = xr;
    if (!stage) return;
    stage.session.requestAnimationFrame(onXR);
    if (gpuError) {
      overlay.textContent = tr("GPU-Fehler: ", "GPU error: ") + gpuError;
      window.__p5.error = gpuError;
      return;
    }
    const dt = lastNow ? (time - lastNow) / 1e3 : 0;
    lastNow = time;
    const { seq, t } = locate(playFrames);
    housekeeping(seq);
    const g = ensureGpu(seq);
    const pose = frame.getViewerPose(stage.refSpace);
    xrFrames++;
    if (pose && !placed?.locked) {
      const o = scene.orbit;
      const center = o.center;
      const f = o.front;
      const cam = [center[0] + f[0] * o.dist, center[1] + f[1] * o.dist, center[2] + f[2] * o.dist];
      const hm = pose.transform.matrix;
      const next = place(cam, center, [o.up[0], o.up[1], o.up[2]], [hm[12], hm[13], hm[14]], [-hm[8], -hm[9], -hm[10]], xrDist);
      if (next) {
        next.locked = !pose.emulatedPosition || xrFrames > 30;
        placed = next;
      }
    }
    if (pose) {
      const hm = pose.transform.matrix;
      const head2 = [hm[12], hm[13], hm[14]];
      const d = ensureDesk();
      d.take = sceneLabel(sceneName);
      d.editing = holding;
      d.head = head2;
      if (!d.placed) d.place(head2, [-hm[8], -hm[9], -hm[10]]);
    }
    const deskToggle = document.getElementById("desk");
    const deskOn = !!desk?.placed && (!deskToggle || deskToggle.checked);
    if (placed?.locked) stage.pinch(frame, placed, time, deskOn ? desk : null);
    if (g && inflight >= 2) {
      gpuBound++;
      bumpPlay(dt);
    } else if (g && pose && placed) {
      try {
        const eyes = stage.eyes(frame, pose, placed);
        if (eyes.some((e) => e.occ)) xrLog.occ++;
        if (eyes.length) {
          if (hudOn) renderer.setHud(fpsWin.length, g.ps.n / 1e3, Math.round(eyes[0].viewport.height * 0.03));
          if (deskOn && desk?.dirty) {
            deskGpu.upload(desk.mesh());
            desk.dirty = false;
          }
          if (deskGpu && deskOn) {
            const [l, r] = readHandPoses(frame, stage.session, stage.refSpace);
            deskGpu.setHandPoses(l, r);
          } else {
            deskGpu?.setHandPoses(null, null);
          }
          renderer.frameEyes(g, t, eyes, deskOn ? (enc, eye, i) => deskGpu.draw(enc, eye, i, desk.model) : void 0);
          inflight++;
          device.queue.onSubmittedWorkDone().then(() => {
            inflight--;
          });
          frames++;
          if (!firstFrameDone) {
            firstFrameDone = true;
            console.info(`p5: first frame submitted ${performance.now().toFixed(0)} ms`);
            device.queue.onSubmittedWorkDone().then(() => {
              marks.first_frame_ms = performance.now();
              console.info(`p5: first frame done ${marks.first_frame_ms.toFixed(0)} ms`);
            });
          }
        }
      } catch (e) {
        if (!xrDrawError) {
          xrDrawError = true;
          console.error("MR", e);
        }
      }
      bumpPlay(dt);
    } else if (playing) {
      stalls++;
    }
    markFps(time);
    hud = () => {
      overlay.textContent = statusText(seq, t, g ? g.ps.n : 0);
    };
    if (frames % 15 === 0) hud();
    if (time - xrLog.at >= 5e3) {
      const last = timings[timings.length - 1];
      console.info(`MR: ${fpsWin.length} fps, ${gpuBound - xrLog.busy} frames skipped (GPU busy), ${stalls - xrLog.stalls} stalls (section not ready)` + (last ? `, last decode ${last.decode_ms.toFixed(0)} ms seq ${last.seq}` : "") + `, ${resident ? `resident ${resident.size}/${nsec}` : "streaming"}, depth occlusion ${stage.depthSensing ? stage.occlude ? `${xrLog.occ} frames` : "off" : "unavailable"}`);
      xrLog.at = time;
      xrLog.busy = gpuBound;
      xrLog.stalls = stalls;
      xrLog.occ = 0;
    }
  };
  const xrLog = { at: 0, busy: 0, stalls: 0, occ: 0 };
  const occBox = document.getElementById("occ");
  const occDebugBox = document.getElementById("occ-debug");
  occBox.checked = setting("occlude") !== "0";
  occDebugBox.checked = setting("occdebug") === "1";
  occBox.addEventListener("change", () => {
    saveSetting("occlude", occBox.checked ? "1" : "0");
    if (xr) xr.occlude = occBox.checked;
  });
  occDebugBox.addEventListener("change", () => {
    saveSetting("occdebug", occDebugBox.checked ? "1" : "0");
    if (xr) xr.occDebug = occDebugBox.checked;
  });
  if (navigator.xr) {
    xrButton.hidden = false;
    const occWrap = document.getElementById("occ-box");
    const occDebugWrap = document.getElementById("occ-debug-box");
    const deskWrap = document.getElementById("desk-box");
    if (occWrap) occWrap.hidden = !xrReady;
    if (occDebugWrap) occDebugWrap.hidden = !xrReady;
    if (deskWrap) deskWrap.hidden = !xrReady;
    const showMrProblem = (msg) => {
      const hint = document.getElementById("xr-hint");
      if (!hint) return;
      hint.hidden = false;
      hint.textContent = "MR: " + msg;
    };
    xrButton.addEventListener("click", () => {
      if (!xrReady) {
        showMrProblem((xrProblem || "unavailable") + questHint());
        return;
      }
      if (xr) {
        void xr.session.end();
        return;
      }
      xrButton.disabled = true;
      xrButton.textContent = "\u2026";
      void (async () => {
        try {
          const stage = await ImmersiveAR.start(device, xrLayerScale, xrOpts);
          stage.occlude = occBox.checked;
          stage.occDebug = occDebugBox.checked;
          stage.renderScale = xrRenderScale;
          const sortOutside = renderer.sortEvery;
          renderer.sortEvery = Math.min(sortOutside, 2);
          stage.session.addEventListener("end", () => {
            xr = null;
            placed = null;
            xrFrames = 0;
            rafOn = true;
            lastNow = 0;
            renderer.sortEvery = sortOutside;
            if (desk) desk.placed = false;
            showGrade();
            xrButton.textContent = "MR";
            camViz.canvas.hidden = false;
            requestAnimationFrame(tick);
          });
          placed = null;
          xrFrames = 0;
          xrDrawError = false;
          stopOrbit();
          camViz.draw(currentCam(), []);
          camViz.canvas.hidden = true;
          rafOn = false;
          lastNow = 0;
          xr = stage;
          xrButton.textContent = "Exit";
          stage.onDoublePinch = () => {
            holding = !holding;
            const deskEl = document.getElementById("desk");
            if (holding && deskEl) {
              deskEl.checked = true;
              ensureDesk();
            }
            showHoldUi();
          };
          stage.session.requestAnimationFrame(onXR);
          if (holding && recolorBox?.checked) void autoGrade(stage);
        } catch (e) {
          console.error("MR", e);
          showMrProblem(e instanceof Error ? e.message : String(e));
          xrButton.textContent = "MR";
        } finally {
          xrButton.disabled = false;
        }
      })();
    });
  }
  const allLoaded = async () => {
    const c = curSeq;
    for (let s = c; s <= c + LOOKAHEAD; s++) await whenDecoded(s);
  };
  const p5 = {
    ready: true,
    info: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, timestamps: hasTs, format },
    marks,
    timings,
    scene: { nsec, total, fps: scene.fps, targetN },
    config: {
      quest: QUEST,
      lookahead: LOOKAHEAD,
      resident_mib: residentMiB,
      xrscale: xrLayerScale,
      xrrs: xrRenderScale,
      sortevery: renderer.sortEvery,
      xr: xrOpts,
      hud: hudOn,
      shadow: renderer.shadow,
      shadow_stride: renderer.shadowStride,
      shadow_floor_y: renderer.shadowFloorY
    },
    /** P40: the XR eye blit with the in-headset readout, rendered offscreen (no XR session needed) */
    xrBlitCheck: (w, h, fps, k) => renderer.debugXrBlit(w, h, fps, k, Math.round(h * 0.03)),
    state: () => ({
      frames,
      stalls,
      playFrames,
      visible: renderer.lastVisible,
      gpu_bytes: renderer.gpuBytes(),
      gpu_bytes_peak: renderer.peakBytes,
      gpu_bytes_by: Object.fromEntries(renderer.bytes),
      sort_dispatches: renderer.sortDispatches(),
      canvas: [canvas.width, canvas.height]
    }),
    setView: (o) => {
      if (o.az !== void 0 || o.elev !== void 0 || o.zoom !== void 0) stopOrbit();
      if (o.az !== void 0) az = o.az * Math.PI / 180;
      if (o.elev !== void 0) elev = o.elev * Math.PI / 180;
      if (o.zoom !== void 0) distScale = o.zoom;
      if (o.t !== void 0) {
        let pf = Math.floor(playFrames / total) * total + o.t % total;
        if (pf < playFrames) pf += total;
        playFrames = pf;
      }
      if (o.playing !== void 0) playing = o.playing;
    },
    /** rAF-paced playback for `seconds`; returns the display-paced frame rate. */
    async rafFps(seconds) {
      await allLoaded();
      playing = true;
      const f0 = frames, t0 = performance.now(), s0 = stalls, b0 = gpuBound;
      renderer.timingsLog.length = 0;
      await new Promise((r) => setTimeout(r, seconds * 1e3));
      const el = (performance.now() - t0) / 1e3;
      return { fps: (frames - f0) / el, frames: frames - f0, seconds: el, stalls: stalls - s0, skipped_gpu_busy: gpuBound - b0, gpu_timings: summarize(renderer.timingsLog) };
    },
    /** frames back-to-back without vsync (max 2 in flight); measures what the GPU sustains. */
    async uncapped(nFrames, timing, maxInflight = 3) {
      await allLoaded();
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const off = device.createTexture({ size: [canvas.width, canvas.height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      renderer.timingsLog.length = 0;
      const inflight2 = [];
      const t0 = performance.now();
      let pf = playFrames;
      for (let i = 0; i < nFrames; i++) {
        const { seq, t } = locate(pf);
        housekeeping(seq);
        let g = ensureGpu(seq);
        if (!g) {
          await whenDecoded(seq);
          g = ensureGpu(seq);
        }
        if (!g) throw new Error(`section seq ${seq} not available`);
        renderer.frame(g, t, currentCam(), off, { timing });
        inflight2.push(device.queue.onSubmittedWorkDone());
        if (inflight2.length >= maxInflight) await inflight2.shift();
        pf += scene.fps / 60;
      }
      await Promise.all(inflight2);
      const el = (performance.now() - t0) / 1e3;
      await new Promise((r) => setTimeout(r, 100));
      off.destroy();
      playFrames = pf;
      rafOn = true;
      lastNow = 0;
      requestAnimationFrame(tick);
      return { fps: nFrames / el, frames: nFrames, seconds: el, max_inflight: maxInflight, gpu_timings: summarize(renderer.timingsLog) };
    },
    /** offscreen mode only: copy the current float target into a 2D canvas over the WebGPU canvas (screenshots). */
    async snapshot() {
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      hud();
      const img = await renderer.readTarget();
      const c2 = document.createElement("canvas");
      c2.width = img.w;
      c2.height = img.h;
      c2.style.cssText = "position:fixed;left:0;top:0;width:100vw;height:100vh";
      const id = new ImageData(img.w, img.h);
      for (let i = 0; i < img.w * img.h; i++) {
        id.data.set(img.rgb.subarray(i * 3, i * 3 + 3), i * 4);
        id.data[i * 4 + 3] = 255;
      }
      c2.getContext("2d").putImageData(id, 0, 0);
      document.body.appendChild(c2);
      return { w: img.w, h: img.h };
    },
    /** render a test camera at frame t into the float target and return 8-bit RGB (base64). */
    async renderTest(idx, t) {
      await allLoaded();
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const c = scene.testcams[idx];
      t = (t % total + total) % total;
      const si = scene.sections.findIndex((s) => t >= s.g0 && t < s.g0 + s.nframes);
      let seq = Math.floor(curSeq / nsec) * nsec + si;
      if (seq < curSeq) seq += nsec;
      const tt = t;
      await whenDecoded(seq);
      const g = ensureGpu(seq);
      if (!g) throw new Error(`section seq ${seq} not available (cur ${curSeq}, maxDecoded ${maxDecoded}, decoded [${[...decoded.keys()]}], gpu [${[...gpu.keys()]}])`);
      renderer.frame(g, tt, camFromRt(c.R, c.t, c.fx, c.fy, c.cx, c.cy, c.w, c.h), null);
      const img = await renderer.readTarget();
      const ids = Array.from({ length: Math.min(256, g.ps.n) }, (_, k) => Math.floor(k * g.ps.n / 256));
      const got = await renderer.readWorld(ids);
      let maxErr = 0, checked = 0;
      ids.forEach((i, k) => {
        const ref = evalSplat(g.ps, i, tt);
        if (!ref.alive) return;
        checked++;
        const flat = [...ref.vals[0], ...ref.vals[1], ...ref.vals[2], ...ref.vals[3], ...ref.vals[4]];
        const gpuv = [got[k][0], got[k][1], got[k][2], got[k][4], got[k][5], got[k][6], got[k][7], got[k][8], got[k][9], got[k][10], got[k][11], got[k][12], got[k][13], got[k][14]];
        flat.forEach((v, j) => {
          maxErr = Math.max(maxErr, Math.abs(v - gpuv[j]) / Math.max(1, Math.abs(v)));
        });
      });
      rafOn = true;
      lastNow = 0;
      requestAnimationFrame(tick);
      let bin = "";
      for (let i = 0; i < img.rgb.length; i += 32768) bin += String.fromCharCode(...img.rgb.subarray(i, i + 32768));
      return { w: img.w, h: img.h, rgb_b64: btoa(bin), eval_check: { splats: checked, max_rel_err: maxErr }, n: g.ps.n, seq, t: tt };
    },
    /** P40 (WB-1341, owner: black spikes on a red background): render the orbit view (az/elev deg, zoom) or test camera
     *  `testcam` at frame t (fractional allowed) and return the PREMULTIPLIED RGBA target as 8-bit (base64), so the
     *  caller can composite any background; world=true also returns the evaluated world buffer (16 floats per splat). */
    async renderRGBA(o) {
      await allLoaded();
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const t = (o.t % total + total) % total;
      const si = scene.sections.findIndex((s) => t >= s.g0 && t < s.g0 + s.nframes);
      let seq = Math.floor(curSeq / nsec) * nsec + si;
      if (seq < curSeq) seq += nsec;
      await whenDecoded(seq);
      const g = ensureGpu(seq);
      if (!g) throw new Error(`section seq ${seq} not available`);
      let cam;
      if (o.testcam !== void 0) {
        const c = scene.testcams[o.testcam];
        cam = camFromRt(c.R, c.t, c.fx, c.fy, c.cx, c.cy, c.w, c.h);
      } else {
        cam = orbitCam(
          scene,
          (o.az ?? 0) * Math.PI / 180,
          (o.elev ?? scene.orbit.elev_deg) * Math.PI / 180,
          o.zoom ?? 1,
          canvas.width,
          canvas.height
        );
      }
      renderer.frame(g, t, cam, null);
      const px = await renderer.readTargetRGBA(1);
      const u8 = new Uint8Array(px.length);
      for (let i = 0; i < px.length; i++) u8[i] = Math.round(Math.min(1, Math.max(0, px[i])) * 255);
      const b64 = (a) => {
        let s = "";
        for (let i = 0; i < a.length; i += 32768) s += String.fromCharCode(...a.subarray(i, i + 32768));
        return btoa(s);
      };
      let world = null;
      if (o.world) world = b64(new Uint8Array(await renderer.readWorldAll()));
      rafOn = true;
      lastNow = 0;
      requestAnimationFrame(tick);
      return { w: cam.width, h: cam.height, rgba_b64: b64(u8), n: g.ps.n, seq, t, world_b64: world };
    }
  };
  window.__p5 = p5;
}
function summarize(log) {
  if (!log.length) return null;
  const out = {};
  for (const k of Object.keys(log[0])) {
    const v = log.map((r) => r[k]).filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
    if (!v.length) continue;
    out[k] = { median: v[Math.floor(v.length / 2)], p95: v[Math.floor(v.length * 0.95)], mean: v.reduce((a, b) => a + b, 0) / v.length };
  }
  return { samples: log.length, ...out };
}
main().catch((e) => fail(String(e?.stack ?? e)));
