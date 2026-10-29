/* H-space explainer: tiny drawing + simulation helpers. No dependencies.
   Everything draws on a <canvas> in plain black, blue, red, green and gray.

   Quick reference
   ---------------
   var c = S.canvas(hostEl, {aspect: 0.6});      // canvas sized to host width; c.ctx, c.W, c.H (CSS px)
   c.draw = function () { ... };                   // called on resize and by c.redraw()
   var v = S.view(c, {xmin, xmax, ymin, ymax, pad}); // math coords -> pixels
   v.X(x), v.Y(y), v.invX(px), v.invY(py)
   v.axes({xlabel, ylabel, grid: true, xticks: 1, yticks: 1})
   v.fn(f, {color, width, dash, from, to})        // plot y = f(x)
   v.line(x1,y1,x2,y2,opts) v.arrow(x1,y1,x2,y2,opts) v.dot(x,y,opts)
   v.text(str,x,y,opts) v.poly([[x,y],...],opts) v.rect(x,y,w,h,opts)
   v.heat(f, {min, max}) v.contour(f, levels, opts)
   S.slider(host, {label, min, max, step, value, unit, digits, oninput}) -> {get(), set(v), el}
   S.button(host, label, onclick)   S.readout(host) -> element (set .textContent)
   S.drag(c, view, points, onmove)  // points: [{x,y,r?,fixedX?,fixedY?}]
   S.rng(seed) -> {u(), n()}  uniform and standard normal
   S.linalg: det2, inv2, mul2, eigSym2, eigSym (Jacobi, any n), matVec, dot, norm
   S.fmt(x, digits)
   S.v3(c, {azimuth, elevation, scale, zscale}) -> 3-D wireframe helper: p(x,y,z) -> [px,py], surface(f, opts)
*/
(function () {
  "use strict";
  var S = {};
  S.col = { ink: "#000", blue: "#0000cc", red: "#cc0000", green: "#007700", gray: "#888", light: "#ddd", faint: "#f0f0f0", orange: "#cc6600", purple: "#660099" };
  S.font = '15px "Times New Roman", Times, serif';
  S.fontSmall = '13px "Times New Roman", Times, serif';

  S.fmt = function (x, d) {
    if (d == null) d = 2;
    if (!isFinite(x)) return String(x);
    var s = x.toFixed(d);
    if (/^-0\.?0*$/.test(s)) s = s.replace("-", "");
    return s.replace("-", "−");
  };

  /* ---------- canvas ---------- */
  S.canvas = function (host, opts) {
    opts = opts || {};
    var cv = document.createElement("canvas");
    host.appendChild(cv);
    var o = { cv: cv, ctx: cv.getContext("2d"), W: 0, H: 0, aspect: opts.aspect || 0.6, maxH: opts.maxH || 520, draw: null };
    function size() {
      var w = Math.max(240, Math.floor(host.clientWidth || 600));
      var h = Math.min(o.maxH, Math.floor(w * o.aspect));
      var dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.height = h + "px";
      o.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      o.W = w; o.H = h;
    }
    o.redraw = function () {
      o.ctx.clearRect(0, 0, o.W, o.H);
      o.ctx.fillStyle = "#fff"; o.ctx.fillRect(0, 0, o.W, o.H);
      if (o.draw) o.draw();
    };
    size();
    var t = null;
    window.addEventListener("resize", function () {
      clearTimeout(t);
      t = setTimeout(function () { size(); o.redraw(); }, 60);
    });
    return o;
  };

  /* ---------- 2-D view ---------- */
  S.view = function (c, r) {
    var v = { c: c, ctx: c.ctx };
    v.set = function (nr) { for (var k in nr) r[k] = nr[k]; };
    function pad() { return r.pad != null ? r.pad : 36; }
    function box() {
      var p = pad(), pl = r.padLeft != null ? r.padLeft : p, pb = r.padBottom != null ? r.padBottom : p;
      var x0 = r.left != null ? r.left : 0, w = r.width != null ? r.width : c.W;
      return { l: x0 + pl, r: x0 + w - (r.padRight != null ? r.padRight : 12), t: (r.padTop != null ? r.padTop : 12), b: c.H - pb };
    }
    v.box = box;
    v.X = function (x) { var b = box(); return b.l + (x - r.xmin) / (r.xmax - r.xmin) * (b.r - b.l); };
    v.Y = function (y) { var b = box(); return b.b - (y - r.ymin) / (r.ymax - r.ymin) * (b.b - b.t); };
    v.invX = function (px) { var b = box(); return r.xmin + (px - b.l) / (b.r - b.l) * (r.xmax - r.xmin); };
    v.invY = function (py) { var b = box(); return r.ymin + (b.b - py) / (b.b - b.t) * (r.ymax - r.ymin); };
    v.r = r;
    function style(o, defColor) {
      var ctx = v.ctx; o = o || {};
      ctx.strokeStyle = o.color || defColor || S.col.ink;
      ctx.fillStyle = o.fill || o.color || defColor || S.col.ink;
      ctx.lineWidth = o.width || 1.5;
      ctx.setLineDash(o.dash || []);
      ctx.globalAlpha = o.alpha != null ? o.alpha : 1;
      return o;
    }
    function done() { v.ctx.setLineDash([]); v.ctx.globalAlpha = 1; }
    function niceStep(span, target) {
      var raw = span / (target || 6), p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
    }
    v.axes = function (o) {
      o = o || {};
      var ctx = v.ctx, b = box();
      var xs = o.xticks || niceStep(r.xmax - r.xmin), ys = o.yticks || niceStep(r.ymax - r.ymin);
      ctx.font = S.fontSmall; ctx.textBaseline = "top"; ctx.textAlign = "center";
      var x, y;
      if (o.grid !== false) {
        ctx.strokeStyle = S.col.faint; ctx.lineWidth = 1; ctx.beginPath();
        for (x = Math.ceil(r.xmin / xs) * xs; x <= r.xmax + 1e-9; x += xs) { ctx.moveTo(v.X(x), b.t); ctx.lineTo(v.X(x), b.b); }
        for (y = Math.ceil(r.ymin / ys) * ys; y <= r.ymax + 1e-9; y += ys) { ctx.moveTo(b.l, v.Y(y)); ctx.lineTo(b.r, v.Y(y)); }
        ctx.stroke();
      }
      var ax = Math.min(Math.max(0, r.ymin), r.ymax), ay = Math.min(Math.max(0, r.xmin), r.xmax);
      if (o.frame) { ax = r.ymin; ay = r.xmin; }
      ctx.strokeStyle = S.col.ink; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(b.l, v.Y(ax)); ctx.lineTo(b.r, v.Y(ax));
      ctx.moveTo(v.X(ay), b.t); ctx.lineTo(v.X(ay), b.b); ctx.stroke();
      ctx.fillStyle = S.col.ink;
      var dx = o.xdigits != null ? o.xdigits : (xs < 1 ? Math.min(3, Math.ceil(-Math.log10(xs))) : 0);
      var dy = o.ydigits != null ? o.ydigits : (ys < 1 ? Math.min(3, Math.ceil(-Math.log10(ys))) : 0);
      if (o.xnums !== false) for (x = Math.ceil(r.xmin / xs) * xs; x <= r.xmax + 1e-9; x += xs) {
        if (Math.abs(x - ay) < 1e-9 && !o.frame) continue;
        ctx.fillText(S.fmt(x, dx), v.X(x), v.Y(ax) + 3);
      }
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      if (o.ynums !== false) for (y = Math.ceil(r.ymin / ys) * ys; y <= r.ymax + 1e-9; y += ys) {
        if (Math.abs(y - ax) < 1e-9 && !o.frame) continue;
        ctx.fillText(S.fmt(y, dy), v.X(ay) - 4, v.Y(y));
      }
      ctx.font = S.font;
      if (o.xlabel) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(o.xlabel, b.r, v.Y(ax) - 3); }
      if (o.ylabel) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(o.ylabel, v.X(ay) + 5, b.t); }
    };
    v.fn = function (f, o) {
      o = style(o, S.col.blue);
      var ctx = v.ctx, b = box(), from = o.from != null ? o.from : r.xmin, to = o.to != null ? o.to : r.xmax;
      var n = o.n || Math.max(200, Math.round(b.r - b.l)), started = false, i, x, y;
      ctx.save(); ctx.beginPath(); ctx.rect(b.l - 1, b.t - 1, b.r - b.l + 2, b.b - b.t + 2); ctx.clip();
      ctx.beginPath();
      for (i = 0; i <= n; i++) {
        x = from + (to - from) * i / n; y = f(x);
        if (!isFinite(y) || y > r.ymax + (r.ymax - r.ymin) * 4 || y < r.ymin - (r.ymax - r.ymin) * 4) { started = false; continue; }
        if (!started) { ctx.moveTo(v.X(x), v.Y(y)); started = true; } else ctx.lineTo(v.X(x), v.Y(y));
      }
      ctx.stroke(); ctx.restore(); done();
    };
    v.line = function (x1, y1, x2, y2, o) {
      style(o); var ctx = v.ctx; ctx.beginPath(); ctx.moveTo(v.X(x1), v.Y(y1)); ctx.lineTo(v.X(x2), v.Y(y2)); ctx.stroke(); done();
    };
    v.arrow = function (x1, y1, x2, y2, o) {
      o = style(o); var ctx = v.ctx;
      var a = v.X(x1), b = v.Y(y1), c2 = v.X(x2), d = v.Y(y2), ang = Math.atan2(d - b, c2 - a), hl = o.head || 9;
      ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c2, d); ctx.stroke();
      if (Math.hypot(c2 - a, d - b) > 2) {
        ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(c2, d);
        ctx.lineTo(c2 - hl * Math.cos(ang - 0.4), d - hl * Math.sin(ang - 0.4));
        ctx.lineTo(c2 - hl * Math.cos(ang + 0.4), d - hl * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fill();
      }
      done();
    };
    v.dot = function (x, y, o) {
      o = style(o); var ctx = v.ctx; ctx.beginPath(); ctx.arc(v.X(x), v.Y(y), o.r || 4.5, 0, 2 * Math.PI);
      if (o.hollow) { ctx.fillStyle = "#fff"; ctx.fill(); ctx.stroke(); } else ctx.fill();
      done();
    };
    v.text = function (s, x, y, o) {
      o = o || {}; var ctx = v.ctx;
      ctx.font = o.font || S.font; ctx.fillStyle = o.color || S.col.ink;
      ctx.textAlign = o.align || "left"; ctx.textBaseline = o.baseline || "alphabetic";
      ctx.fillText(s, v.X(x) + (o.dx || 0), v.Y(y) + (o.dy || 0));
    };
    v.poly = function (pts, o) {
      o = style(o); var ctx = v.ctx; ctx.beginPath();
      pts.forEach(function (p, i) { if (i) ctx.lineTo(v.X(p[0]), v.Y(p[1])); else ctx.moveTo(v.X(p[0]), v.Y(p[1])); });
      if (o.close !== false) ctx.closePath();
      if (o.fill) { ctx.globalAlpha = o.fillAlpha != null ? o.fillAlpha : 0.15; ctx.fillStyle = o.fill; ctx.fill(); ctx.globalAlpha = 1; }
      if (o.stroke !== false) ctx.stroke();
      done();
    };
    v.rect = function (x, y, w, h, o) { v.poly([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], o); };
    v.heat = function (f, o) {
      o = o || {}; var ctx = v.ctx, b = box(), step = o.step || 4;
      var lo = o.min, hi = o.max;
      for (var px = b.l; px < b.r; px += step) for (var py = b.t; py < b.b; py += step) {
        var z = f(v.invX(px + step / 2), v.invY(py + step / 2));
        ctx.fillStyle = S.diverge(z, lo, hi);
        ctx.fillRect(px, py, step + 0.5, step + 0.5);
      }
    };
    v.contour = function (f, levels, o) {
      o = style(o, S.col.gray); var ctx = v.ctx, b = box(), n = o.res || 60;
      var dx = (r.xmax - r.xmin) / n, dy = (r.ymax - r.ymin) / n;
      ctx.save(); ctx.beginPath(); ctx.rect(b.l, b.t, b.r - b.l, b.b - b.t); ctx.clip();
      levels.forEach(function (lv) {
        ctx.beginPath();
        for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
          var x = r.xmin + i * dx, y = r.ymin + j * dy;
          var c = [f(x, y) - lv, f(x + dx, y) - lv, f(x + dx, y + dy) - lv, f(x, y + dy) - lv];
          var P = [[x, y], [x + dx, y], [x + dx, y + dy], [x, y + dy]], cross = [];
          for (var k = 0; k < 4; k++) {
            var a = c[k], bb = c[(k + 1) % 4];
            if ((a < 0) !== (bb < 0)) {
              var t = a / (a - bb), p = P[k], q = P[(k + 1) % 4];
              cross.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
          }
          for (k = 0; k + 1 < cross.length; k += 2) {
            ctx.moveTo(v.X(cross[k][0]), v.Y(cross[k][1])); ctx.lineTo(v.X(cross[k + 1][0]), v.Y(cross[k + 1][1]));
          }
        }
        ctx.stroke();
      });
      ctx.restore(); done();
    };
    return v;
  };

  /* blue (negative) -> white (zero) -> red (positive) */
  S.diverge = function (z, lo, hi) {
    if (lo == null) lo = -1; if (hi == null) hi = 1;
    var t;
    if (z >= 0) { t = Math.min(1, z / (hi || 1)); return "rgb(255," + Math.round(255 - 175 * t) + "," + Math.round(255 - 175 * t) + ")"; }
    t = Math.min(1, z / (lo || -1)); return "rgb(" + Math.round(255 - 175 * t) + "," + Math.round(255 - 175 * t) + ",255)";
  };
  S.gray = function (z, lo, hi) {
    var t = Math.max(0, Math.min(1, (z - lo) / (hi - lo))), g = Math.round(255 - 200 * t);
    return "rgb(" + g + "," + g + "," + g + ")";
  };

  /* ---------- controls ---------- */
  function controlsOf(host) {
    var c = host.querySelector(":scope > .controls");
    if (!c) { c = document.createElement("div"); c.className = "controls"; host.appendChild(c); }
    return c;
  }
  S.slider = function (host, o) {
    var wrap = document.createElement("div"); wrap.className = "ctl nogl";
    var id = "s" + Math.random().toString(36).slice(2, 8);
    wrap.innerHTML = '<label for="' + id + '">' + o.label + '</label><input type="range" id="' + id + '" min="' + o.min + '" max="' + o.max + '" step="' + (o.step || (o.max - o.min) / 100) + '" value="' + o.value + '"><span class="val"></span>';
    controlsOf(host).appendChild(wrap);
    var inp = wrap.querySelector("input"), val = wrap.querySelector(".val");
    var api = {
      el: inp,
      get: function () { return parseFloat(inp.value); },
      set: function (x) { inp.value = x; show(); },
      label: function (s) { wrap.querySelector("label").innerHTML = s; }
    };
    function show() { val.textContent = "= " + S.fmt(api.get(), o.digits != null ? o.digits : 2) + (o.unit ? " " + o.unit : ""); }
    inp.addEventListener("input", function () { show(); if (o.oninput) o.oninput(api.get()); });
    show();
    return api;
  };
  S.button = function (host, label, onclick) {
    var b = document.createElement("button"); b.type = "button"; b.textContent = label;
    b.addEventListener("click", onclick); controlsOf(host).appendChild(b); return b;
  };
  S.checkbox = function (host, label, checked, onchange) {
    var w = document.createElement("label"); w.className = "nogl";
    w.innerHTML = '<input type="checkbox"' + (checked ? " checked" : "") + "> " + label;
    var i = w.querySelector("input"); i.addEventListener("change", function () { onchange(i.checked); });
    controlsOf(host).appendChild(w); return { get: function () { return i.checked; }, set: function (v) { i.checked = v; } };
  };
  S.readout = function (host) {
    var d = document.createElement("div"); d.className = "readout"; host.appendChild(d); return d;
  };

  /* ---------- dragging ---------- */
  S.drag = function (c, v, points, onmove) {
    var active = null;
    function pos(ev) {
      var r = c.cv.getBoundingClientRect(), t = ev.touches ? ev.touches[0] : ev;
      return [t.clientX - r.left, t.clientY - r.top];
    }
    function down(ev) {
      var p = pos(ev), best = null, bd = 1e9;
      points.forEach(function (pt) {
        var d = Math.hypot(v.X(pt.x) - p[0], v.Y(pt.y) - p[1]);
        if (d < (pt.r || 16) && d < bd) { bd = d; best = pt; }
      });
      if (best) { active = best; ev.preventDefault(); }
    }
    function move(ev) {
      var p = pos(ev);
      if (!active) {
        var hit = points.some(function (pt) { return Math.hypot(v.X(pt.x) - p[0], v.Y(pt.y) - p[1]) < (pt.r || 16); });
        c.cv.style.cursor = hit ? "grab" : "default";
        return;
      }
      ev.preventDefault();
      var x = v.invX(p[0]), y = v.invY(p[1]), R = v.r;
      if (!active.fixedX) active.x = Math.max(R.xmin, Math.min(R.xmax, x));
      if (!active.fixedY) active.y = Math.max(R.ymin, Math.min(R.ymax, y));
      if (onmove) onmove(active);
    }
    function up() { active = null; }
    c.cv.addEventListener("mousedown", down); window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
    c.cv.addEventListener("touchstart", down, { passive: false }); c.cv.addEventListener("touchmove", move, { passive: false }); window.addEventListener("touchend", up);
  };

  /* ---------- random numbers (seeded, so pictures are reproducible) ---------- */
  S.rng = function (seed) {
    /* scramble the seed (splitmix32-style) so small seeds like 1, 2, 3 still give well-mixed streams */
    var s = (seed >>> 0) + 0x9e3779b9;
    s = Math.imul(s ^ (s >>> 16), 0x85ebca6b); s = Math.imul(s ^ (s >>> 13), 0xc2b2ae35); s = (s ^ (s >>> 16)) >>> 0;
    if (!s) s = 1;
    function u() { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; }
    var spare = null;
    function n() {
      if (spare != null) { var t = spare; spare = null; return t; }
      var a = u(), b = u(), r = Math.sqrt(-2 * Math.log(a));
      spare = r * Math.sin(2 * Math.PI * b); return r * Math.cos(2 * Math.PI * b);
    }
    for (var w = 0; w < 20; w++) u();   /* warm-up: discard the first draws */
    return { u: u, n: n };
  };

  /* ---------- small linear algebra ---------- */
  var L = {};
  L.det2 = function (m) { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; };
  L.inv2 = function (m) { var d = L.det2(m); return [[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]]; };
  L.mul2 = function (a, b) { return [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]], [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]; };
  L.matVec = function (m, x) { return m.map(function (row) { return row.reduce(function (s, a, j) { return s + a * x[j]; }, 0); }); };
  L.dot = function (a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  L.norm = function (a) { return Math.sqrt(L.dot(a, a)); };
  /* symmetric 2x2: returns {vals:[big, small], vecs:[[x,y],[x,y]]} */
  L.eigSym2 = function (m) {
    var a = m[0][0], b = m[0][1], d = m[1][1], tr = a + d, disc = Math.sqrt((a - d) * (a - d) / 4 + b * b);
    var l1 = tr / 2 + disc, l2 = tr / 2 - disc, v1;
    if (Math.abs(b) > 1e-12) v1 = [l1 - d, b]; else v1 = a >= d ? [1, 0] : [0, 1];
    var n = Math.hypot(v1[0], v1[1]); v1 = [v1[0] / n, v1[1] / n];
    return { vals: [l1, l2], vecs: [v1, [-v1[1], v1[0]]] };
  };
  /* Jacobi eigen-decomposition of a symmetric n x n matrix. Returns values sorted
     high to low and matching unit eigenvectors (vecs[i] is the i-th eigenvector). */
  L.eigSym = function (A) {
    var n = A.length, a = A.map(function (r) { return r.slice(); }), V = [], i, j, k;
    for (i = 0; i < n; i++) { V.push([]); for (j = 0; j < n; j++) V[i].push(i === j ? 1 : 0); }
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0;
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
      if (off < 1e-20) break;
      for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        var th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        var t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        var c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (k = 0; k < n; k++) {
          var akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (k = 0; k < n; k++) {
          var apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (k = 0; k < n; k++) {
          var vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    var idx = []; for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (x, y) { return a[y][y] - a[x][x]; });
    return {
      vals: idx.map(function (i2) { return a[i2][i2]; }),
      vecs: idx.map(function (i2) { return V.map(function (row) { return row[i2]; }); })
    };
  };
  S.linalg = L;

  /* ---------- simple 3-D wireframe ---------- */
  S.v3 = function (c, o) {
    o = o || {};
    var g = { azimuth: o.azimuth != null ? o.azimuth : -0.6, elevation: o.elevation != null ? o.elevation : 0.55, scale: o.scale || 1, zscale: o.zscale || 1, cx: o.cx, cy: o.cy };
    g.p = function (x, y, z) {
      var ca = Math.cos(g.azimuth), sa = Math.sin(g.azimuth), ce = Math.cos(g.elevation), se = Math.sin(g.elevation);
      var X = x * ca - y * sa, Y = x * sa + y * ca, Z = z * g.zscale;
      var sx = X, sy = Y * se - Z * ce;
      var k = Math.min(c.W, c.H) * 0.22 * g.scale;
      return [(g.cx != null ? g.cx : c.W / 2) + sx * k, (g.cy != null ? g.cy : c.H * 0.55) + sy * k];
    };
    g.surface = function (f, s) {
      s = s || {}; var ctx = c.ctx, n = s.n || 24, r = s.range || 2;
      ctx.lineWidth = s.width || 1; ctx.strokeStyle = s.color || S.col.gray;
      for (var i = 0; i <= n; i++) {
        var a = -r + 2 * r * i / n;
        ctx.beginPath();
        for (var j = 0; j <= n; j++) { var b = -r + 2 * r * j / n, P = g.p(a, b, f(a, b)); if (j) ctx.lineTo(P[0], P[1]); else ctx.moveTo(P[0], P[1]); }
        ctx.stroke(); ctx.beginPath();
        for (j = 0; j <= n; j++) { b = -r + 2 * r * j / n; P = g.p(b, a, f(b, a)); if (j) ctx.lineTo(P[0], P[1]); else ctx.moveTo(P[0], P[1]); }
        ctx.stroke();
      }
    };
    g.line = function (a, b, s) {
      s = s || {}; var ctx = c.ctx, P = g.p(a[0], a[1], a[2]), Q = g.p(b[0], b[1], b[2]);
      ctx.strokeStyle = s.color || S.col.ink; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []);
      ctx.beginPath(); ctx.moveTo(P[0], P[1]); ctx.lineTo(Q[0], Q[1]); ctx.stroke(); ctx.setLineDash([]);
    };
    g.dot = function (a, s) {
      s = s || {}; var ctx = c.ctx, P = g.p(a[0], a[1], a[2]);
      ctx.fillStyle = s.color || S.col.red; ctx.beginPath(); ctx.arc(P[0], P[1], s.r || 4.5, 0, 2 * Math.PI); ctx.fill();
    };
    g.text = function (str, a, s) {
      s = s || {}; var ctx = c.ctx, P = g.p(a[0], a[1], a[2]);
      ctx.font = S.font; ctx.fillStyle = s.color || S.col.ink; ctx.textAlign = s.align || "left"; ctx.fillText(str, P[0] + (s.dx || 4), P[1] + (s.dy || -4));
    };
    return g;
  };

  /* run page sims once the shared machinery (KaTeX, glossary) has finished */
  S.ready = function (fn) {
    if (window.__explainerReady) fn();
    else document.addEventListener("explainer:ready", fn);
  };
  document.addEventListener("explainer:ready", function () { window.__explainerReady = true; });

  window.S = S;
})();
