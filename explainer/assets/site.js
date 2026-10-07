/* H-space explainer: shared page machinery.
   1. top bar + prev/next navigation
   2. KaTeX rendering
   3. glossary auto-linking and hover pop-ups (switchable)
   4. exercises (number check + reveal)
   Page-specific simulations run after this, on the "explainer:ready" event. */

(function () {
  "use strict";

  var LEVELS = [
    { n: 0, file: "00.html", title: "Numbers, units and rates" },
    { n: 1, file: "01.html", title: "Variables, functions and graphs" },
    { n: 2, file: "02.html", title: "Slope and the derivative" },
    { n: 3, file: "03.html", title: "The second derivative: curvature" },
    { n: 4, file: "04.html", title: "Taylor approximation" },
    { n: 5, file: "05.html", title: "Two inputs: partial derivatives and the gradient" },
    { n: 6, file: "06.html", title: "Interaction: the mixed partial derivative" },
    { n: 7, file: "07.html", title: "Vectors, matrices and the determinant" },
    { n: 8, file: "08.html", title: "The Jacobian" },
    { n: 9, file: "09.html", title: "The Hessian" },
    { n: 10, file: "10.html", title: "The chain rule and backpropagation" },
    { n: 11, file: "11.html", title: "Hessian-vector products" },
    { n: 12, file: "12.html", title: "The second-order chain rule and the rectangle identity" },
    { n: 13, file: "13.html", title: "Statistics of many directions" },
    { n: 14, file: "14.html", title: "Transformers, the residual stream and the J-lens" },
    { n: 15, file: "15.html", title: "The H-lens and what the runs found" }
  ];
  window.EXPLAINER_LEVELS = LEVELS;

  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* ignore */ } }
  };

  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }

  /* ---------- 1. top bar and navigation ---------- */
  function buildChrome() {
    var body = document.body;
    var lvl = body.getAttribute("data-level");
    var bar = el("div", { "class": "topbar" });
    bar.innerHTML =
      '<a href="index.html">H-space explainer</a>' +
      '<a href="index.html#contents">Contents</a>' +
      '<a href="glossary.html">Glossary</a>' +
      '<span class="spacer"></span>' +
      '<span id="offline" class="small" title="Once every file is saved, the whole site works with no internet."></span>' +
      '<label><input type="checkbox" id="glosstoggle"> Definition pop-ups</label>';
    body.insertBefore(bar, body.firstChild);

    var box = document.getElementById("glosstoggle");
    var on = store.get("explainer-gloss") !== "off";
    box.checked = on;
    body.classList.toggle("nogloss", !on);
    box.addEventListener("change", function () {
      body.classList.toggle("nogloss", !box.checked);
      store.set("explainer-gloss", box.checked ? "on" : "off");
      hidePop();
    });

    if (lvl == null) return;
    var n = parseInt(lvl, 10);
    var prev = LEVELS[n - 1], next = LEVELS[n + 1];
    function navHtml() {
      var parts = [];
      parts.push(prev ? '&lt;&lt; <a href="' + prev.file + '">Level ' + prev.n + ": " + prev.title + "</a>" : "&lt;&lt; (start)");
      parts.push('<a href="index.html#contents">Contents</a>');
      parts.push(next ? '<a href="' + next.file + '">Level ' + next.n + ": " + next.title + "</a> &gt;&gt;" : "(end) &gt;&gt;");
      return parts.join(" &nbsp;|&nbsp; ");
    }
    var page = document.querySelector(".page");
    var top = el("div", { "class": "nav top" }, navHtml());
    page.insertBefore(top, page.firstChild);
    var bot = el("div", { "class": "nav bottom" }, navHtml());
    page.appendChild(bot);
    var ft = el("div", { "class": "footer" },
      'H-space explainer, part of <a href="https://github.com/RaphaelKhalid/hspace">github.com/RaphaelKhalid/hspace</a>. ' +
      "Plain HTML. Best viewed in any browser.");
    page.appendChild(ft);
  }

  /* ---------- 2. KaTeX ---------- */
  function renderMath() {
    if (typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      ignoredClasses: ["nomath"],
      throwOnError: false,
      strict: false
    });
  }

  /* ---------- 3. glossary ---------- */
  var GL = window.GLOSSARY || {};
  var matcher = null, lookup = {};

  function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  /* Acronyms and mixed-case names (RoPE, KL, SiLU, PCA) must match exactly,
     so ordinary words like "rope" are not linked. Other terms ignore case. */
  function caseSensitive(f) { return /[A-Z]/.test(f.slice(1)); }
  function formPattern(f) {
    if (caseSensitive(f)) return escapeRe(f);
    return f.split("").map(function (ch) {
      var lo = ch.toLowerCase(), up = ch.toUpperCase();
      return lo !== up ? "[" + escapeRe(lo) + escapeRe(up) + "]" : escapeRe(ch);
    }).join("");
  }

  function buildMatcher() {
    var forms = [];
    Object.keys(GL).forEach(function (key) {
      var entry = GL[key];
      var all = [entry.t || key].concat(entry.a || []);
      all.forEach(function (f) {
        var id = caseSensitive(f) ? f : f.toLowerCase();
        if (!lookup[id]) { lookup[id] = key; forms.push(f); }
      });
    });
    if (!forms.length) return;
    forms.sort(function (a, b) { return b.length - a.length; });
    matcher = new RegExp("(?<![A-Za-z0-9\\u0370-\\u03FF_-])(" + forms.map(formPattern).join("|") + ")(?![A-Za-z0-9\\u0370-\\u03FF_])", "g");
  }
  function keyFor(text) { return lookup[text] || lookup[text.toLowerCase()]; }

  var SKIP_TAGS = { H1: 1, H2: 1, H3: 1, H4: 1, A: 1, CODE: 1, PRE: 1, SCRIPT: 1, STYLE: 1, BUTTON: 1, CANVAS: 1, INPUT: 1, SELECT: 1, TEXTAREA: 1, SUMMARY: 0, LABEL: 1, TITLE: 1 };

  function skippable(node) {
    for (var p = node.parentNode; p && p !== document.body; p = p.parentNode) {
      if (p.nodeType !== 1) continue;
      if (SKIP_TAGS[p.tagName]) return true;
      var c = p.classList;
      if (c && (c.contains("katex") || c.contains("g") || c.contains("nogl") || c.contains("topbar") || c.contains("nav") || c.contains("readout") || c.contains("footer"))) return true;
    }
    return false;
  }

  /* Each term is linked once per block (paragraph, list item, table cell, caption),
     so every paragraph that uses a word marks it, without underlining every repeat. */
  var BLOCKS = { P: 1, LI: 1, TD: 1, TH: 1, DD: 1, DT: 1, FIGCAPTION: 1, BLOCKQUOTE: 1, SUMMARY: 1, DIV: 1 };
  function blockOf(node) {
    for (var p = node.parentNode; p && p !== document.body; p = p.parentNode) {
      if (p.nodeType === 1 && BLOCKS[p.tagName]) return p;
    }
    return document.body;
  }

  function linkTerms(root, selfKey) {
    if (!matcher) return;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    var seen = new WeakMap();
    nodes.forEach(function (tn) {
      if (!tn.nodeValue || !/[A-Za-z]/.test(tn.nodeValue)) return;
      if (skippable(tn)) return;
      var blk = blockOf(tn);
      var used = seen.get(blk);
      if (!used) { used = {}; seen.set(blk, used); }
      var text = tn.nodeValue;
      matcher.lastIndex = 0;
      var m, last = 0, frag = null;
      while ((m = matcher.exec(text)) !== null) {
        var key = keyFor(m[1]);
        if (!key || key === selfKey || used[key]) continue;
        /* "mean" as a verb ("does mean", "to mean", "they mean") is not the statistic */
        if (key === "mean" && /\b(does|do|did|to|will|would|can|could|might|may|must|they|we|you|I|these|those|which|that|this|it|words?|not|n't)\s+$/i.test(text.slice(Math.max(0, m.index - 12), m.index))) continue;
        used[key] = 1;
        if (!frag) frag = document.createDocumentFragment();
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        var s = document.createElement("span");
        s.className = "g";
        s.setAttribute("data-g", key);
        s.setAttribute("tabindex", "0");
        s.textContent = m[1];
        frag.appendChild(s);
        last = m.index + m[1].length;
      }
      if (frag) {
        frag.appendChild(document.createTextNode(text.slice(last)));
        tn.parentNode.replaceChild(frag, tn);
      }
    });
  }

  var pop = null, hideTimer = null, current = null;

  function levelLink(l) {
    var L = LEVELS[l];
    return L ? '<a href="' + L.file + '">Level ' + L.n + ": " + L.title + "</a>" : "";
  }

  function showPop(target) {
    if (document.body.classList.contains("nogloss")) return;
    var key = target.getAttribute("data-g");
    var e = GL[key];
    if (!e) return;
    clearTimeout(hideTimer);
    current = target;
    pop.innerHTML = '<div class="gt">' + (e.t || key) + "</div><div>" + e.d + "</div>" +
      (e.l != null ? '<div class="gl">Introduced in ' + levelLink(e.l) + ' &middot; <a href="glossary.html#' + encodeURIComponent(key) + '">glossary</a></div>' : "");
    pop.style.display = "block";
    var r = target.getBoundingClientRect();
    var pw = pop.offsetWidth, ph = pop.offsetHeight;
    var vw = document.documentElement.clientWidth;
    var left = r.left + window.scrollX;
    if (left + pw > window.scrollX + vw - 8) left = window.scrollX + vw - pw - 8;
    if (left < window.scrollX + 8) left = window.scrollX + 8;
    var top = r.bottom + window.scrollY + 6;
    if (r.bottom + ph + 12 > window.innerHeight && r.top - ph - 6 > 0) top = r.top + window.scrollY - ph - 6;
    pop.style.left = left + "px";
    pop.style.top = top + "px";
  }
  function hidePop() { if (pop) { pop.style.display = "none"; current = null; } }
  function hideSoon() { clearTimeout(hideTimer); hideTimer = setTimeout(hidePop, 220); }

  function wirePopups() {
    pop = el("div", { id: "gpop", role: "tooltip" });
    document.body.appendChild(pop);
    pop.addEventListener("mouseenter", function () { clearTimeout(hideTimer); });
    pop.addEventListener("mouseleave", hideSoon);
    document.addEventListener("mouseover", function (ev) {
      var t = ev.target.closest && ev.target.closest(".g");
      if (t) showPop(t);
    });
    document.addEventListener("mouseout", function (ev) {
      var t = ev.target.closest && ev.target.closest(".g");
      if (t) hideSoon();
    });
    document.addEventListener("focusin", function (ev) {
      var t = ev.target.closest && ev.target.closest(".g");
      if (t) showPop(t);
    });
    document.addEventListener("focusout", function (ev) {
      if (ev.target.closest && ev.target.closest(".g")) hideSoon();
    });
    document.addEventListener("click", function (ev) {
      var t = ev.target.closest && ev.target.closest(".g");
      if (t) {
        if (current === t && pop.style.display === "block") hidePop(); else showPop(t);
        return;
      }
      if (!ev.target.closest("#gpop")) hidePop();
    });
    document.addEventListener("keydown", function (ev) { if (ev.key === "Escape") hidePop(); });
  }

  function buildGlossaryPage() {
    var host = document.getElementById("glossary-list");
    if (!host) return;
    var keys = Object.keys(GL).sort(function (a, b) {
      return (GL[a].t || a).toLowerCase().localeCompare((GL[b].t || b).toLowerCase());
    });
    var html = '<p>' + keys.length + ' terms. Each links back to the level that introduces it.</p><dl>';
    keys.forEach(function (k) {
      var e = GL[k];
      html += '<dt id="' + k.replace(/"/g, "") + '"><b>' + (e.t || k) + "</b>" +
        (e.a && e.a.length ? ' <span class="small">(also: ' + e.a.join(", ") + ")</span>" : "") + "</dt>" +
        '<dd data-self="' + k + '">' + e.d + (e.l != null ? ' <span class="small">[' + levelLink(e.l) + "]</span>" : "") + "</dd>";
    });
    host.innerHTML = html + "</dl>";
    Array.prototype.forEach.call(host.querySelectorAll("dd"), function (dd) {
      linkTerms(dd, dd.getAttribute("data-self"));
    });
  }

  /* ---------- 4. exercises ---------- */
  function parseNum(s) {
    s = String(s).trim().replace(/\s+/g, "").replace(/−/g, "-");
    if (!s) return NaN;
    var frac = s.match(/^(-?[0-9.]+)\/(-?[0-9.]+)$/);
    if (frac) return parseFloat(frac[1]) / parseFloat(frac[2]);
    return parseFloat(s);
  }
  function wireExercises() {
    Array.prototype.forEach.call(document.querySelectorAll(".ex[data-answer]"), function (ex) {
      var ans = parseFloat(ex.getAttribute("data-answer"));
      var tol = parseFloat(ex.getAttribute("data-tol") || "0.01");
      var unit = ex.getAttribute("data-unit") || "";
      var row = el("div", { "class": "check" });
      row.innerHTML = '<label class="nogl">Your answer: <input type="text" inputmode="decimal" autocomplete="off"></label>' +
        (unit ? '<span class="small">' + unit + "</span>" : "") +
        '<button type="button">Check</button><span class="verdict" aria-live="polite"></span>';
      var det = ex.querySelector("details");
      ex.insertBefore(row, det || null);
      var inp = row.querySelector("input"), out = row.querySelector(".verdict");
      function check() {
        var v = parseNum(inp.value);
        if (isNaN(v)) { out.textContent = "Type a number (fractions like 3/4 are fine)."; return; }
        var ok = Math.abs(v - ans) <= tol * Math.max(1, Math.abs(ans));
        out.textContent = ok ? "Correct." : "Not quite. Try again, or open the answer below.";
      }
      row.querySelector("button").addEventListener("click", check);
      inp.addEventListener("keydown", function (e) { if (e.key === "Enter") check(); });
    });
  }

  /* ---------- 5. offline copy (service worker, see tools/build_offline.py) ---------- */
  function setupOffline() {
    var label = document.getElementById("offline");
    if (!("serviceWorker" in navigator) || location.protocol === "file:") {
      /* e.g. Chrome or Firefox on iPhone: their tabs cannot keep an offline copy, a Home Screen icon can */
      if (label) label.innerHTML = '<a href="index.html#offline">Offline: add to Home Screen</a>';
      return;
    }
    /* local development: no offline cache (it would serve stale pages while editing),
       unless the URL has ?sw to test it */
    if (/^(localhost|127\.0\.0\.1)$/.test(location.hostname) && !/[?&]sw\b/.test(location.search) && !store.get("explainer-sw-test")) {
      navigator.serviceWorker.getRegistrations().then(function (rs) { rs.forEach(function (r) { r.unregister(); }); });
      if (window.caches) caches.keys().then(function (ks) { ks.forEach(function (k) { if (k.indexOf("hspace-explainer-") === 0) caches.delete(k); }); });
      if (label) label.textContent = "";
      return;
    }
    if (/[?&]sw\b/.test(location.search)) store.set("explainer-sw-test", "1");
    function ask() {
      if (navigator.serviceWorker.controller) navigator.serviceWorker.controller.postMessage({ type: "status" });
    }
    navigator.serviceWorker.addEventListener("message", function (ev) {
      var d = ev.data || {};
      if (d.type !== "status" || !label) return;
      if (d.cached >= d.total) {
        label.textContent = "Saved for offline ✓";
      } else {
        label.textContent = "Saving for offline: " + d.cached + "/" + d.total;
        setTimeout(ask, 1500);
      }
    });
    navigator.serviceWorker.register("sw.js").then(function () {
      if (label) label.textContent = "Saving for offline…";
      navigator.serviceWorker.ready.then(function () { setTimeout(ask, 300); });
      navigator.serviceWorker.addEventListener("controllerchange", ask);
    }).catch(function () { if (label) label.textContent = ""; });
  }

  function init() {
    buildChrome();
    setupOffline();
    renderMath();
    buildMatcher();
    var main = document.querySelector(".page");
    if (main && !document.getElementById("glossary-list")) linkTerms(main, null);
    buildGlossaryPage();
    wirePopups();
    wireExercises();
    document.dispatchEvent(new Event("explainer:ready"));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
