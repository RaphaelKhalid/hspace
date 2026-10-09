/* Static checks for explainer pages. Usage (from the explainer folder):
     node tools/check_page.js 00.html 01.html ...      (no args = every NN.html)
   Checks:
     - every \( \), \[ \] and $$ $$ math segment renders in KaTeX with throwOnError
     - inline <script> blocks parse as JavaScript
     - the page has data-level, a "Symbols on this page" table, figures with captions, exercises
     - house style: no em dashes in visible text
   Exit code 1 if any error. */
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");
global.window = global;
global.document = undefined;
const katex = require(path.join(ROOT, "assets", "katex", "katex.min.js"));

function stripTags(s) { return s.replace(/<script[\s\S]*?<\/script>/gi, "").replace(/<style[\s\S]*?<\/style>/gi, ""); }

function decode(s) {
  return s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/&nbsp;/g, " ").replace(/&quot;/g, '"');
}

function checkFile(file) {
  const errors = [], warns = [];
  const html = fs.readFileSync(path.join(ROOT, file), "utf8");
  const body = stripTags(html);

  // math
  const re = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g;
  let m, nmath = 0;
  while ((m = re.exec(body)) !== null) {
    const tex = decode(m[1] || m[2] || m[3]);
    const display = !m[3];
    nmath++;
    try { katex.renderToString(tex, { throwOnError: true, displayMode: display, strict: "ignore" }); }
    catch (e) { errors.push("KaTeX: " + e.message.split("\n")[0] + "   in: " + tex.slice(0, 80).replace(/\s+/g, " ")); }
  }
  // stray single-dollar math is not rendered: warn on $x$-looking patterns
  const stray = body.replace(/\$\$[\s\S]+?\$\$/g, "").match(/\$[A-Za-z\\][^$<]{0,40}\$/g);
  if (stray) warns.push("possible single-$ math (not rendered; use \\( \\)): " + stray.slice(0, 3).join(" | "));

  // scripts
  const sre = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
  let s, nscript = 0;
  while ((s = sre.exec(html)) !== null) {
    nscript++;
    try { new Function(s[1]); } catch (e) { errors.push("JS syntax error in inline script #" + nscript + ": " + e.message); }
  }

  // structure
  const isLevel = /^\d\d\.html$/.test(path.basename(file));
  if (isLevel) {
    if (!/<body[^>]*data-level="\d+"/.test(html)) errors.push("missing <body data-level=\"N\">");
    if (!/Symbols on this page/.test(html)) errors.push("missing 'Symbols on this page' table");
    const figs = (html.match(/<figure class="sim"/g) || []).length;
    const caps = (html.match(/<figure class="sim"[^>]*>[\s\S]*?<figcaption>/g) || []).length;
    if (figs < 2) warns.push("only " + figs + " simulation figure(s)");
    if (caps !== figs) errors.push("a figure.sim lacks a figcaption");
    const ids = [...html.matchAll(/<figure class="sim" id="([^"]+)"/g)].map(x => x[1]);
    ids.forEach(id => { if (!html.includes('getElementById("' + id + '")') && !html.includes("getElementById('" + id + "')")) warns.push("figure #" + id + " is never wired up by a script"); });
    const ex = (html.match(/class="ex"/g) || []).length;
    if (ex < 5) warns.push("only " + ex + " exercises");
    ["assets/katex/katex.min.js", "assets/katex/auto-render.min.js", "assets/glossary.js", "assets/sim.js", "assets/site.js"].forEach(src => {
      if (!html.includes('src="' + src + '"')) errors.push("missing script " + src);
    });
  }

  // house style
  const visible = body.replace(/<[^>]+>/g, " ");
  const em = visible.match(/.{0,30}—.{0,30}/g);
  if (em) errors.push("em dash in text (house style): " + em.slice(0, 3).join(" | "));
  // also in scripts' visible strings
  const scriptText = (html.match(/<script(?![^>]*\bsrc=)[^>]*>[\s\S]*?<\/script>/gi) || []).join("\n");
  if (/—/.test(scriptText)) errors.push("em dash inside a script string (house style)");

  const kb = (Buffer.byteLength(html) / 1024).toFixed(0);
  return { file, errors, warns, info: `${nmath} math, ${nscript} inline scripts, ${kb} KB` };
}

let files = process.argv.slice(2);
if (!files.length) files = fs.readdirSync(ROOT).filter(f => /^\d\d\.html$/.test(f)).sort();
let bad = 0;
files.forEach(f => {
  const r = checkFile(f);
  console.log((r.errors.length ? "FAIL " : "ok   ") + r.file + "  (" + r.info + ")");
  r.errors.forEach(e => console.log("   error: " + e));
  r.warns.forEach(w => console.log("   warn:  " + w));
  if (r.errors.length) bad++;
});
process.exit(bad ? 1 : 0);
