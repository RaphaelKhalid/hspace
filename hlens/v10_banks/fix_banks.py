"""Fix the two banks whose punctuation read site preceded the factor words (validator: BLOCKING), give per-cell natural
sites where the shared one carried no factor information, and use the conservative cluster unit for relational.
Pre-data (no model has seen any bank). Run once:  python fix_banks.py   (from v10_banks/)."""
import io
import json
import re

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")


def ids(s):
    return tok(s, add_special_tokens=False)["input_ids"]


def is_p(t):
    s = tok.decode([t]).strip()
    return s != "" and not any(ch.isalnum() for ch in s)


def check(item):
    for k, p in item["prompts"].items():
        t = ids(p)
        assert len(t) <= 40 and not p.endswith(" "), (item["item"], k, p)
        assert any(is_p(x) for x in t[:-1]) and not is_p(t[-1]), (item["item"], k, p)
        ns = item["natural_site"][k] if isinstance(item["natural_site"], dict) else item["natural_site"]
        assert p.count(ns) == 1, (item["item"], k, ns, p)
    assert ids(item["X"])[0] != ids(item["Y"])[0]


def load(f):
    return [json.loads(x) for x in io.open(f, encoding="utf-8") if x.strip()]


def save(f, items):
    for it in items:
        check(it)
    io.open(f, "w", encoding="utf-8", newline="\n").write("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    print(f, len(items), "items ok")


# entity_attribute: "What is the <rel_b> of <entity_a>? It is"  (the '?' follows both factor words)
E = load("entity_attribute.jsonl")
for it in E:
    m = it["meta"]
    ents, rels = m["entities"], [m["rel0"], m["rel1"]]
    it["prompts"] = {f"{a}{b}": f"What is the {rels[b]} of {ents[a]}? It is" for a in (0, 1) for b in (0, 1)}
    it["natural_site"] = {f"{a}{b}": f" {ents[a]}?" .rstrip("?") for a in (0, 1) for b in (0, 1)}
    it["template_fix"] = "pre-data: template changed from 'Fact: The <rel> of <entity> is' so the punctuation site follows the factor words"
save("entity_attribute.jsonl", E)

# binding_backup: the asked name now precedes the last punctuation (',')
B = load("binding_backup.jsonl")
for it in B:
    m = it["meta"]
    P, Q, c1, c2, obj = m["NameP"], m["NameQ"], m["c1"], m["c2"], m["object"]
    pr, ns = {}, {}
    for a in (0, 1):
        for b in (0, 1):
            cP, cQ = (c1, c2) if b == 0 else (c2, c1)
            name = P if a == 0 else Q
            pr[f"{a}{b}"] = f"{P} has a {cP} {obj}. {Q} has a {cQ} {obj}. Asked about {name}'s {obj}, the color is"
            ns[f"{a}{b}"] = f" about {name}"
    it["prompts"], it["natural_site"] = pr, ns
    it["template_fix"] = "pre-data: asked name moved before the last punctuation (',') so the site carries both factors"
save("binding_backup.jsonl", B)

# null_unrelated: natural site = the unrelated variant word (per cell)
N = load("null_unrelated.jsonl")
for it in N:
    ns = {}
    for k, p in it["prompts"].items():
        sent2 = [s for s in p.split(".") if s.strip()][1]
        w = sent2.strip().split()[-1]
        ns[k] = f" {w}"          # the bare variant word (a trailing "." would coincide with the punctuation site)
    it["natural_site"] = ns
save("null_unrelated.jsonl", N)

# relational_composition: conservative cluster unit (entity family)
Rr = load("relational_composition.jsonl")
for it in Rr:
    it["cluster_scene"] = it["cluster"]
    it["cluster"] = it["meta"]["entity_family"]
save("relational_composition.jsonl", Rr)
