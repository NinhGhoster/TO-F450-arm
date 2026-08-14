"""Verify the manuscript's reference list against CrossRef, then load Zotero.

Two jobs, in this order, because the second is only safe after the first:

1. **Verify.** Every reference in ``paper_v4_modal.md`` is looked up on
   CrossRef by title. The returned title is compared with the manuscript's,
   and year / journal are checked. Anything that does not match closely is
   reported rather than silently "corrected" — a reference that cannot be
   found is exactly what needs a human's eye before submission.

2. **Load.** Verified items are pushed into the running Zotero via its local
   connector API (the same endpoint the browser connector uses), with the DOI
   CrossRef returned attached.

Nothing is invented. Where CrossRef disagrees with the manuscript, both
versions are printed and the item is flagged, never quietly overwritten.

Usage::

    ./venv/bin/python -m sims.verify_references            # verify only
    ./venv/bin/python -m sims.verify_references --push     # verify + Zotero
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
import urllib.parse
import urllib.request

PAPER = "paper_v4_modal.md"
# CrossRef asks API users to identify themselves ("polite pool"). Set
# CROSSREF_MAILTO to your own address; the queries work without it but
# are rate-limited more aggressively.
MAILTO = os.environ.get("CROSSREF_MAILTO", "")
ZOTERO = "http://127.0.0.1:23119"

# Cited in the manuscript body but absent from the numbered list.
DANGLING = {
    "P2000": "Maximization of eigenvalues using topology optimization",
    "DO2007": ("Topological design of freely vibrating continuum structures "
               "for maximum values of simple and multiple eigenfrequencies "
               "and frequency gaps"),
}

REF_RE = re.compile(r"^(\d+)\.\s+(.*?)\s*\((\d{4}[a-z]?)\)\.\s*(.*)$")


def parse_refs(path=PAPER):
    out, in_refs = [], False
    for line in open(path, encoding="utf-8"):
        if re.match(r"^#+\s*References", line):
            in_refs = True
            continue
        if not in_refs:
            continue
        m = REF_RE.match(line.strip())
        if not m:
            continue
        num, authors, year, rest = m.groups()
        # Two layouts appear in the list:
        #   article: "Title. *Journal* vol(issue): pages."
        #   book:    "*Title.* Publisher, City."
        # The book form starts with the italic marker, so test it first —
        # otherwise the publisher ends up glued onto the title.
        book = re.match(r"^\*(.+?)\.?\*\s*(.*)$", rest)
        art = re.match(r"(.*?)\.\s*\*(.*?)\*(.*)$", rest)
        if book:
            title = book.group(1).strip().rstrip(".")
            venue, tail, is_book = book.group(2).strip().lstrip(". ").rstrip("."), "", True
        elif art:
            title, venue, tail = art.group(1), art.group(2), art.group(3)
            is_book = False
        else:
            title, venue, tail, is_book = rest.rstrip("."), "", "", True
        out.append(dict(n=int(num), authors=authors.strip(), year=year,
                        title=title.strip(), venue=venue.strip(),
                        tail=tail.strip(), is_book=is_book))
    return out


def crossref(title, rows=3):
    q = urllib.parse.urlencode({
        "query.bibliographic": title, "rows": rows,
        "select": "DOI,title,issued,container-title,type,author,volume,page",
    })
    req = urllib.request.Request(
        f"https://api.crossref.org/works?{q}",
        headers={"User-Agent": f"qcopter-arm-to/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)["message"]["items"]


def similarity(a, b):
    norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
    return difflib.SequenceMatcher(None, " ".join(norm(a)),
                                   " ".join(norm(b))).ratio()


def verify(refs):
    """Match each reference on CrossRef.

    Querying on the *full* bibliographic string (authors + title + venue +
    year) rather than the title alone matters: short generic titles like
    "Filters in topology optimization" otherwise match the wrong paper. The
    first author's surname must also appear in the returned author list
    before a hit is accepted, which is what separates a real match from a
    plausible-looking one.
    """
    rows = []
    for r in refs:
        query = f"{r['authors']} {r['title']} {r['venue']} {r['year']}"
        try:
            hits = crossref(query)
        except Exception as e:
            rows.append({**r, "status": "LOOKUP-FAILED", "note": str(e)[:60]})
            time.sleep(0.5)
            continue
        best, score = None, 0.0
        for h in hits:
            s = similarity(r["title"], (h.get("title") or [""])[0])
            if s > score:
                best, score = h, s
        if best is None:
            rows.append({**r, "status": "NOT-FOUND"})
            time.sleep(0.4)
            continue

        surname = re.split(r"[ ,]", r["authors"].strip())[0].lower()
        authors = " ".join((a.get("family") or "")
                           for a in (best.get("author") or [])).lower()
        author_ok = bool(surname) and surname in authors
        cy = str((best.get("issued", {}).get("date-parts") or [[None]])[0][0])

        if score >= 0.88 and author_ok:
            status = "OK" if cy == r["year"] else "YEAR-MISMATCH"
        elif score >= 0.88:
            status = "TITLE-ONLY"
        else:
            # CrossRef does not index books, theses or many proceedings, so a
            # miss here is not evidence the reference is wrong — only that it
            # cannot be machine-verified.
            status = "UNVERIFIED"
        rows.append({**r, "status": status, "score": round(score, 3),
                     "author_match": author_ok,
                     "doi": best.get("DOI") if score >= 0.88 else None,
                     "cr_title": (best.get("title") or [""])[0],
                     "cr_year": cy,
                     "cr_venue": (best.get("container-title") or [""])[0],
                     "cr_type": best.get("type")})
        time.sleep(0.4)
    return rows


def to_zotero_item(row):
    """CSL-ish JSON the Zotero connector accepts."""
    creators = []
    for chunk in re.split(r",\s*(?![A-Z]\.)", row["authors"]):
        chunk = chunk.replace("et al.", "").strip(" .")
        if not chunk:
            continue
        m = re.match(r"^([^,]+?)\s*,\s*(.+)$", chunk)
        if m:
            last, first = m.group(1), m.group(2)
        else:
            parts = chunk.split()
            last, first = parts[-1], " ".join(parts[:-1])
        creators.append({"creatorType": "author",
                         "firstName": first.strip(),
                         "lastName": last.strip()})
    is_book = row.get("is_book", False) or not row["venue"]
    item = {
        "itemType": "book" if is_book else "journalArticle",
        "title": row["title"],
        "creators": creators or [{"creatorType": "author",
                                  "firstName": "", "lastName": "Unknown"}],
        "date": row["year"],
        "extra": f"manuscript ref [{row['n']}]"
                 + (f"; verification: {row['status']}"
                    if row["status"] != "OK" else ""),
    }
    if row.get("doi"):
        item["DOI"] = row["doi"]
    if is_book:
        item["publisher"] = row["venue"] or ""
    else:
        item["publicationTitle"] = row["venue"]
        vol = re.search(r"(\d+)\s*\(", row["tail"]) or \
            re.search(r"^\s*(\d+)", row["tail"])
        if vol:
            item["volume"] = vol.group(1)
        pg = re.search(r":\s*([\d–\-]+)", row["tail"])
        if pg:
            item["pages"] = pg.group(1).replace("–", "-")
    return item


def push(items, collection="F450 arm TO (v4 manuscript)"):
    body = json.dumps({"items": items,
                       "uri": "https://github.com/qcopter-arm-to",
                       "sessionID": "qarm-refs"}).encode()
    req = urllib.request.Request(
        f"{ZOTERO}/connector/saveItems", data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "Zotero Connector"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.status, r.read()[:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    refs = parse_refs()
    print(f"parsed {len(refs)} references from {PAPER}\n")
    rows = verify(refs)

    bad = [r for r in rows if r["status"] != "OK"]
    print(f"{'#':>3} {'status':>14}  title")
    for r in rows:
        print(f"{r['n']:>3} {r['status']:>14}  {r['title'][:64]}")
        if r["status"] not in ("OK",):
            if r.get("cr_title"):
                print(f"{'':>19}  CrossRef: {r['cr_title'][:60]}"
                      f"  ({r.get('cr_year')})  score={r.get('score')}")
    print(f"\n{len(rows)-len(bad)}/{len(rows)} verified clean; "
          f"{len(bad)} need a look")

    print("\nDangling citations (used in the text, absent from the list):")
    for key, title in DANGLING.items():
        try:
            h = crossref(title, rows=1)
            if h:
                d = h[0]
                yr = (d.get("issued", {}).get("date-parts") or [[None]])[0][0]
                print(f"  [{key}] -> {(d.get('title') or [''])[0][:70]}")
                print(f"           {(d.get('container-title') or [''])[0]}"
                      f"  ({yr})  doi:{d.get('DOI')}")
        except Exception as e:
            print(f"  [{key}] lookup failed: {e}")

    with open("notes/reference_verification.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote notes/reference_verification.json")

    if a.push:
        items = [to_zotero_item(r) for r in rows]
        st, resp = push(items)
        print(f"\nZotero connector: HTTP {st}  {resp!r}")


if __name__ == "__main__":
    main()
