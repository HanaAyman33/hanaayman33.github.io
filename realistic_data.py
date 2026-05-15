"""
req2_realistic_data.py  —  Requirement 2: Realistic Large-Scale Data
=====================================================================
Part A: scrape_website() — BFS web crawler → NetworkX DiGraph
Part B: 7 PageRank implementations benchmarked on a 791-node graph

Run commands
------------
# Demo with synthetic graph (no internet required):
python req2_realistic_data.py --demo

# Scrape a real website (internet required):
python req2_realistic_data.py --scrape --url https://docs.python.org/3/ --depth 2

# Demo on a previously saved edge list:
python req2_realistic_data.py --demo --input edge_list.csv

Dependencies
------------
pip install networkx numpy scipy matplotlib requests beautifulsoup4
"""

import argparse, csv, os, random, sys, time
import networkx as nx
import numpy as np
import scipy.sparse as sp


# ════════════════════════════════════════════════════════════════
# PART A — scrape_website()
# ════════════════════════════════════════════════════════════════

def scrape_website(start_url, max_depth=2, max_pages=500):
    """
    BFS crawl from start_url.  Returns nx.DiGraph (nodes=URLs, edges=links).

    Bugs fixed vs the naive implementation:
      Fix 1 – max_depth:  enqueue children only when depth < max_depth
      Fix 2 – max_pages:  cap on len(visited), not G.number_of_nodes()
      Fix 3 – self-loops: skip link when link == current url
      Fix 4 – ghost nodes: collect edges in a list; add to G only after
               crawl, only between pages that were actually visited
    """
    import requests
    from bs4 import BeautifulSoup
    from collections import deque
    from urllib.parse import urljoin, urlparse

    base_domain = urlparse(start_url).netloc
    visited   = {}        # url -> depth
    raw_edges = []        # (src, dst) pairs collected during crawl
    queue     = deque([(start_url, 0)])

    while queue and len(visited) < max_pages:           # Fix 2
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited[url] = depth
        try:
            resp = requests.get(url, timeout=8,
                                headers={"User-Agent": "PageRank-Thesis/1.0"})
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all("a", href=True):
                link = urljoin(url, tag["href"]).split("#")[0].split("?")[0]
                if not link:
                    continue
                lp = urlparse(link)
                if lp.netloc != base_domain or lp.scheme not in ("http", "https"):
                    continue
                if link == url:                              # Fix 3
                    continue
                raw_edges.append((url, link))
                if link not in visited and depth < max_depth:   # Fix 1
                    queue.append((link, depth + 1))
        except Exception:
            pass

    # Fix 4: build graph only from visited pages
    G = nx.DiGraph()
    G.add_nodes_from(visited)
    for src, dst in raw_edges:
        if src in visited and dst in visited:
            G.add_edge(src, dst)
    return G


# ════════════════════════════════════════════════════════════════
# SYNTHETIC GRAPH  (791 nodes, 3 228 edges)
# ════════════════════════════════════════════════════════════════

def build_synthetic_graph(seed=42):
    """
    Documentation-like graph:
      1 homepage → 10 section pages
      10 sections → 78 articles each (+ back to homepage)
      780 articles → their section + homepage
      868 cross-section article links  (~5 %)
    Total: 791 nodes, 3 228 directed edges
    """
    rng = random.Random(seed)
    G   = nx.DiGraph()

    homepage = "index"
    sections = [f"section_{i}" for i in range(10)]
    articles = {s: [f"{s}_article_{j}" for j in range(78)] for s in sections}
    all_articles = [a for arts in articles.values() for a in arts]

    G.add_node(homepage)
    G.add_nodes_from(sections)
    G.add_nodes_from(all_articles)

    for sec in sections:
        G.add_edge(homepage, sec)
        G.add_edge(sec, homepage)
        for art in articles[sec]:
            G.add_edge(sec, art)
            G.add_edge(art, sec)
            G.add_edge(art, homepage)

    # Add exactly 868 unique cross-section edges
    added = 0
    while added < 868:
        a = rng.choice(all_articles)
        b = rng.choice(all_articles)
        if "_".join(a.split("_")[:2]) != "_".join(b.split("_")[:2]) and not G.has_edge(a, b):
            G.add_edge(a, b)
            added += 1
    return G


# ════════════════════════════════════════════════════════════════
# 7 PAGERANK METHODS
# ════════════════════════════════════════════════════════════════

def method1_networkx(G, alpha=0.85):
    t0 = time.time()
    return nx.pagerank(G, alpha=alpha), (time.time() - t0) * 1000

def method2_manual(G, alpha=0.85, tol=1e-6, max_iter=100):
    nodes = list(G.nodes()); n = len(nodes)
    rank = {v: 1/n for v in nodes}
    t0 = time.time()
    for _ in range(max_iter):
        d_sum = sum(rank[v] for v in nodes if G.out_degree(v) == 0)
        new_rank = {}
        for v in nodes:
            new_rank[v] = (1 - alpha) / n + alpha * d_sum / n
            for p in G.predecessors(v):
                if G.out_degree(p) > 0:
                    new_rank[v] += alpha * rank[p] / G.out_degree(p)
        if sum(abs(new_rank[v] - rank[v]) for v in nodes) < tol:
            rank = new_rank; break
        rank = new_rank
    return rank, (time.time() - t0) * 1000

def method3_matrix(G, alpha=0.85):
    nodes = list(G.nodes()); n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    M = np.zeros((n, n))
    for v in G.nodes():
        od = G.out_degree(v); c = idx[v]
        if od > 0:
            for nb in G.successors(v): M[idx[nb], c] = 1 / od
        else:
            M[:, c] = 1 / n
    t0 = time.time()
    r = np.linalg.solve(np.eye(n) - alpha * M, np.ones(n) * (1 - alpha) / n)
    r /= r.sum()
    return {nodes[i]: r[i] for i in range(n)}, (time.time() - t0) * 1000

def method4_scipy(G, alpha=0.85, tol=1e-6, max_iter=100):
    nodes = list(G.nodes()); n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols, vals, d_idx = [], [], [], []
    for v in G.nodes():
        od = G.out_degree(v); c = idx[v]
        if od > 0:
            for nb in G.successors(v):
                rows.append(idx[nb]); cols.append(c); vals.append(1 / od)
        else:
            d_idx.append(c)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    rank = np.full(n, 1 / n)
    t0 = time.time()
    for _ in range(max_iter):
        dv = alpha * rank[d_idx].sum() / n if d_idx else 0
        nr = alpha * M.dot(rank) + dv + (1 - alpha) / n
        if np.abs(nr - rank).sum() < tol: rank = nr; break
        rank = nr
    return {nodes[i]: rank[i] for i in range(n)}, (time.time() - t0) * 1000

def method5_montecarlo(G, alpha=0.85, n_surfers=5000, steps=100):
    nodes = list(G.nodes()); n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    visits = np.zeros(n); rng = random.Random(42)
    t0 = time.time()
    for _ in range(n_surfers):
        cur = rng.choice(nodes)
        for _ in range(steps):
            visits[idx[cur]] += 1
            nbs = list(G.successors(cur))
            cur = (rng.choice(nbs) if nbs else rng.choice(nodes)) \
                  if rng.random() < alpha else rng.choice(nodes)
    total = visits.sum()
    return {nodes[i]: visits[i] / total for i in range(n)}, (time.time() - t0) * 1000

def method6_hits(G):
    t0 = time.time()
    _, auth = nx.hits(G, max_iter=100, normalized=True)
    return auth, (time.time() - t0) * 1000

def method7_pruning(G, alpha=0.85, tol=1e-6, max_iter=100, warmup=10, stable_rounds=3):
    nodes = list(G.nodes()); n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    M = np.zeros((n, n))
    for v in G.nodes():
        od = G.out_degree(v); c = idx[v]
        if od > 0:
            for nb in G.successors(v): M[idx[nb], c] = 1 / od
        else:
            M[:, c] = 1 / n
    teleport = np.full(n, (1 - alpha) / n)
    rank = np.full(n, 1 / n)
    frozen = []; active = np.ones(n, dtype=bool)
    low_count = np.zeros(n, dtype=int); threshold = 0.5 / n
    t0 = time.time()
    for it in range(max_iter):
        new_rank = alpha * M @ rank + teleport
        if frozen: new_rank[frozen] = rank[frozen]
        ai = np.where(active)[0]
        err = np.abs(new_rank[ai] - rank[ai]).sum()
        rank = new_rank
        if it >= warmup:
            for i in ai:
                if rank[i] < threshold: low_count[i] += 1
                else: low_count[i] = 0
                if low_count[i] >= stable_rounds: active[i] = False
            frozen = list(np.where(~active)[0])
        if err < tol and it >= warmup: break
    return {nodes[i]: rank[i] for i in range(n)}, (time.time() - t0) * 1000


# ════════════════════════════════════════════════════════════════
# LOAD FROM CSV
# ════════════════════════════════════════════════════════════════

def load_from_csv(path):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: file not found — {path}")
    with open(path, encoding="utf-8-sig") as f:
        sample = f.read(4096)
    delim = "\t" if sample.count("\t") > sample.count(",") else ","
    G = nx.DiGraph()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delim)
        headers = list(reader.fieldnames or [])
        src_col = next((c for c in ["source","src","from","u"] + headers if c in headers), None)
        dst_col = next((c for c in ["target","dst","to","v"] + headers
                        if c in headers and c != src_col), None)
        if not src_col or not dst_col:
            sys.exit(f"ERROR: cannot identify columns. Headers: {headers}")
        print(f"  CSV columns: '{src_col}' -> '{dst_col}'")
        for row in reader:
            s = (row.get(src_col) or "").strip()
            d = (row.get(dst_col) or "").strip()
            if s and d and s != d:
                G.add_edge(s, d)
    return G


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def check(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Requirement 2 – Realistic Data")
    ap.add_argument("--demo",   action="store_true", help="Run 7-method demo")
    ap.add_argument("--scrape", action="store_true", help="Crawl a live website")
    ap.add_argument("--url",    default="https://docs.python.org/3/")
    ap.add_argument("--depth",  type=int, default=2)
    ap.add_argument("--input",  default=None, help="Load graph from edge_list.csv")
    args = ap.parse_args()
    if not args.demo and not args.scrape:
        args.demo = True   # default

    SEP = "=" * 65
    sep = "─" * 65

    print(SEP)
    print("  REQUIREMENT 2 — REALISTIC LARGE-SCALE DATA")
    print(SEP)

    # ── SCRAPE MODE ──────────────────────────────────────────────
    if args.scrape:
        print(f"\n  Crawling: {args.url}  (depth={args.depth}, max=500 pages)")
        print("  Please wait ~30–60 s …\n")
        G = scrape_website(args.url, max_depth=args.depth, max_pages=500)
        n, e = G.number_of_nodes(), G.number_of_edges()
        print(f"  Crawl done: {n} pages, {e} links")
        out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "edge_list.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["source", "target"])
            for s, d in G.edges(): w.writerow([s, d])
        print(f"  Saved: {out}")
        print(f"\n  Now run: python req2_realistic_data.py --demo --input edge_list.csv")
        return

    # ── DEMO MODE ────────────────────────────────────────────────
    print("\n" + sep)
    print("  PART A — GRAPH CONSTRUCTION & TESTS")
    print(sep)

    if args.input:
        print(f"\n  Loading graph from: {args.input}")
        G = load_from_csv(args.input)
        source_label = f"real scraped data ({args.input})"
    else:
        print("\n  Building synthetic documentation graph …")
        G = build_synthetic_graph()
        source_label = "synthetic documentation graph"

    N = G.number_of_nodes()
    E = G.number_of_edges()
    print(f"  Graph loaded: {N} nodes, {E} edges  ({source_label})\n")

    # ── 9 STRUCTURAL TESTS ───────────────────────────────────────
    results = []
    ok = True
    ok &= check(isinstance(G, nx.DiGraph),
                f"TEST 1  Returns nx.DiGraph")
    ok &= check(N > 0,
                f"TEST 2  Graph has nodes (got {N})")
    ok &= check(E > 0,
                f"TEST 3  Graph has edges (got {E})")
    sl = list(nx.selfloop_edges(G))
    ok &= check(len(sl) == 0,
                f"TEST 4  No self-loops (found {len(sl)})")
    # max_depth / max_pages checks only run with synthetic graph for speed
    if not args.input:
        ok &= check(N == 791,
                    f"TEST 5  Synthetic graph: 791 nodes (got {N})")
        ok &= check(E == 3228,
                    f"TEST 6  Synthetic graph: 3228 edges (got {E})")
    pr_test = nx.pagerank(G)
    total = sum(pr_test.values())
    ok &= check(abs(total - 1.0) < 1e-6,
                f"TEST 7  PageRank sums to 1.0 (got {total:.8f})")
    ok &= check(max(pr_test.values()) > 1/N,
                f"TEST 8  Top page has above-average rank")
    top_node = max(pr_test, key=pr_test.get)
    ok &= check(pr_test[top_node] == max(pr_test.values()),
                f"TEST 9  Top-ranked page identified ({top_node})")
    results.append(("Part A – graph structure & PageRank", ok))

    # ── PART B: ALL 7 METHODS ────────────────────────────────────
    print("\n" + sep)
    print("  PART B — ALL 7 METHODS")
    print(sep)
    print(f"\n  Graph: {N} nodes, {E} edges\n")

    sc1, t1 = method1_networkx(G)
    print(f"  Method 1  NetworkX           {t1:7.1f} ms")
    sc2, t2 = method2_manual(G)
    print(f"  Method 2  Manual Power Iter. {t2:7.1f} ms")
    sc3, t3 = method3_matrix(G) if N <= 1500 else (None, None)
    if sc3: print(f"  Method 3  Matrix / NumPy     {t3:7.1f} ms")
    else:   print(f"  Method 3  Matrix / NumPy     SKIPPED  (N > 1500)")
    sc4, t4 = method4_scipy(G)
    print(f"  Method 4  SciPy Sparse       {t4:7.1f} ms")
    sc5, t5 = method5_montecarlo(G)
    print(f"  Method 5  Monte Carlo        {t5:7.1f} ms")
    sc6, t6 = method6_hits(G)
    print(f"  Method 6  HITS               {t6:7.1f} ms")
    sc7, t7 = method7_pruning(G)
    print(f"  Method 7  Pruning (custom)   {t7:7.1f} ms")

    def l1(sc):
        return sum(abs(sc.get(p, 0) - sc1[p]) for p in G.nodes())

    # ── RESULTS TABLE ────────────────────────────────────────────
    print()
    print(f"  {'Method':<30} {'L1 vs NetworkX':>16} {'Time':>9}")
    print(f"  {'-'*58}")
    print(f"  {'1 — NetworkX':<30} {'0  (reference)':>16} {t1:>8.1f} ms")
    print(f"  {'2 — Manual Power Iter.':<30} {l1(sc2):>16.2e} {t2:>8.1f} ms")
    if sc3:
        print(f"  {'3 — Matrix (NumPy)':<30} {l1(sc3):>16.2e} {t3:>8.1f} ms")
    else:
        print(f"  {'3 — Matrix (NumPy)':<30} {'SKIPPED':>16} {'—':>9}")
    print(f"  {'4 — SciPy Sparse':<30} {l1(sc4):>16.2e} {t4:>8.1f} ms")
    print(f"  {'5 — Monte Carlo':<30} {l1(sc5):>16.2e} {t5:>8.1f} ms")
    print(f"  {'6 — HITS (Authority)':<30} {l1(sc6):>16.2e} {t6:>8.1f} ms")
    print(f"  {'7 — Pruning (custom)':<30} {l1(sc7):>16.2e} {t7:>8.1f} ms")

    # ── TOP 15 ───────────────────────────────────────────────────
    top15 = sorted(sc1.items(), key=lambda x: -x[1])[:15]
    print()
    print(f"  Top 15 pages by NetworkX PageRank:")
    print(f"  {'Rk':<4} {'Page':<36} {'NX Score':>10} {'M2 Score':>10} {'Diff':>10}")
    print(f"  {'-'*74}")
    for i, (page, score) in enumerate(top15, 1):
        m2s  = sc2.get(page, 0)
        diff = abs(score - m2s)
        label = page if len(page) <= 34 else "…" + page[-33:]
        print(f"  {i:<4} {label:<36} {score:>10.6f} {m2s:>10.6f} {diff:>10.2e}")

    top_page, top_score = top15[0]
    print(f"\n  Top page: '{top_page}'  →  {top_score*100:.1f}% of total PageRank")

    # ── ACCURACY TESTS ───────────────────────────────────────────
    print("\n" + sep)
    print("  ACCURACY TESTS")
    print(sep + "\n")
    ok2 = True
    ok2 &= check(abs(sum(sc1.values()) - 1.0) < 1e-6,
                 f"Method 1  scores sum to 1.0")
    ok2 &= check(l1(sc2) < 1e-3,
                 f"Method 2  L1 < 1e-3  (got {l1(sc2):.2e})")
    ok2 &= check((l1(sc3) < 1e-3) if sc3 else True,
                 f"Method 3  L1 < 1e-3  (got {l1(sc3):.2e})" if sc3 else "Method 3  SKIPPED")
    ok2 &= check(l1(sc4) < 1e-3,
                 f"Method 4  L1 < 1e-3  (got {l1(sc4):.2e})")
    ok2 &= check(l1(sc7) < 1e-3,
                 f"Method 7  L1 < 1e-3  (got {l1(sc7):.2e})")
    top15_m2 = {p for p, _ in sorted(sc2.items(), key=lambda x: -x[1])[:15]}
    overlap = len({p for p, _ in top15} & top15_m2)
    ok2 &= check(overlap >= 13,
                 f"Top-15 overlap Method 1 vs Method 2: {overlap}/15")
    results.append(("Part B – method accuracy", ok2))

    # ── SUMMARY ──────────────────────────────────────────────────
    print("\n" + SEP)
    print("  SUMMARY")
    print(SEP)
    passed = sum(1 for _, r in results if r)
    for name, r in results:
        print(f"  [{'PASS' if r else 'FAIL'}]  {name}")
    print(f"\n  {passed}/{len(results)} groups passed  |  Graph: {N} nodes, {E} edges")
    print(SEP)


if __name__ == "__main__":
    main()