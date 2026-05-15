"""
Run with:  python pruning_test.py
"""

import time
import random
import networkx as nx
import numpy as np


# ── paste your method here exactly as-is ─────────────────────────────────────

def pruning_pagerank(G, alpha=0.85, max_iter=100, tol=1e-6,
                     warmup=10, prune_threshold_factor=0.5, stable_rounds=3):
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    M = np.zeros((N, N))
    for node in G.nodes():
        out_deg = G.out_degree(node)
        col = idx[node]
        if out_deg > 0:
            for succ in G.successors(node):
                M[idx[succ], col] = 1.0 / out_deg
        else:
            M[:, col] = 1.0 / N

    teleport    = np.full(N, (1.0 - alpha) / N)
    rank        = np.full(N, 1.0 / N)
    frozen      = set()
    frozen_list = []
    active_mask = np.ones(N, dtype=bool)
    low_count   = np.zeros(N, dtype=int)
    threshold   = prune_threshold_factor / N
    history     = []
    t0 = time.time()

    for it in range(max_iter):
        new_rank = alpha * M @ rank + teleport

        if frozen_list:
            new_rank[frozen_list] = rank[frozen_list]

        active_idx = np.where(active_mask)[0]
        error = np.abs(new_rank[active_idx] - rank[active_idx]).sum()
        rank = new_rank

        if it >= warmup:
            for i in active_idx:
                if rank[i] < threshold:
                    low_count[i] += 1
                    if low_count[i] >= stable_rounds:
                        frozen.add(i)
                        active_mask[i] = False
                else:
                    low_count[i] = 0
            frozen_list = list(frozen)

        history.append((it + 1, int(active_mask.sum())))

        if error < tol and it >= warmup:
            break

    elapsed = (time.time() - t0) * 1000
    return {nodes[i]: rank[i] for i in range(N)}, it + 1, elapsed, len(frozen), history


# ── reference: standard pagerank to verify correctness ───────────────────────

def standard_pagerank(G, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    M = np.zeros((N, N))
    for node in G.nodes():
        out_deg = G.out_degree(node)
        col = idx[node]
        if out_deg > 0:
            for succ in G.successors(node):
                M[idx[succ], col] = 1.0 / out_deg
        else:
            M[:, col] = 1.0 / N
    teleport = np.full(N, (1.0 - alpha) / N)
    rank = np.full(N, 1.0 / N)
    for it in range(max_iter):
        new_rank = alpha * M @ rank + teleport
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank
    return {nodes[i]: rank[i] for i in range(N)}


# ── helpers ───────────────────────────────────────────────────────────────────

def check(condition, label):
    """Print PASS/FAIL and return True/False."""
    symbol = "PASS" if condition else "FAIL"
    print(f"  [{symbol}] {label}")
    return condition


def build_web_graph(n, seed=42):
    """Synthetic web-like graph: ~5% hubs, ~50% normal, ~45% leaves."""
    rng = random.Random(seed)
    G = nx.DiGraph()
    n_hubs   = max(3, n // 20)
    n_normal = n // 2
    n_leaves = n - n_hubs - n_normal
    hubs    = [f"Hub_{i}"  for i in range(n_hubs)]
    normals = [f"Page_{i}" for i in range(n_normal)]
    leaves  = [f"Leaf_{i}" for i in range(n_leaves)]
    G.add_nodes_from(hubs + normals + leaves)
    for h1 in hubs:
        for h2 in hubs:
            if h1 != h2: G.add_edge(h1, h2)
    for p in normals:
        for h in rng.sample(hubs, k=min(3, len(hubs))): G.add_edge(p, h)
        others = [q for q in normals if q != p]
        for q in rng.sample(others, k=min(2, len(others))): G.add_edge(p, q)
    for h in hubs:
        for p in rng.sample(normals, k=min(5, len(normals))): G.add_edge(h, p)
    for leaf in leaves:
        G.add_edge(leaf, rng.choice(hubs))
        if rng.random() < 0.2: G.add_edge(rng.choice(normals), leaf)
    return G


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = []   # (test_name, passed)

    print("=" * 60)
    print("  PRUNING PAGERANK — MAIN TEST")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Return type and structure
    # ─────────────────────────────────────────────────────────────
    print("\nTEST 1 — Return values have correct types and shapes")

    G = nx.DiGraph()
    G.add_edges_from([("A","B"),("B","C"),("C","A"),("D","A")])

    scores, iters, elapsed_ms, n_frozen, history = pruning_pagerank(G)

    ok = True
    ok &= check(isinstance(scores, dict),        "scores is a dict")
    ok &= check(len(scores) == 4,                "scores has one entry per node (4)")
    ok &= check(isinstance(iters, int),          "iters is an int")
    ok &= check(iters >= 1,                      "iters >= 1")
    ok &= check(isinstance(elapsed_ms, float),   "elapsed_ms is a float")
    ok &= check(elapsed_ms >= 0,                 "elapsed_ms >= 0")
    ok &= check(isinstance(n_frozen, int),       "n_frozen is an int")
    ok &= check(isinstance(history, list),       "history is a list")
    ok &= check(len(history) == iters,           "history length == iters")
    ok &= check(all(len(h)==2 for h in history), "each history entry is (iter, active_count)")

    total = sum(scores.values())
    ok &= check(abs(total - 1.0) < 1e-6, f"scores sum to 1.0 (got {total:.8f})")

    results.append(("Return types & shapes", ok))

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Scores match standard PageRank (correctness)
    # ─────────────────────────────────────────────────────────────
    print("\nTEST 2 — Pruning scores match standard PageRank")
    print("         Graph: 10-node named web graph")
    print()

    G2 = nx.DiGraph()
    G2.add_edges_from([
        ("Hub_A","Hub_B"), ("Hub_B","Hub_A"),
        ("Hub_A","Hub_C"), ("Hub_C","Hub_A"),
        ("Hub_B","Hub_C"), ("Hub_C","Hub_B"),
        ("Page_1","Hub_A"), ("Page_1","Hub_B"),
        ("Page_2","Hub_B"), ("Page_2","Hub_C"),
        ("Page_3","Hub_A"), ("Page_3","Page_2"),
        ("Hub_A","Page_1"), ("Hub_B","Page_2"),
        ("Leaf_1","Hub_A"),
        ("Leaf_2","Hub_A"),
        ("Leaf_3","Hub_B"),
        ("Orphan","Hub_C"),
    ])

    ref   = standard_pagerank(G2)
    prune, iters2, ms2, nf2, hist2 = pruning_pagerank(G2)

    N2        = G2.number_of_nodes()
    threshold = 0.5 / N2   # = 0.05

    print(f"  {'Page':<12} {'Standard':>10} {'Pruning':>10} {'Diff':>12} {'Frozen?':>8}")
    print(f"  {'-'*56}")
    all_match = True
    for node in sorted(ref, key=lambda x: ref[x], reverse=True):
        diff   = abs(ref[node] - prune[node])
        frozen = "FROZEN" if ref[node] < threshold else ""
        ok_row = diff < 1e-9
        if not ok_row:
            all_match = False
        flag = "" if ok_row else "  <-- MISMATCH"
        print(f"  {node:<12} {ref[node]:>10.6f} {prune[node]:>10.6f} {diff:>12.2e} {frozen:>8}{flag}")

    print()
    print(f"  Threshold = 0.5 / {N2} = {threshold:.4f}")
    print(f"  Pages frozen: {nf2}/{N2}")
    print(f"  Iterations:   {iters2}")
    print()

    ok2 = all_match
    ok2 &= check(all_match,  "all scores match standard PageRank (diff < 1e-9)")
    ok2 &= check(nf2 == 5,   f"exactly 5 pages frozen (got {nf2})")
    ok2 &= check(
        all(ref[p] < threshold for p in ["Page_3","Leaf_1","Leaf_2","Leaf_3","Orphan"]),
        "correct pages are below threshold (Page_3, Leaf_1/2/3, Orphan)"
    )

    print()
    print("  EXPECTED output:")
    print("    Hub_B  ~ 0.270  (active)   Hub_A  ~ 0.253  (active)")
    print("    Hub_C  ~ 0.218  (active)   Page_2 ~ 0.098  (active)")
    print("    Page_1 ~ 0.087  (active)   Page_3 ~ 0.015  FROZEN")
    print("    Leaf_1 ~ 0.015  FROZEN     Leaf_2 ~ 0.015  FROZEN")
    print("    Leaf_3 ~ 0.015  FROZEN     Orphan ~ 0.015  FROZEN")

    results.append(("Correctness vs standard PageRank", ok2))

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Warmup respected — no freezing before warmup iters
    # ─────────────────────────────────────────────────────────────
    print("\nTEST 3 — Warmup: no pages frozen before warmup iterations")

    G3 = build_web_graph(100)
    _, _, _, _, hist3 = pruning_pagerank(G3, warmup=10)

    # During warmup (iterations 0–9) active count must stay at N
    N3 = G3.number_of_nodes()
    warmup_counts = [h[1] for h in hist3[:10]]   # first 10 entries
    ok3 = check(
        all(c == N3 for c in warmup_counts),
        f"active count = {N3} (all pages) during all warmup iterations"
    )
    results.append(("Warmup respected", ok3))

    # ─────────────────────────────────────────────────────────────
    # TEST 4: History shrinks after warmup (pages do get frozen)
    # ─────────────────────────────────────────────────────────────
    print("\nTEST 4 — Active count shrinks after warmup on leaf-heavy graph")

    G4 = build_web_graph(200)
    _, _, _, nf4, hist4 = pruning_pagerank(G4, warmup=10)

    post_warmup_counts = [h[1] for h in hist4[10:]]
    shrinks = any(post_warmup_counts[i] < post_warmup_counts[i-1]
                  for i in range(1, len(post_warmup_counts)))

    ok4 = check(nf4 > 0,   f"at least one page was frozen (got {nf4})")
    ok4 &= check(shrinks,  "active count decreases at least once after warmup")
    results.append(("Active count shrinks post-warmup", ok4))

    # ─────────────────────────────────────────────────────────────
    # TEST 5: Accuracy and speed across graph sizes
    # ─────────────────────────────────────────────────────────────
    print("\nTEST 5 — Accuracy and timing: n = 50, 500, 2000")
    print()
    print(f"  {'N':>5}  {'Std time':>9}  {'Prune time':>10}  "
          f"{'Frozen %':>9}  {'L1 error':>10}  {'Speedup':>8}")
    print(f"  {'-'*60}")

    ok5 = True
    for n in [50, 500, 2000]:
        G = build_web_graph(n)
        ref_sc = standard_pagerank(G)

        t0 = time.time(); _ = standard_pagerank(G); std_ms = (time.time()-t0)*1000
        prune_sc, _, prune_ms, nf, _ = pruning_pagerank(G)

        l1      = sum(abs(ref_sc[p] - prune_sc[p]) for p in G.nodes())
        speedup = std_ms / prune_ms if prune_ms > 0 else 1.0
        pct     = 100.0 * nf / n
        row_ok  = l1 < 1e-6
        if not row_ok:
            ok5 = False
        flag = "OK" if row_ok else "FAIL"

        print(f"  {n:>5}  {std_ms:>8.1f}ms  {prune_ms:>9.1f}ms  "
              f"{pct:>8.1f}%  {l1:>10.2e}  {speedup:>7.2f}x  [{flag}]")

    print()
    print("  EXPECTED:")
    print("    L1 error = 0.00e+00 for all sizes")
    print("    ~56% pages frozen at n=50 (mostly leaves)")
    print("    Speedup grows with graph size")

    ok5 = check(ok5, "L1 error < 1e-6 for all graph sizes")
    results.append(("Accuracy across sizes", ok5))

    # ─────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        symbol = "PASS" if ok else "FAIL"
        print(f"  [{symbol}] {name}")
    print()
    print(f"  {passed}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()