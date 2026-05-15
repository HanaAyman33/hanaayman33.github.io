"""
search_engine_test.py  —  Main test method for Requirement 3: SearchEngine
==========================================================================
Tests the SearchEngine class across 8 structured test cases.

Run:
    python search_engine_test.py

Dependencies:
    pip install networkx numpy
"""

import numpy as np
import networkx as nx
from datetime import datetime, timedelta


# ══════════════════════════════════════════════════════════════════
# PASTE SearchEngine CLASS HERE (from req345_search_seo.py)
# ══════════════════════════════════════════════════════════════════

class SearchEngine:
    def __init__(self, G, page_content=None, page_dates=None):
        self.G          = G
        self.nodes      = list(G.nodes())
        self.N          = len(self.nodes)
        self.page_content = page_content or {}
        self.page_dates   = page_dates   or {}
        self._base_pr   = nx.pagerank(G, alpha=0.85)
        self._recency   = self._build_recency()

    def _build_recency(self):
        now = datetime.now()
        scores = {}
        for p in self.nodes:
            if p in self.page_dates:
                days = max(0, (now - self.page_dates[p]).days)
                scores[p] = np.exp(-0.01 * days)
            else:
                scores[p] = 0.5
        mn, mx = min(scores.values()), max(scores.values())
        return {p: (s-mn)/(mx-mn) if mx > mn else s for p, s in scores.items()}

    def _relevance(self, keyword):
        kw = keyword.lower().strip()
        scores = {}
        for p in self.nodes:
            text  = str(self.page_content.get(p, str(p))).lower()
            count = sum(text.count(w) for w in kw.split())
            scores[p] = float(count)
        total = sum(scores.values())
        return {p: s/total if total > 0 else 0.0 for p, s in scores.items()}

    def _build_M(self):
        idx = {n: i for i, n in enumerate(self.nodes)}
        M   = np.zeros((self.N, self.N))
        for node in self.nodes:
            od  = self.G.out_degree(node)
            col = idx[node]
            if od > 0:
                for nb in self.G.successors(node):
                    M[idx[nb], col] = 1.0 / od
            else:
                M[:, col] = 1.0 / self.N
        return M

    def _topic_pr(self, seeds, alpha=0.85):
        idx      = {n: i for i, n in enumerate(self.nodes)}
        teleport = np.zeros(self.N)
        valid    = [s for s in seeds if s in idx]
        for s in valid:
            teleport[idx[s]] += 1.0 / len(valid)
        rank = np.full(self.N, 1.0 / self.N)
        M    = self._build_M()
        for _ in range(100):
            new_rank = alpha * M @ rank + (1 - alpha) * teleport
            if np.abs(new_rank - rank).sum() < 1e-6:
                break
            rank = new_rank
        return {self.nodes[i]: rank[i] for i in range(self.N)}

    def search(self, keyword='', alpha=0.85, relevance_weight=0.0,
               recency_weight=0.0, topic_weight=0.0,
               topic_seeds=None, top_k=10):
        pr  = nx.pagerank(self.G, alpha=alpha)
        rel = self._relevance(keyword)
        rec = self._recency
        top = self._topic_pr(topic_seeds or []) if topic_weight > 0 else pr
        w_pr = max(0.0, 1.0 - relevance_weight - recency_weight - topic_weight)
        results = []
        for p in self.nodes:
            score = (w_pr             * pr.get(p, 0)
                   + relevance_weight * rel.get(p, 0)
                   + recency_weight   * rec.get(p, 0)
                   + topic_weight     * top.get(p, 0))
            results.append({'page': p, 'score': score, 'pr': pr.get(p, 0),
                            'relevance': rel.get(p, 0), 'recency': rec.get(p, 0)})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


# ══════════════════════════════════════════════════════════════════
# TEST GRAPH FIXTURE
# ══════════════════════════════════════════════════════════════════

def build_test_graph():
    """
    15-node website graph covering Python tutorials, ML, APIs, blogs.
    Same graph used in the document's Scenario results.
    """
    G = nx.DiGraph()
    edges = [
        ('homepage',        'python_tutorial'),
        ('homepage',        'data_science_intro'),
        ('homepage',        'api_guide'),
        ('homepage',        'blog_index'),
        ('python_tutorial', 'homepage'),
        ('python_tutorial', 'ml_algorithms'),
        ('python_tutorial', 'api_guide'),
        ('data_science_intro','ml_algorithms'),
        ('data_science_intro','ml_deep_learning'),
        ('data_science_intro','homepage'),
        ('ml_algorithms',   'ml_deep_learning'),
        ('ml_algorithms',   'python_tutorial'),
        ('ml_deep_learning','blog_ml_news'),
        ('ml_deep_learning','homepage'),
        ('api_guide',       'deployment_guide'),
        ('api_guide',       'homepage'),
        ('deployment_guide','homepage'),
        ('deployment_guide','api_guide'),
        ('blog_index',      'blog_python_tips'),
        ('blog_index',      'blog_ml_news'),
        ('blog_index',      'homepage'),
        ('blog_python_tips','python_tutorial'),
        ('blog_python_tips','blog_index'),
        ('blog_ml_news',    'ml_algorithms'),
        ('blog_ml_news',    'blog_index'),
        ('db_guide',        'api_guide'),
        ('db_guide',        'homepage'),
        ('advanced_python', 'python_tutorial'),
        ('advanced_python', 'api_guide'),
    ]
    G.add_edges_from(edges)
    return G

def build_page_content():
    return {
        'homepage':         'python programming home welcome python tutorial guide',
        'python_tutorial':  'python tutorial beginner python programming python basics',
        'data_science_intro':'python data science machine learning introduction python',
        'ml_algorithms':    'machine learning algorithms python classification regression',
        'ml_deep_learning': 'deep learning neural networks machine learning python keras',
        'api_guide':        'api guide rest python requests http integration',
        'deployment_guide': 'deployment docker kubernetes cloud api production',
        'blog_index':       'blog articles posts python machine learning news',
        'blog_python_tips': 'python tips tricks tutorial beginner advanced python',
        'blog_ml_news':     'machine learning news research deep learning updates',
        'db_guide':         'database sql postgresql python orm sqlalchemy',
        'advanced_python':  'advanced python metaclasses decorators generators asyncio',
    }

def build_page_dates():
    now = datetime.now()
    return {
        'homepage':         now - timedelta(days=1),
        'python_tutorial':  now - timedelta(days=30),
        'data_science_intro':now - timedelta(days=15),
        'ml_algorithms':    now - timedelta(days=120),
        'ml_deep_learning': now - timedelta(days=7),
        'api_guide':        now - timedelta(days=60),
        'deployment_guide': now - timedelta(days=5),
        'blog_index':       now - timedelta(days=2),
        'blog_python_tips': now - timedelta(days=10),
        'blog_ml_news':     now - timedelta(days=3),
        'db_guide':         now - timedelta(days=200),
        'advanced_python':  now - timedelta(days=45),
    }


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def check(condition, label):
    symbol = 'PASS' if condition else 'FAIL'
    print(f"    [{symbol}] {label}")
    return condition

def print_results(results, top_n=5):
    print(f"    {'Rank':<5} {'Page':<22} {'Score':>8} {'PR':>8} {'Rel':>8} {'Rec':>8}")
    print(f"    {'-'*60}")
    for i, r in enumerate(results[:top_n], 1):
        print(f"    {i:<5} {r['page']:<22} {r['score']:>8.4f} {r['pr']:>8.4f} "
              f"{r['relevance']:>8.4f} {r['recency']:>8.4f}")


# ══════════════════════════════════════════════════════════════════
# MAIN TEST METHOD
# ══════════════════════════════════════════════════════════════════

def main():
    SEP = '=' * 65
    sep = '-' * 65

    print(SEP)
    print('  REQUIREMENT 3 — SearchEngine: MAIN TEST')
    print(SEP)

    G       = build_test_graph()
    content = build_page_content()
    dates   = build_page_dates()
    se      = SearchEngine(G, page_content=content, page_dates=dates)

    results_summary = []   # (test_name, passed)

    # ── TEST 1: Constructor and initialisation ────────────────────
    print(f"\nTEST 1 — Constructor and initialisation")
    ok = True
    ok &= check(hasattr(se, 'G'),            "se.G is set")
    ok &= check(hasattr(se, 'nodes'),        "se.nodes is set")
    ok &= check(hasattr(se, '_base_pr'),     "se._base_pr computed")
    ok &= check(hasattr(se, '_recency'),     "se._recency computed")
    ok &= check(se.N == G.number_of_nodes(), f"se.N matches graph ({se.N} nodes)")
    pr_sum = sum(se._base_pr.values())
    ok &= check(abs(pr_sum - 1.0) < 1e-6,   f"_base_pr sums to 1.0 (got {pr_sum:.8f})")
    rec_vals = list(se._recency.values())
    ok &= check(all(0.0 <= v <= 1.0 for v in rec_vals),
                "_recency values all in [0, 1]")
    results_summary.append(('Constructor & init', ok))

    # ── TEST 2: search() return format ───────────────────────────
    print(f"\nTEST 2 — search() return type and structure")
    res = se.search(top_k=10)
    ok = True
    ok &= check(isinstance(res, list),                  "returns a list")
    ok &= check(len(res) <= 10,                         f"respects top_k=10 (got {len(res)})")
    ok &= check(all(isinstance(r, dict) for r in res),  "each result is a dict")
    required_keys = {'page', 'score', 'pr', 'relevance', 'recency'}
    ok &= check(all(required_keys.issubset(r.keys()) for r in res),
                f"each result has keys: {required_keys}")
    ok &= check(all(isinstance(r['page'], str) for r in res),
                "page field is a string")
    ok &= check(all(isinstance(r['score'], float) for r in res),
                "score field is a float")
    results_summary.append(('Return type & structure', ok))

    # ── TEST 3: Pure PageRank (no keyword, no weights) ────────────
    print(f"\nTEST 3 — Pure PageRank mode (all weights = 0)")
    res = se.search(keyword='', relevance_weight=0.0, recency_weight=0.0,
                    topic_weight=0.0, top_k=10)
    ok = True
    # results must be sorted descending by score
    scores = [r['score'] for r in res]
    ok &= check(scores == sorted(scores, reverse=True),
                "results sorted descending by score")
    # top result must be highest-PR page
    top_pr_page = max(se._base_pr, key=se._base_pr.get)
    ok &= check(res[0]['page'] == top_pr_page,
                f"top result is highest-PR page: '{top_pr_page}'")
    # score == pr when all weights are 0 (w_pr = 1.0)
    ok &= check(all(abs(r['score'] - r['pr']) < 1e-9 for r in res),
                "score == pr when all other weights are 0")
    print(f"  Top 5 (pure PageRank):")
    print_results(res)
    results_summary.append(('Pure PageRank mode', ok))

    # ── TEST 4: Keyword relevance ─────────────────────────────────
    print(f"\nTEST 4 — Keyword relevance (keyword='python', relevance_weight=0.5)")
    res = se.search(keyword='python', relevance_weight=0.5, top_k=10)
    ok = True
    ok &= check(scores == sorted(scores, reverse=True),
                "results sorted descending")
    # pages with 'python' in content must have relevance > 0
    python_pages = [r for r in res if 'python' in r['page'] or
                    'python' in content.get(r['page'], '')]
    ok &= check(all(r['relevance'] > 0 for r in python_pages),
                "pages with 'python' content have relevance > 0")
    # a page with no 'python' in content must have relevance == 0
    # deployment_guide content: 'deployment docker kubernetes cloud api production' — no 'python'
    no_python = next((r for r in se.search(keyword='python', top_k=se.N)
                      if r['page'] == 'deployment_guide'), None)
    if no_python:
        ok &= check(no_python['relevance'] == 0.0,
                    "deployment_guide has 0 relevance for 'python' query")
    ok &= check(sum(r['relevance'] for r in se.search(keyword='python', top_k=se.N))
                <= 1.0 + 1e-9,
                "relevance scores sum to <= 1.0 across all pages")
    print(f"  Top 5 (keyword='python', rel_weight=0.5):")
    print_results(res)
    results_summary.append(('Keyword relevance', ok))

    # ── TEST 5: Recency weight ────────────────────────────────────
    print(f"\nTEST 5 — Recency weight (recency_weight=0.5)")
    res_rec  = se.search(recency_weight=0.5, top_k=10)
    res_base = se.search(recency_weight=0.0, top_k=10)
    ok = True
    # ranking must differ from pure-PR ranking when recency matters
    ok &= check([r['page'] for r in res_rec] != [r['page'] for r in res_base],
                "recency changes ranking vs pure PageRank")
    # homepage was updated 1 day ago — should rank higher with recency
    rec_ranks = {r['page']: i for i, r in enumerate(res_rec, 1)}
    base_ranks = {r['page']: i for i, r in enumerate(res_base, 1)}
    # most-recently-updated page should improve vs pure-PR
    freshest = max(dates, key=lambda p: dates[p])
    ok &= check(rec_ranks.get(freshest, 99) <= base_ranks.get(freshest, 99),
                f"freshest page '{freshest}' ranks at least as high with recency")
    # recency values must all be in [0, 1]
    ok &= check(all(0.0 <= r['recency'] <= 1.0 for r in res_rec),
                "all recency values in [0, 1]")
    print(f"  Top 5 (recency_weight=0.5):")
    print_results(res_rec)
    results_summary.append(('Recency weight', ok))

    # ── TEST 6: Topic-Sensitive PageRank ─────────────────────────
    print(f"\nTEST 6 — Topic-Sensitive PageRank (seeds=['ml_algorithms'])")
    res_topic = se.search(topic_weight=0.4, topic_seeds=['ml_algorithms'], top_k=10)
    res_plain = se.search(topic_weight=0.0, top_k=10)
    ok = True
    ok &= check(len(res_topic) > 0, "returns results with topic_weight > 0")
    # ML-related pages should rank higher with ML seeds
    ml_pages = {'ml_algorithms', 'ml_deep_learning', 'blog_ml_news'}
    topic_ml_rank = [r['page'] for r in res_topic[:5]]
    plain_ml_rank = [r['page'] for r in res_plain[:5]]
    ml_in_topic_top5 = len(ml_pages & set(topic_ml_rank))
    ml_in_plain_top5 = len(ml_pages & set(plain_ml_rank))
    ok &= check(ml_in_topic_top5 >= ml_in_plain_top5,
                f"ML pages in top-5 with topic={ml_in_topic_top5} vs plain={ml_in_plain_top5}")
    # ranking must change vs plain PageRank
    ok &= check([r['page'] for r in res_topic] != [r['page'] for r in res_plain],
                "topic-sensitive ranking differs from plain PageRank")
    print(f"  Top 5 (topic seed: ml_algorithms):")
    print_results(res_topic)
    results_summary.append(('Topic-sensitive PageRank', ok))

    # ── TEST 7: Combined weights ──────────────────────────────────
    print(f"\nTEST 7 — Combined weights (keyword + recency + topic)")
    res = se.search(keyword='machine learning', relevance_weight=0.3,
                    recency_weight=0.2, topic_weight=0.2,
                    topic_seeds=['ml_deep_learning'], top_k=10)
    ok = True
    ok &= check(len(res) > 0, "returns results with all weights combined")
    scores_combined = [r['score'] for r in res]
    ok &= check(scores_combined == sorted(scores_combined, reverse=True),
                "results sorted descending with combined weights")
    # w_pr = 1 - 0.3 - 0.2 - 0.2 = 0.3 (must be >= 0)
    w_pr = max(0.0, 1.0 - 0.3 - 0.2 - 0.2)
    ok &= check(w_pr >= 0.0, f"w_pr = {w_pr:.2f} >= 0 (weights don't exceed 1.0)")
    # all scores must be >= 0
    ok &= check(all(r['score'] >= 0.0 for r in res),
                "all combined scores >= 0")
    print(f"  Top 5 (keyword='machine learning', rel=0.3, rec=0.2, topic=0.2):")
    print_results(res)
    results_summary.append(('Combined weights', ok))

    # ── TEST 8: Edge cases ────────────────────────────────────────
    print(f"\nTEST 8 — Edge cases")
    ok = True

    # Empty keyword — should not crash
    try:
        res_empty = se.search(keyword='', top_k=5)
        ok &= check(len(res_empty) > 0, "empty keyword returns results without error")
    except Exception as e:
        ok &= check(False, f"empty keyword raised: {e}")

    # top_k=1 — returns exactly 1 result
    res_k1 = se.search(top_k=1)
    ok &= check(len(res_k1) == 1, "top_k=1 returns exactly 1 result")

    # Unknown topic seed — should not crash
    try:
        res_unk = se.search(topic_weight=0.3, topic_seeds=['nonexistent_page'], top_k=5)
        ok &= check(isinstance(res_unk, list), "unknown topic seed handled gracefully")
    except Exception as e:
        ok &= check(False, f"unknown topic seed raised: {e}")

    # Overweighted — weights sum > 1.0, w_pr clamps to 0
    res_ow = se.search(keyword='python', relevance_weight=0.6,
                       recency_weight=0.6, top_k=5)
    ok &= check(all(r['score'] >= 0.0 for r in res_ow),
                "overweighted query (>1.0 total) still returns non-negative scores")

    # Missing page_content — falls back to page name
    G2 = nx.DiGraph()
    G2.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'a')])
    se2 = SearchEngine(G2)
    res2 = se2.search(keyword='b', relevance_weight=0.5, top_k=3)
    ok &= check(len(res2) == 3, "works correctly with no page_content provided")
    ok &= check(res2[0]['page'] == 'b',
                "page name used as fallback content — 'b' ranks first for keyword 'b'")

    results_summary.append(('Edge cases', ok))

    # ── SUMMARY ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SUMMARY")
    print(SEP)
    passed = sum(1 for _, ok in results_summary if ok)
    for name, ok in results_summary:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print(f"\n  {passed}/{len(results_summary)} tests passed")
    print(SEP)


if __name__ == '__main__':
    main()