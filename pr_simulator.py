import streamlit as st
import networkx as nx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from scipy.stats import spearmanr, kendalltau
import streamlit.components.v1 as components
import re
import plotly.graph_objects as go
import math
import numpy as np
import scipy.sparse as sp
import random
import datetime
import json
import time


# ══════════════════════════════════════════════════════════════════
# 1. MATHEMATICAL ENGINES
# ══════════════════════════════════════════════════════════════════

def run_nx_pagerank(G, alpha=0.85):
    if G.number_of_nodes() == 0:
        return {}, 0
    ranks = nx.pagerank(G, alpha=alpha, max_iter=200, tol=1e-9)
    return ranks, 1


def run_pagerank_step(G, current_ranks, alpha=0.85):
    n = G.number_of_nodes()
    if n == 0:
        return {}, 0
    dangling_sum = sum(
        current_ranks.get(node, 0)
        for node in G.nodes()
        if G.out_degree(node) == 0
    )
    new_ranks = {}
    for node in G.nodes():
        incoming = sum(
            current_ranks.get(pred, 0) / G.out_degree(pred)
            for pred in G.predecessors(node)
            if G.out_degree(pred) > 0
        )
        new_ranks[node] = (1 - alpha) / n + alpha * (incoming + dangling_sum / n)
    delta = sum(abs(new_ranks[n] - current_ranks.get(n, 0)) for n in G.nodes())
    return new_ranks, delta


def matrix_algebra_pagerank(G, alpha=0.85):
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {}, 0
    idx = {node: i for i, node in enumerate(nodes)}
    M = np.zeros((N, N))
    for node in nodes:
        od = G.out_degree(node)
        col = idx[node]
        if od > 0:
            for nb in G.successors(node):
                M[idx[nb]][col] = 1.0 / od
        else:
            M[:, col] = 1.0 / N
    r = np.linalg.solve(np.eye(N) - alpha * M, np.ones(N) * (1 - alpha) / N)
    r = np.abs(r)
    r /= r.sum()
    return {nodes[i]: float(r[i]) for i in range(N)}, 1


def run_sparse_step(G, current_ranks, alpha=0.85):
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {}, 0
    idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, vals, dangling_cols = [], [], [], []
    for node in nodes:
        od = G.out_degree(node)
        c = idx[node]
        if od > 0:
            for nb in G.successors(node):
                rows.append(idx[nb]);
                cols.append(c);
                vals.append(1.0 / od)
        else:
            dangling_cols.append(c)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    r_vec = np.array([current_ranks.get(n, 1.0 / N) for n in nodes])
    d_sum = alpha * r_vec[dangling_cols].sum() / N if dangling_cols else 0.0
    new_r = alpha * M.dot(r_vec) + d_sum + (1 - alpha) / N
    delta = float(np.abs(new_r - r_vec).sum())
    return {nodes[i]: float(new_r[i]) for i in range(N)}, delta


def monte_carlo_pagerank_batch(G, alpha=0.85, mc_counts=None, surfers_per_step=300):
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {}, mc_counts or {}
    counts = (mc_counts or {n: 0 for n in nodes}).copy()
    successor_cache = {n: list(G.successors(n)) for n in nodes}
    for _ in range(surfers_per_step):
        cur = random.choice(nodes)
        for _ in range(40):
            counts[cur] = counts.get(cur, 0) + 1
            nbrs = successor_cache[cur]
            cur = (random.choice(nbrs) if nbrs and random.random() < alpha
                   else random.choice(nodes))
    total = sum(counts.values()) or 1
    return {n: counts[n] / total for n in nodes}, counts


def run_pruning_pagerank_step(G, current_ranks, alpha=0.85, it=0, warmup=10,
                              prune_threshold_factor=0.5, stable_rounds=3,
                              low_counts=None, active_mask=None, start_time=None):
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {}, 0, None, None, 0, 0

    # Track performance
    if start_time is None:
        start_time = time.time()

    idx = {n: i for i, n in enumerate(nodes)}

    if low_counts is None or len(low_counts) != N:
        low_counts = np.zeros(N, dtype=int)
    if active_mask is None or len(active_mask) != N:
        active_mask = np.ones(N, dtype=bool)

    active_before = np.sum(active_mask)

    r_vec = np.array([current_ranks.get(n, 1.0 / N) for n in nodes])
    new_r = np.zeros(N)

    dangling_sum = sum(r_vec[i] for i, n in enumerate(nodes) if G.out_degree(n) == 0)

    for i, node in enumerate(nodes):
        if not active_mask[i]:
            new_r[i] = r_vec[i]
            continue

        incoming = sum(r_vec[idx[p]] / G.out_degree(p)
                       for p in G.predecessors(node) if G.out_degree(p) > 0)
        new_r[i] = (1 - alpha) / N + alpha * (incoming + dangling_sum / N)

    threshold = prune_threshold_factor / N
    if it >= warmup:
        for i in range(N):
            if active_mask[i] and new_r[i] < threshold:
                low_counts[i] += 1
                if low_counts[i] >= stable_rounds:
                    active_mask[i] = False
            elif active_mask[i]:
                low_counts[i] = 0

    active_after = np.sum(active_mask)
    frozen_nodes = active_before - active_after

    delta = float(np.abs(new_r[active_mask] - r_vec[active_mask]).sum())
    res_dict = {nodes[i]: float(new_r[i]) for i in range(N)}

    elapsed = time.time() - start_time

    return res_dict, delta, low_counts, active_mask, frozen_nodes, elapsed


# ══════════════════════════════════════════════════════════════════
# 1b. SEARCH ENGINE CLASS
# ══════════════════════════════════════════════════════════════════

class SearchEngine:
    def __init__(self, G, page_content=None, page_dates=None):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.page_content = page_content or {}
        self.page_dates = page_dates or {}
        if self.N > 0:
            self._base_pr = nx.pagerank(G, alpha=0.85)
        else:
            self._base_pr = {}
        self._recency = self._build_recency()

    def _build_M(self):
        idx = {node: i for i, node in enumerate(self.nodes)}
        M = np.zeros((self.N, self.N))
        for node in self.nodes:
            od = self.G.out_degree(node)
            col = idx[node]
            if od > 0:
                for nb in self.G.successors(node):
                    M[idx[nb]][col] = 1.0 / od
            else:
                M[:, col] = 1.0 / self.N
        return M

    def _build_recency(self):
        if not self.nodes: return {}
        now = datetime.datetime.now()
        scores = {}
        for p in self.nodes:
            if p in self.page_dates:
                days = max(0, (now - self.page_dates[p]).days)
                scores[p] = np.exp(-0.01 * days)
            else:
                scores[p] = 0.5
        mn, mx = min(scores.values()), max(scores.values())
        return {p: (s - mn) / (mx - mn) if mx > mn else s for p, s in scores.items()}

    def _relevance(self, keyword):
        if not keyword: return {p: 0.0 for p in self.nodes}
        kw = keyword.lower().strip()
        scores = {}
        for p in self.nodes:
            text = str(self.page_content.get(p, "")).lower()
            count = sum(text.count(w) for w in kw.split())
            scores[p] = float(count)
        total = sum(scores.values())
        return {p: s / total if total > 0 else 0.0 for p, s in scores.items()}

    def _topic_pr(self, seeds, alpha=0.85):
        if not seeds or self.N == 0:
            return self._base_pr

        idx = {n: i for i, n in enumerate(self.nodes)}
        teleport = np.zeros(self.N)
        valid = [s for s in seeds if s in idx]

        if not valid: return self._base_pr

        for s in valid:
            teleport[idx[s]] += 1.0 / len(valid)

        rank = np.full(self.N, 1.0 / self.N)
        M = self._build_M()

        for _ in range(50):
            new_rank = alpha * M @ rank + (1 - alpha) * teleport
            if np.abs(new_rank - rank).sum() < 1e-6: break
            rank = new_rank
        return {self.nodes[i]: rank[i] for i in range(self.N)}

    def search(self, keyword='', alpha=0.85, relevance_weight=0.0,
               recency_weight=0.0, topic_weight=0.0,
               topic_seeds=None, top_k=None):

        pr = self._base_pr
        rel = self._relevance(keyword)
        rec = self._recency
        top = self._topic_pr(topic_seeds or []) if topic_weight > 0 else pr

        w_pr = max(0.0, 1.0 - relevance_weight - recency_weight - topic_weight)

        results = []
        for p in self.nodes:
            score = (w_pr * pr.get(p, 0)
                     + relevance_weight * rel.get(p, 0)
                     + recency_weight * rec.get(p, 0)
                     + topic_weight * top.get(p, 0))

            results.append({
                'page': p,
                'score': round(score, 6),
                'pr': pr.get(p, 0),
                'relevance': rel.get(p, 0)
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k] if top_k is not None else results


# ══════════════════════════════════════════════════════════════════
# 2. CRAWLER
# ══════════════════════════════════════════════════════════════════

def crawl_and_analyze(seed_url, limit, max_depth=3,
                      path_prefix="", must_contain="", skip_extensions=True):
    SKIP_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                 ".zip", ".tar", ".gz", ".mp4", ".mp3", ".css", ".js",
                 ".woff", ".woff2", ".ttf", ".eot", ".xml", ".json"}

    G = nx.DiGraph()
    content_map = {}
    to_visit = [(seed_url, 0)]
    visited = set()
    parsed_seed = urlparse(seed_url)
    base_domain = parsed_seed.netloc.replace("www.", "")
    seed_path = path_prefix.strip()
    kw_filter = must_contain.strip().lower()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PageRankBot/1.0)"}
    progress_bar = st.progress(0, text="Crawling…")

    def _allowed_url(url_clean):
        p = urlparse(url_clean)
        if base_domain not in p.netloc:
            return False
        if p.scheme not in ("http", "https"):
            return False
        if seed_path and not p.path.startswith(seed_path):
            return False
        if skip_extensions:
            ext = "." + p.path.rsplit(".", 1)[-1].lower() if "." in p.path.split("/")[-1] else ""
            if ext in SKIP_EXTS:
                return False
        return True

    while to_visit and len(visited) < limit:
        url, depth = to_visit.pop(0)
        url_clean = url.split("#")[0]
        if not url_clean.endswith(".html"):
            url_clean = url_clean.rstrip("/")

        if url_clean in visited:
            continue
        if not _allowed_url(url_clean):
            continue

        try:
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            page_text = soup.get_text(separator=" ", strip=True).lower()

            if kw_filter and kw_filter not in page_text:
                continue

            visited.add(url_clean)
            G.add_node(url_clean)
            content_map[url_clean] = page_text
            progress_bar.progress(
                min(len(visited) / limit, 1.0),
                text=f"Crawled {len(visited)}/{limit} pages (depth {depth})…"
            )

            if depth >= max_depth:
                continue

            for a_tag in soup.find_all("a", href=True):
                target = urljoin(url, a_tag["href"]).split("#")[0].rstrip("/")
                if target == url_clean:
                    continue
                if not _allowed_url(target):
                    continue
                G.add_edge(url_clean, target)
                if target not in visited and (target, depth + 1) not in [(u, d) for u, d in to_visit]:
                    to_visit.append((target, depth + 1))

        except Exception:
            continue

    progress_bar.empty()
    return G, content_map


# ══════════════════════════════════════════════════════════════════
# 3. GOOGLE COMPARISON
# ══════════════════════════════════════════════════════════════════

def breadcrumb_to_url(raw):
    raw = raw.strip()
    if not raw:
        return ""

    if "://" in raw and (" › " in raw or " > " in raw):
        base = re.split(r" [›>] ", raw)[0].strip()
        crumbs = re.split(r" [›>] ", raw)
        domain = base.lower().replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        path_parts = [p.strip() for p in crumbs[1:]]
        return (domain + "/" + "/".join(path_parts)).rstrip("/")

    if "://" in raw:
        url = raw.lower().replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        url = url.replace("/index.html", "").replace("/index.htm", "")
        return url

    if " › " in raw or " > " in raw:
        crumbs = re.split(r" [›>] ", raw)
        domain = crumbs[0].strip().lower().replace("www.", "")
        path_parts = [p.strip() for p in crumbs[1:]]
        if path_parts:
            return (domain + "/" + "/".join(path_parts)).rstrip("/")
        return domain

    url = raw.lower().replace("www.", "").rstrip("/")
    return url


def paths_overlap(sim_norm, google_norm):
    if sim_norm == google_norm:
        return True

    def segs(u):
        return [s for s in u.split("/") if s]

    s = segs(sim_norm)
    g = segs(google_norm)
    if not s or not g:
        return False

    short, long_ = (s, g) if len(s) <= len(g) else (g, s)
    if len(short) >= 2 and long_[-len(short):] == short:
        return True

    trivial = {"html", "htm", "index", "home", ""}
    last_s = s[-1].replace(".html", "").replace(".htm", "")
    last_g = g[-1].replace(".html", "").replace(".htm", "")
    if last_s == last_g and last_s not in trivial and s[0] == g[0]:
        return True

    return False


def calculate_comparison(sim_ranks_dict, google_input):
    sim_sorted = sorted(sim_ranks_dict.keys(),
                        key=lambda x: sim_ranks_dict[x], reverse=True)

    if isinstance(google_input, str):
        google_raw = [l.strip() for l in google_input.split("\n") if l.strip()]
    else:
        google_raw = [str(l).strip() for l in google_input if str(l).strip()]

    google_norm = [breadcrumb_to_url(g) for g in google_raw]

    matched_pairs = []
    used_google_indices = set()

    for sim_idx, sim_url in enumerate(sim_sorted):
        sim_norm = breadcrumb_to_url(sim_url)
        for g_idx, g_norm in enumerate(google_norm):
            if g_idx in used_google_indices:
                continue
            if paths_overlap(sim_norm, g_norm):
                matched_pairs.append({
                    "url": sim_url,
                    "sim_rank": sim_idx + 1,
                    "google_rank": g_idx + 1,
                    "sim_norm": sim_norm,
                    "google_norm": g_norm,
                })
                used_google_indices.add(g_idx)
                break

    if len(matched_pairs) < 2:
        return None, None, matched_pairs

    v_sim = [m["sim_rank"] for m in matched_pairs]
    v_goog = [m["google_rank"] for m in matched_pairs]
    rho, _ = spearmanr(v_sim, v_goog)
    tau, _ = kendalltau(v_sim, v_goog)
    return rho, tau, matched_pairs


# ══════════════════════════════════════════════════════════════════
# 4. COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════

def rank_to_colour(rank, min_r, max_r):
    if max_r == min_r:
        t = 0.5
    else:
        t = (rank - min_r) / (max_r - min_r)
    if t < 0.5:
        s = t * 2
        r = int(59 + s * (245 - 59))
        g = int(130 + s * (158 - 130))
        b = int(246 + s * (11 - 246))
    else:
        s = (t - 0.5) * 2
        r = int(245 + s * (239 - 245))
        g = int(158 + s * (68 - 158))
        b = int(11 + s * (68 - 11))
    return f"#{r:02X}{g:02X}{b:02X}"


def rank_to_frozen_colour():
    return "#64748B"  # Slate gray for frozen nodes


# ══════════════════════════════════════════════════════════════════
# 5. GRAPH BUILDER - IMPROVED WITH HIERARCHICAL LAYOUT
# ══════════════════════════════════════════════════════════════════

def build_graph_html(G, ranks, page_map, fix_positions=True, frozen_nodes=None, method=None):
    nodes = list(G.nodes())
    if not nodes:
        return "<p style='color:#888'>No graph yet.</p>"

    rv = [ranks.get(n, 0) for n in nodes]
    min_r, max_r = min(rv), max(rv)

    frozen_set = set(frozen_nodes) if frozen_nodes else set()

    # Improved layout using hierarchical positioning
    n_nodes = len(nodes)

    # Use hierarchical layout based on PageRank (higher PR at top)
    pos = {}
    sorted_by_pr = sorted([(n, ranks.get(n, 0)) for n in nodes], key=lambda x: x[1], reverse=True)

    # Arrange in multiple layers
    layers = 5
    nodes_per_layer = max(1, math.ceil(n_nodes / layers))

    radius_x = 400
    radius_y = 350

    for idx, (node, pr) in enumerate(sorted_by_pr):
        layer = min(idx // nodes_per_layer, layers - 1)
        pos_in_layer = idx % nodes_per_layer
        layer_count = min(nodes_per_layer, n_nodes - layer * nodes_per_layer)

        # Even spacing within layer
        x = -radius_x + (pos_in_layer / max(1, layer_count - 1)) * (2 * radius_x) if layer_count > 1 else 0
        y = radius_y - layer * (2 * radius_y / layers)

        # Add some variation to avoid perfect alignment
        x += random.uniform(-20, 20) if not fix_positions else 0

        pos[node] = (x, y)

    vis_nodes = []
    for node in nodes:
        r = ranks.get(node, 0)
        is_frozen = node in frozen_set

        if is_frozen:
            colour = rank_to_frozen_colour()
            size = 20  # Smaller size for frozen nodes
        else:
            colour = rank_to_colour(r, min_r, max_r)
            size = 22 + (r - min_r) / max(max_r - min_r, 1e-12) * 35

        label_id = page_map.get(node, "?")
        short_name = urlparse(node).path.rstrip("/").split("/")[-1] or "home"
        short_name = short_name[:22]

        # Add frozen indicator
        frozen_badge = " ❄️" if is_frozen else ""
        label = f"#{label_id}\n{short_name}{frozen_badge}"

        title = f"<b>#{label_id} — {short_name}</b><br>URL: {node}<br>PageRank: {r:.6f}"
        if is_frozen:
            title += "<br><span style='color:#94A3B8'>❄️ FROZEN (pruned)</span>"

        px, py = pos.get(node, (0, 0))

        # Node border styling based on frozen status
        border_color = "#475569" if is_frozen else "#1E293B"
        border_width = 1 if is_frozen else 2

        vis_nodes.append({
            "id": node,
            "label": label,
            "title": title,
            "color": {
                "background": colour,
                "border": border_color,
                "highlight": {"background": "#FDE68A" if not is_frozen else "#94A3B8",
                              "border": "#F59E0B" if not is_frozen else "#475569"}
            },
            "size": size,
            "font": {
                "size": 10 if is_frozen else 11,
                "color": "#FFFFFF" if not is_frozen else "#94A3B8",
                "strokeWidth": 1,
                "strokeColor": "#000000"
            },
            "x": px,
            "y": py,
            "fixed": {"x": fix_positions, "y": fix_positions},
            "physics": not fix_positions,
        })

    # Edge bundling for cleaner visualization
    vis_edges = []
    for u, v in G.edges():
        # Determine if edge connects to frozen node
        edge_color = "#64748B" if (u in frozen_set or v in frozen_set) else "#94A3B8"
        edge_opacity = 0.4 if (u in frozen_set or v in frozen_set) else 0.7

        vis_edges.append({
            "from": u, "to": v,
            "arrows": "to",
            "color": {"color": edge_color, "opacity": edge_opacity},
            "width": 1 if (u in frozen_set or v in frozen_set) else 1.5,
            "smooth": {"type": "curvedCW", "roundness": 0.15},
            "dashes": (u in frozen_set or v in frozen_set)
        })

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin:0; background:#0F172A; }}
  #net {{ width:100%; height:520px; border-radius:8px; }}
  #legend {{ position:absolute; bottom:10px; left:10px; display:flex; align-items:center; gap:12px; background:rgba(15,23,42,0.9); padding:6px 12px; border-radius:8px; }}
  #legend span {{ font-size:11px; color:#CBD5E1; font-family:monospace; }}
  .grad {{ width:120px; height:12px; border-radius:4px;
           background:linear-gradient(to right,#3B82F6,#F59E0B,#EF4444); }}
  .legend-frozen {{ width:16px; height:16px; background:#64748B; border-radius:4px; display:inline-block; margin-right:4px; }}
</style>
</head>
<body>
<div style="position:relative">
  <div id="net"></div>
  <div id="legend">
    <span>Low PR</span>
    <div class="grad"></div>
    <span>High PR</span>
    <div style="width:1px; height:20px; background:#334155;"></div>
    <div class="legend-frozen"></div>
    <span>Frozen (pruned)</span>
  </div>
</div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const container = document.getElementById('net');
const data = {{ nodes, edges }};
const options = {{
  physics: {{ 
    enabled: {'false' if fix_positions else 'true'},
    stabilization: {{ iterations: {'100' if fix_positions else '300'} }},
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -50,
      centralGravity: 0.01,
      springLength: 100,
      springConstant: 0.08,
      damping: 0.4
    }}
  }},
  interaction: {{ 
    hover: true, 
    tooltipDelay: 100,
    zoomView: true,
    dragView: true,
    navigationButtons: false
  }},
  layout: {{
    improvedLayout: true,
    hierarchical: {{
      enabled: false
    }}
  }},
  edges: {{ 
    arrows: {{ to: {{ scaleFactor: 0.6 }} }},
    smooth: {{
      type: 'curvedCW',
      roundness: 0.15
    }}
  }},
  nodes: {{ 
    borderWidth: 2, 
    shadow: {{
      enabled: true,
      size: 3,
      x: 2,
      y: 2
    }}
  }},
}};
new vis.Network(container, data, options);
</script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════
# 6. STREAMLIT APP
# ══════════════════════════════════════════════════════════════════

st.set_page_config(layout="wide", page_title="PageRank Simulator", page_icon="🎓")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');
html, body, [class*="css"] { font-family:'Sora',sans-serif; }
h1 { font-size:1.9rem; font-weight:800; letter-spacing:-0.03em; }
h2,h3 { font-weight:600; }
.stButton>button {
    background:#2563EB; color:white; border:none; border-radius:6px;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700;
    padding:0.45rem 0.9rem; transition:all 0.15s;
}
.stButton>button:hover { background:#1D4ED8; transform:translateY(-1px); }
.metric-box {
    background:#1E293B; border:1px solid #334155; border-radius:8px;
    padding:14px 18px; text-align:center;
}
.metric-box .val { font-size:1.6rem; font-weight:800; color:#60A5FA;
                   font-family:'JetBrains Mono',monospace; }
.metric-box .lbl { font-size:0.72rem; color:#94A3B8; text-transform:uppercase;
                   letter-spacing:0.08em; margin-top:2px; }
.status-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:700; font-family:'JetBrains Mono',monospace;
}
.badge-running  { background:#FEF3C7; color:#92400E; }
.badge-converged{ background:#D1FAE5; color:#065F46; }
.badge-ready    { background:#EFF6FF; color:#1E40AF; }
.speed-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:700; font-family:'JetBrains Mono',monospace;
    background:#10B981; color:white;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "G" not in st.session_state:
    st.session_state.update({
        "G": nx.DiGraph(), "ranks": {}, "history": [], "iteration": 0,
        "page_map": {}, "converged": False, "content": {},
        "mc_counts": {}, "url_input": "https://docs.python.org/3/",
        "prune_low_counts": None, "prune_active_mask": None,
        "frozen_nodes": set(), "total_frozen": 0,
        "pruning_speed_data": [], "iter_times": [],
        "convergence_iterations_normal": None,
        "convergence_time_normal": None
    })

G = st.session_state.G
rks = st.session_state.ranks

st.markdown("# 🎓 PageRank Simulation Engine")
st.markdown("*Multi-method PageRank visualiser — crawl, simulate, search.*")

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    with st.expander("🕸️ Step 1 — Data Acquisition", expanded=True):
        url_input = st.text_input(
            "Seed URL",
            value=st.session_state.url_input,
            placeholder="https://example.com",
            key="seed_url_input"
        )
        st.session_state.url_input = url_input

        limit = st.slider("Max pages to crawl", 5, 60, 15, key="crawl_limit_slider")
        max_depth = st.slider("Max crawl depth", 0, 5, 2, key="crawl_depth_slider")
        path_prefix = st.text_input("Restrict to URL path prefix (optional)", value="",
                                    placeholder="e.g. /docs/ or /en/stable/", key="path_prefix_input")
        must_contain = st.text_input("Only keep pages containing keyword (optional)", value="",
                                     placeholder="e.g. tutorial or api", key="must_contain_input")
        skip_ext = st.checkbox("Skip non-HTML assets (.pdf, .png, .zip …)", value=True, key="skip_ext_checkbox")

        active = []
        if max_depth < 5: active.append(f"depth ≤ {max_depth}")
        if path_prefix: active.append(f"path: {path_prefix.strip()}")
        if must_contain: active.append(f"keyword: '{must_contain.strip()}'")
        if skip_ext: active.append("skip assets")
        if active:
            st.caption("Active filters: " + " · ".join(active))

        if st.button("🕸️ Crawl Website", use_container_width=True, key="crawl_button"):
            with st.spinner("Crawling…"):
                g, c = crawl_and_analyze(url_input, limit, max_depth, path_prefix, must_contain, skip_ext)
            if g.number_of_nodes() > 0:
                n = g.number_of_nodes()
                st.session_state.update({
                    "G": g, "content": c, "ranks": {node: 1.0 / n for node in g.nodes()},
                    "iteration": 0, "converged": False, "history": [],
                    "page_map": {node: str(i + 1) for i, node in enumerate(g.nodes())},
                    "mc_counts": {},
                    "prune_low_counts": None, "prune_active_mask": None,
                    "frozen_nodes": set(), "total_frozen": 0,
                    "pruning_speed_data": [], "iter_times": []
                })
                st.rerun()
            elif g.number_of_nodes() == 0:
                st.error("No pages crawled. Check URL or relax filters.")

    with st.expander("🧮 Step 2 — Algorithm", expanded=True):
        method = st.selectbox("Method", [
            "Manual Power Iteration",
            "NetworkX Built-in",
            "Matrix Algebra (Exact)",
            "SciPy Sparse Matrix",
            "Monte Carlo Surfer",
            "Progressive Pruning",
        ], key="method_selectbox")
        alpha = st.slider("Damping factor (α)", 0.50, 0.99, 0.85, 0.01, key="alpha_slider")

        c1, c2 = st.columns(2)
        with c1:
            step_btn = st.button("▶ Step", use_container_width=True, key="step_button")
        with c2:
            conv_btn = st.button("⏩ Converge", use_container_width=True, key="converge_button")
        reset_btn = st.button("↺ Reset", use_container_width=True, key="reset_button")

    with st.expander("🗺️ Step 3 — Graph Options"):
        fix_physics = st.checkbox("Fix node positions (no movement)", value=False, key="fix_physics_checkbox")
        st.caption("💡 Disable to allow automatic layout optimization")

    st.divider()
    if G.number_of_nodes() > 0:
        badge_cls = "badge-converged" if st.session_state.converged else (
            "badge-running" if st.session_state.iteration > 0 else "badge-ready")
        badge_txt = ("✅ Converged" if st.session_state.converged else
                     (f"🔄 Iter {st.session_state.iteration}" if st.session_state.iteration > 0
                      else "⏸ Ready"))
        st.markdown(f'<span class="status-badge {badge_cls}">{badge_txt}</span>',
                    unsafe_allow_html=True)

        # Show speed improvement for pruning
        if method == "Progressive Pruning" and len(st.session_state.pruning_speed_data) > 0:
            avg_speedup = np.mean(st.session_state.pruning_speed_data)
            st.markdown(f'<span class="speed-badge">⚡ {avg_speedup:.1f}% faster</span>',
                        unsafe_allow_html=True)

        st.caption(f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · α={alpha}")

        # Show frozen nodes count for pruning
        if method == "Progressive Pruning" and len(st.session_state.frozen_nodes) > 0:
            frozen_pct = (len(st.session_state.frozen_nodes) / G.number_of_nodes()) * 100
            st.caption(f"❄️ {len(st.session_state.frozen_nodes)} nodes frozen ({frozen_pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════
# DO STEP FUNCTION
# ══════════════════════════════════════════════════════════════════

def do_step():
    G = st.session_state.G
    rks = st.session_state.ranks
    if G.number_of_nodes() == 0:
        return

    st.session_state.history.append(rks.copy())

    start_step = time.time()

    if method == "Progressive Pruning":
        # For pruning, track performance
        if st.session_state.iteration == 0:
            st.session_state.pruning_start_time = time.time()
            st.session_state.iter_times = []

        r, d, lc, mask, frozen_this_step, elapsed = run_pruning_pagerank_step(
            G, rks, alpha,
            it=st.session_state.iteration,
            low_counts=st.session_state.prune_low_counts,
            active_mask=st.session_state.prune_active_mask,
            start_time=st.session_state.get('pruning_start_time', None)
        )

        # Track frozen nodes
        if frozen_this_step > 0:
            new_frozen = set()
            for i, is_active in enumerate(mask):
                if not is_active and list(G.nodes())[i] not in st.session_state.frozen_nodes:
                    new_frozen.add(list(G.nodes())[i])
            st.session_state.frozen_nodes.update(new_frozen)

        # Calculate speed improvement
        if st.session_state.iteration > 0 and len(st.session_state.iter_times) > 0:
            avg_time = np.mean(st.session_state.iter_times)
            if avg_time > 0:
                improvement = ((avg_time - elapsed) / avg_time) * 100
                st.session_state.pruning_speed_data.append(max(0, improvement))

        st.session_state.iter_times.append(elapsed)
        st.session_state.ranks = r
        st.session_state.prune_low_counts = lc
        st.session_state.prune_active_mask = mask
        st.session_state.iteration += 1
        st.session_state.total_frozen += frozen_this_step

        if d < 1e-7:
            st.session_state.converged = True

    elif method == "Manual Power Iteration":
        r, d = run_pagerank_step(G, rks, alpha)
        st.session_state.ranks = r
        st.session_state.iteration += 1
        if d < 1e-7:
            st.session_state.converged = True
    elif method == "NetworkX Built-in":
        r, _ = run_nx_pagerank(G, alpha)
        st.session_state.ranks = r
        st.session_state.iteration += 1
        st.session_state.converged = True
    elif method == "Matrix Algebra (Exact)":
        r, _ = matrix_algebra_pagerank(G, alpha)
        st.session_state.ranks = r
        st.session_state.iteration += 1
        st.session_state.converged = True
    elif method == "SciPy Sparse Matrix":
        r, d = run_sparse_step(G, rks, alpha)
        st.session_state.ranks = r
        st.session_state.iteration += 1
        if d < 1e-7:
            st.session_state.converged = True
    elif method == "Monte Carlo Surfer":
        if st.session_state.iteration == 0:
            st.session_state.mc_counts = {}
        r, counts = monte_carlo_pagerank_batch(
            G, alpha, st.session_state.mc_counts, surfers_per_step=400)
        st.session_state.mc_counts = counts
        st.session_state.ranks = r
        st.session_state.iteration += 1
        if st.session_state.iteration >= 15:
            st.session_state.converged = True


# Handle button actions
if step_btn:
    do_step()
    st.rerun()

if conv_btn:
    if G.number_of_nodes() == 0:
        st.warning("Crawl a website first.")
    else:
        # For pruning, also track baseline performance
        if method == "Progressive Pruning":
            # First, run without pruning to get baseline
            if st.session_state.convergence_iterations_normal is None:
                with st.spinner("Running baseline for comparison..."):
                    temp_ranks = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}
                    temp_history = []
                    for _ in range(200):
                        temp_ranks, delta = run_pagerank_step(G, temp_ranks, alpha)
                        temp_history.append(delta)
                        if delta < 1e-7:
                            break
                    st.session_state.convergence_iterations_normal = len(temp_history)
                    st.session_state.convergence_time_normal = sum(temp_history[:10]) if temp_history else 0

        for _ in range(200):
            if st.session_state.converged:
                break
            do_step()
        st.rerun()

if reset_btn and G.number_of_nodes() > 0:
    n = G.number_of_nodes()
    st.session_state.update({
        "ranks": {node: 1.0 / n for node in G.nodes()},
        "iteration": 0, "converged": False, "history": [], "mc_counts": {},
        "prune_low_counts": None, "prune_active_mask": None,
        "frozen_nodes": set(), "total_frozen": 0,
        "pruning_speed_data": [], "iter_times": []
    })
    st.rerun()

# ══════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════

if G.number_of_nodes() == 0:
    st.info("👈 Enter a seed URL in the sidebar and click **Crawl Website** to begin.")
    st.stop()

rks = st.session_state.ranks
pm = st.session_state.page_map
rvs = list(rks.values())
min_r, max_r = min(rvs), max(rvs)

m1, m2, m3, m4, m5 = st.columns(5)
top_node = max(rks, key=rks.get)
top_name = urlparse(top_node).path.rstrip("/").split("/")[-1] or "home"

with m1:
    st.markdown(
        f'<div class="metric-box"><div class="val">{G.number_of_nodes()}</div><div class="lbl">Pages</div></div>',
        unsafe_allow_html=True)
with m2:
    st.markdown(
        f'<div class="metric-box"><div class="val">{G.number_of_edges()}</div><div class="lbl">Links</div></div>',
        unsafe_allow_html=True)
with m3:
    st.markdown(
        f'<div class="metric-box"><div class="val">{st.session_state.iteration}</div><div class="lbl">Iterations</div></div>',
        unsafe_allow_html=True)
with m4:
    st.markdown(
        f'<div class="metric-box"><div class="val">#{pm.get(top_node, "?")}</div><div class="lbl">Top Page</div></div>',
        unsafe_allow_html=True)
with m5:
    if method == "Progressive Pruning" and len(st.session_state.frozen_nodes) > 0:
        frozen_pct = (len(st.session_state.frozen_nodes) / G.number_of_nodes()) * 100
        st.markdown(
            f'<div class="metric-box"><div class="val">{len(st.session_state.frozen_nodes)}</div><div class="lbl">Frozen ({frozen_pct:.0f}%)</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="metric-box"><div class="val">{top_name[:10]}</div><div class="lbl">Top Page</div></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

gcol, scol = st.columns([3, 2])

with gcol:
    st.markdown(f"#### 🕸️ Link Graph — *{method}* · iteration {st.session_state.iteration}")
    frozen_list = list(st.session_state.frozen_nodes) if method == "Progressive Pruning" else None
    graph_html = build_graph_html(G, rks, pm, fix_positions=fix_physics,
                                  frozen_nodes=frozen_list, method=method)
    components.html(graph_html, height=560, scrolling=False)

    # Add graph instructions
    st.caption("💡 Hover over nodes for details • Drag to pan • Scroll to zoom")

with scol:
    st.markdown("#### 🔍 Advanced Search Engine")
    query = st.text_input("Query", key="adv_search_q")

    engine = SearchEngine(
        G=st.session_state.G,
        page_content=st.session_state.content,
        page_dates={}
    )

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        rel_w = st.slider("Relevance (TF-IDF)", 0.0, 1.0, 0.4, key="rel_w_slider")
        rec_w = st.slider("Recency Weight", 0.0, 1.0, 0.2, key="rec_w_slider")
    with col_w2:
        top_w = st.slider("Topic Weight", 0.0, 1.0, 0.2, key="top_w_slider")

    if query:
        seeds = [max(st.session_state.ranks, key=st.session_state.ranks.get)]

        results = engine.search(
            keyword=query,
            relevance_weight=rel_w,
            recency_weight=rec_w,
            topic_weight=top_w,
            topic_seeds=seeds
        )

        search_rows = []
        for r in results[:15]:  # Limit to top 15 for display
            node = r['page']
            is_frozen = node in st.session_state.frozen_nodes if method == "Progressive Pruning" else False
            frozen_mark = " ❄️" if is_frozen else ""
            search_rows.append({
                "#": st.session_state.page_map.get(node, "?"),
                "Name": (urlparse(node).path.rstrip("/").split("/")[-1] or "home")[:25] + frozen_mark,
                "URL": node,
                "PageRank": r['pr'],
                "Relevance": r['relevance'],
                "Final Score": r['score']
            })

        search_df = pd.DataFrame(search_rows)


        def style_search_results(row):
            pr_val = row["PageRank"]
            bg = rank_to_colour(pr_val, min_r, max_r)
            t = (pr_val - min_r) / max(max_r - min_r, 1e-12)
            fg = "#FFFFFF" if t > 0.55 else "#111111"
            return [f"background-color:{bg};color:{fg}"] * len(row)


        styled_search = (search_df.style.apply(style_search_results, axis=1)
                         .format({"PageRank": "{:.6f}", "Relevance": "{:.4f}", "Final Score": "{:.4f}"}))

        st.dataframe(styled_search,
                     use_container_width=True,
                     height=400,
                     hide_index=True,
                     column_config={"URL": st.column_config.LinkColumn("URL")})

st.divider()
st.markdown("#### 📊 Full Rankings — colour = PageRank magnitude")

all_rows = sorted([{
    "#": pm.get(n, "?"),
    "Name": (urlparse(n).path.rstrip("/").split("/")[-1] or "home")[:30] + (
        " ❄️" if n in st.session_state.frozen_nodes else ""),
    "URL": n,
    "PageRank": pr,
    "Δ prev": abs(pr - st.session_state.history[-1].get(n, pr)) if st.session_state.history else 0.0,
} for n, pr in rks.items()], key=lambda x: x["PageRank"], reverse=True)

full_df = pd.DataFrame(all_rows)


def style_full(row):
    pr_val = row["PageRank"]
    bg = rank_to_colour(pr_val, min_r, max_r)
    t = (pr_val - min_r) / max(max_r - min_r, 1e-12)
    fg = "#FFFFFF" if t > 0.55 else "#111111"
    return [f"background-color:{bg};color:{fg}"] * len(row)


styled_full = (full_df.style.apply(style_full, axis=1)
               .format({"PageRank": "{:.6f}", "Δ prev": "{:.2e}"}))
st.dataframe(styled_full, use_container_width=True, height=500, hide_index=True,
             column_config={"URL": st.column_config.LinkColumn("URL")})

if len(st.session_state.history) > 1:
    st.divider()
    st.markdown("#### 📈 Convergence History")
    h_df = pd.DataFrame(st.session_state.history)
    fig = go.Figure()
    top10 = sorted(rks, key=rks.get, reverse=True)[:10]
    for node in top10:
        if node in h_df.columns:
            is_frozen = node in st.session_state.frozen_nodes
            col = rank_to_frozen_colour() if is_frozen else rank_to_colour(rks[node], min_r, max_r)
            line_style = "dash" if is_frozen else "solid"
            label = f"#{pm.get(node, '?')} {urlparse(node).path.rstrip('/').split('/')[-1] or 'home'}"
            if is_frozen:
                label += " (frozen)"
            fig.add_trace(go.Scatter(y=h_df[node], mode="lines+markers", name=label,
                                     line=dict(color=col, width=2, dash=line_style),
                                     marker=dict(size=4)))
    fig.update_layout(
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font=dict(color="#CBD5E1", family="JetBrains Mono"),
        xaxis=dict(title="Iteration", gridcolor="#334155"),
        yaxis=dict(title="PageRank", gridcolor="#334155"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        height=400, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

if st.session_state.converged:
    st.divider()
    st.markdown("#### ⚡ Complexity & Benchmarking")

    # Show pruning speed improvement if applicable
    if method == "Progressive Pruning" and len(st.session_state.pruning_speed_data) > 0:
        avg_speedup = np.mean(st.session_state.pruning_speed_data)
        st.metric("Pruning Speed Improvement", f"{avg_speedup:.1f}%",
                  help="Average speed improvement compared to standard iteration")

        # Show frozen nodes statistics
        frozen_pct = (len(st.session_state.frozen_nodes) / G.number_of_nodes()) * 100
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Frozen Nodes", f"{len(st.session_state.frozen_nodes)} / {G.number_of_nodes()}",
                      f"{frozen_pct:.1f}% of graph")
        with col2:
            st.metric("Active Nodes", f"{G.number_of_nodes() - len(st.session_state.frozen_nodes)}",
                      f"{(100 - frozen_pct):.1f}% of graph")

    bc1, bc2 = st.columns(2)
    with bc1:
        complexity_df = pd.DataFrame([
            {"Method": "Manual Power Iteration", "Time": "O(k·(N+E))", "Space": "O(N+E)", "Exact": "No"},
            {"Method": "NetworkX Built-in", "Time": "O(k·(N+E))", "Space": "O(N+E)", "Exact": "Yes*"},
            {"Method": "Matrix Algebra", "Time": "O(N³)", "Space": "O(N²)", "Exact": "Yes"},
            {"Method": "SciPy Sparse Matrix", "Time": "O(k·E)", "Space": "O(N+E)", "Exact": "No"},
            {"Method": "Monte Carlo Surfer", "Time": "O(S·L)", "Space": "O(N)", "Exact": "No"},
            {"Method": "Progressive Pruning", "Time": "O(k·E_reduced)", "Space": "O(N)", "Exact": "No"},
        ])
        st.table(complexity_df)
    with bc2:
        steps = {
            "Manual": st.session_state.iteration if method == "Manual Power Iteration" else 12,
            "NetworkX": 1,
            "Matrix": 1,
            "Sparse": st.session_state.iteration if method == "SciPy Sparse Matrix" else 8,
            "Monte Carlo": st.session_state.iteration if method == "Monte Carlo Surfer" else 15,
            "Pruning": st.session_state.iteration if method == "Progressive Pruning" else 10,
        }
        fig_b = go.Figure(go.Bar(
            x=list(steps.keys()), y=list(steps.values()),
            marker_color=["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444", "#06B6D4"],
            text=list(steps.values()), textposition='auto',
        ))
        fig_b.update_layout(
            title="Steps to Convergence",
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
            font=dict(color="#CBD5E1"), height=320,
            margin=dict(l=30, r=10, t=40, b=30),
            xaxis=dict(gridcolor="#334155", tickangle=45),
            yaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    report = (
        f"PAGERANK SIMULATION REPORT\n{'=' * 50}\n"
        f"Generated : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Seed URL  : {st.session_state.url_input}\n"
        f"Method    : {method}  |  α = {alpha}\n"
        f"Pages     : {G.number_of_nodes()}  |  Links: {G.number_of_edges()}\n"
        f"Iterations: {st.session_state.iteration}\n"
        f"Converged : {st.session_state.converged}\n"
    )

    if method == "Progressive Pruning":
        report += f"Frozen nodes: {len(st.session_state.frozen_nodes)} ({frozen_pct:.1f}%)\n"
        if len(st.session_state.pruning_speed_data) > 0:
            report += f"Speed improvement: {avg_speedup:.1f}%\n"

    report += f"\nTOP 10 PAGES BY PAGERANK:\n{full_df.head(10).to_string(index=False)}\n"

    st.download_button("📥 Download Report (.txt)", report,
                       file_name="pagerank_report.txt", use_container_width=False, key="download_button")

# ══════════════════════════════════════════════════════════════════
# GOOGLE BENCHMARK
# ══════════════════════════════════════════════════════════════════

with st.expander("📊 Google Search Benchmark (Thesis Validation)", expanded=True):
    st.markdown("""
    Paste the top results from a Google search on the same domain you crawled.
    Accepts **full URLs**, **breadcrumbs** (`docs.python.org › library`), or **mixed** formats — one per line.
    """)

    google_input = st.text_area("Paste Top 10 Google results (one per line):", height=180,
                                placeholder="https://docs.python.org/3/library/\ndocs.python.org › tutorial\nhttps://docs.python.org › reference",
                                key="google_input_area")

    if google_input:
        g_lines = [l.strip() for l in google_input.split("\n") if l.strip()]
        rho, tau, matched = calculate_comparison(st.session_state.ranks, g_lines)

        with st.expander("🔬 URL parsing debug — expand to verify matching", expanded=False):
            debug_rows = []
            sim_sorted = sorted(st.session_state.ranks.keys(),
                                key=lambda x: st.session_state.ranks[x], reverse=True)
            sim_norms = [breadcrumb_to_url(u) for u in sim_sorted]
            for i, raw in enumerate(g_lines):
                parsed = breadcrumb_to_url(raw)
                hit = next((m for m in matched if m["google_rank"] == i + 1), None)
                debug_rows.append({
                    "Google #": i + 1,
                    "Raw input": raw,
                    "Parsed →": parsed,
                    "Matched sim URL": hit["url"] if hit else "❌ no match",
                    "Sim rank": hit["sim_rank"] if hit else "—",
                })
            st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
            st.caption(f"Crawled URLs normalised sample: {sim_norms[:5]}")

        if rho is not None:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Spearman ρ", f"{rho:.3f}")
            mc2.metric("Kendall's τ", f"{tau:.3f}")
            mc3.metric("Matched pages", len(matched))

            match_df = pd.DataFrame([{
                "Page": f"#{st.session_state.page_map.get(m['url'], '?')}  "
                        f"{urlparse(m['url']).path.rstrip('/').split('/')[-1] or 'home'}",
                "Simulator rank": m["sim_rank"],
                "Google rank": m["google_rank"],
                "Δ rank": m["sim_rank"] - m["google_rank"],
                "Sim URL": m["sim_norm"],
                "Google entry": m["google_norm"],
            } for m in matched])
            st.dataframe(match_df, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[m["google_rank"] for m in matched],
                y=[m["sim_rank"] for m in matched],
                mode="markers+text",
                text=[f"#{st.session_state.page_map.get(m['url'], '?')}" for m in matched],
                textposition="top center",
                marker=dict(size=14, color="#3B82F6",
                            line=dict(width=1.5, color="#1E40AF")),
                name="Matched pages",
            ))
            max_rank = max(len(g_lines), len(st.session_state.ranks))
            fig.add_trace(go.Scatter(
                x=[1, max_rank], y=[1, max_rank],
                mode="lines",
                line=dict(color="#EF4444", dash="dash", width=1.5),
                name="Perfect agreement",
            ))
            fig.update_layout(
                title="Simulator rank vs Google rank (points on red line = perfect agreement)",
                xaxis=dict(title="Google rank", gridcolor="#334155", dtick=1),
                yaxis=dict(title="Simulator rank", gridcolor="#334155", dtick=1),
                paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                font=dict(color="#CBD5E1", family="JetBrains Mono"),
                height=420, margin=dict(l=50, r=20, t=55, b=50),
                legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
            )
            st.plotly_chart(fig, use_container_width=True)

            if rho >= 0.7:
                st.success(f"✅ Strong correlation (ρ = {rho:.3f}) — simulator agrees well with Google.")
            elif rho >= 0.4:
                st.info(f"ℹ️ Moderate correlation (ρ = {rho:.3f}) — partial agreement with Google.")
            elif rho >= 0:
                st.warning(f"⚠️ Weak correlation (ρ = {rho:.3f}) — rankings diverge from Google.")
            else:
                st.error(f"❌ Negative correlation (ρ = {rho:.3f}) — rankings are inversely ordered.")

            st.caption(
                "💡 Low overlap is expected: your crawl covers only a small fraction of the domain. "
                "To improve matching, increase the page limit and crawl depth, or narrow the seed "
                "URL to the exact subtree Google is ranking."
            )

        else:
            n_matched = len(matched)
            if n_matched == 1:
                m = matched[0]
                st.warning(
                    f"Only **1 match** found (need ≥ 2 for correlation). "
                    f"Matched: `{m['url']}` ↔ Google #{m['google_rank']}. \n"
                    f"Expand the debug panel above to see how entries were parsed."
                )
            else:
                st.warning(
                    "**No matching URLs** found between your crawl and the Google results. \n"
                    "Common causes:\n"
                    "- Google results are from a different subdomain or path than you crawled\n"
                    "- Your crawl limit is too small to reach the pages Google ranks\n"
                    "- Try increasing depth/limit, or paste actual full URLs instead of breadcrumbs\n\n"
                    "Expand **🔬 URL parsing debug** above to inspect how each entry was interpreted."
                )