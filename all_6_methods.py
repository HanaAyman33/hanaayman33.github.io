import random
import networkx as nx
import numpy as np
import scipy.sparse as sp

G = nx.DiGraph()
G.add_edges_from([   ('A', 'B'), ('A', 'C'),
                     ('B', 'C'), ('B', 'D'),
                     ('C', 'A'),
                     ('D', 'C'),
                     ('E', 'D')])

# Method1: NetworkX Built-in Function Method
pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
print('=== NetworkX Built-in PageRank ===')
for page, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
    print(f'  Page {page}: {score:.6f}')

# Method2: Manual Power Iteration
def manual_pagerank(G, alpha=0.85, max_iter=100, tol=1e-6):
    N     = len(G)
    nodes = list(G.nodes())
    rank  = {node: 1 / N for node in nodes}
    for iteration in range(max_iter):
        new_rank     = {}
        dangling_sum = sum(rank[n] for n in nodes if G.out_degree(n) == 0)
        for node in nodes:
            new_rank[node]  = (1 - alpha) / N
            new_rank[node] += alpha * dangling_sum / N
            for pred in G.predecessors(node):
                out_deg = G.out_degree(pred)
                if out_deg > 0:
                    new_rank[node] += alpha * rank[pred] / out_deg
                    error = sum(abs(new_rank[n] - rank[n]) for n in nodes)
                    rank  = new_rank
                    if error < tol:
                        print(f'  Converged after {iteration+1} iterations.')
                        break
                    return rank

# Method3: Matrix Algebra
def matrix_algebra_pagerank(G, alpha=0.85):
    nodes = list(G.nodes())
    N     = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    M = np.zeros((N, N))
    for node in G.nodes():
        out_deg = G.out_degree(node)
        if out_deg > 0:
            for nb in G.successors(node):
                M[index[nb]][index[node]] = 1.0 / out_deg
        else:
            M[:, index[node]] = 1.0 / N   # dangling node fix
        A = np.eye(N) - alpha * M
        b = np.ones(N) * (1 - alpha) / N
        r = np.linalg.solve(A, b)
        r = r / r.sum()
    return {nodes[i]: r[i] for i in range(N)}, M

# Method4: Scipy Sparse
def sparse_pagerank(G, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = list(G.nodes())
    N     = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    rows_idx, cols_idx, values = [], [], []
    dangling_indices = []
    for node in G.nodes():
        out_deg = G.out_degree(node)
        col     = index[node]
        if out_deg > 0:
            weight = 1.0 / out_deg
            for nb in G.successors(node):
                rows_idx.append(index[nb]); cols_idx.append(col); values.append(weight)
        else:
            dangling_indices.append(col)
    M        = sp.csr_matrix((values,(rows_idx,cols_idx)), shape=(N,N))
    rank     = np.full(N, 1.0/N)
    teleport = np.full(N, 1.0/N)
    for it in range(max_iter):
        dangling  = alpha * rank[dangling_indices].sum() / N if dangling_indices else 0
        new_rank  = alpha * M.dot(rank) + dangling + (1-alpha) * teleport
        if np.abs(new_rank - rank).sum() < tol:
            print(f'  Converged after {it+1} iterations.')
            break
        rank = new_rank
    return {nodes[i]: rank[i] for i in range(N)}

# Method5: Monte Carlo
def monte_carlo_pagerank(G, num_surfers=10000, steps=100, alpha=0.85, seed=42):
    random.seed(seed)
    nodes       = list(G.nodes())
    N           = len(nodes)
    index       = {node: i for i, node in enumerate(nodes)}
    visit_count = np.zeros(N, dtype=int)
    for _ in range(num_surfers):
        current = random.choice(nodes)
        for _ in range(steps):
            visit_count[index[current]] += 1
            if random.random() < alpha:
                neighbors = list(G.successors(current))
                current   = random.choice(neighbors) if neighbors else random.choice(nodes)
            else:
                current = random.choice(nodes)
    total = visit_count.sum()
    return {nodes[i]: visit_count[i]/total for i in range(N)}

# Method6: HITS Algorithm
def manual_hits(G, max_iter=100, tol=1e-6):
    nodes = list(G.nodes())
    hub   = {n: 1.0 for n in nodes}
    auth  = {n: 1.0 for n in nodes}
    for it in range(max_iter):
        new_auth = {n: sum(hub[p]  for p in G.predecessors(n)) for n in nodes}
        new_hub  = {n: sum(new_auth[s] for s in G.successors(n))  for n in nodes}
        a_norm   = np.sqrt(sum(v**2 for v in new_auth.values())) or 1
        h_norm   = np.sqrt(sum(v**2 for v in new_hub.values()))  or 1
        new_auth = {n: v/a_norm for n,v in new_auth.items()}
        new_hub  = {n: v/h_norm for n,v in new_hub.items()}
        err      = sum(abs(new_auth[n]-auth[n])+abs(new_hub[n]-hub[n]) for n in nodes)
        auth, hub = new_auth, new_hub
        if err < tol:
            print(f'  HITS converged after {it+1} iterations.')
            break
    return hub, auth
