# Implementing PageRank Using Python NetworkX

**Bachelor's Thesis — Faculty of Media Engineering and Technology**
**German University in Cairo (GUC) · June 2026**

> *From an invisible startup to the top of search results — all through graph structure.*

---

## Overview

This repository contains the full implementation and experimental codebase for my bachelor's thesis on **PageRank computation, analysis, and application**. The thesis implements six distinct PageRank computation methods, introduces a novel **Progressive Pruning** algorithm, and applies the results to a real-world SEO case study using the Metro Daily news website simulation.

All source code, experimental results, and an interactive live demo are available at the companion website:

🌐 **[hanaayman33.github.io](https://hanaayman33.github.io)**

---

## Demo

### Interactive PageRank Simulator
> 📹 **Semonstrating the live BFS crawler, step-by-step power iteration, Progressive Pruning with frozen node visualisation, TF-IDF search, and Google benchmark comparison.



https://github.com/user-attachments/assets/84c1125f-af68-4359-a521-e38b4ffed252



### Progressive Pruning Algorithm

> 📹 **Progressive Pruning  — showing the warmup phase, node freezing in real time (purple ❄️ nodes), and the measured speedup on web-like graphs.



https://github.com/user-attachments/assets/d2feff1a-44cb-4ac9-ac3e-c978c8d59b8b


---

## Repository Structure

| File | Lines | Description |
|---|---|---|
| `all_6_methods.py` | 124 | Six PageRank computation methods: NetworkX built-in, Manual Power Iteration, Matrix Algebra, SciPy Sparse, Monte Carlo, HITS |
| `Experiments.py` | 1,050 | Five controlled experiments — convergence speed, scalability, accuracy, damping factor sensitivity, Monte Carlo surfer count |
| `progressive_pruning.py` | 315 | Novel Progressive Pruning PageRank algorithm — freezes converged pages to reduce iteration cost |
| `realistic_data.py` | 454 | BFS web scraper, 791-node synthetic graph builder, 9-test validation suite, all six PageRank methods benchmarked |
| `search_engine.py` | 406 | Four-signal configurable search engine, five search scenarios, Google benchmark comparison, five SEO proofs |
| `seo_with_recommend.py` | 156 | Five data-proven SEO recommendations applied sequentially to the Metro Daily website demo |
| `pr_simulator.py` | 1,328 | Full Streamlit simulation engine — BFS crawler, Vis.js graph, TF-IDF search, convergence chart, Google benchmark |

**Total: 7 files · 6,767 lines of Python**

---

## Key Results

| Contribution | Result |
|---|---|
| Six PageRank methods compared under identical conditions | All convergent iterative methods reach L1 error < 4×10⁻⁷ — indistinguishable in practice |
| Progressive Pruning algorithm (novel) | 56% of nodes frozen on n=50 graph · L1 error = 0.00 in all tests |
| 791-node large-scale evaluation | Custom implementation matches NetworkX within 1.47×10⁻⁴ in 79.2 ms |
| Metro Daily SEO case study | **+103% PageRank improvement** through graph structure alone — zero advertising |
| Google search correlation | Spearman ρ = 0.685 · 100% overlap in top-5 and top-10 results |

---

## Methodology Summary

### Six Computation Methods
1. **NetworkX Built-in** — ground-truth reference, O(k·(n+m))
2. **Manual Power Iteration** — from first principles in Python dictionaries, O(k·(n+m))
3. **Matrix Algebra** — exact LU-decomposition solution, O(n³), practical up to n≈200
4. **SciPy Sparse (CSR)** — 33M× memory reduction over dense, O(k·m), fastest iterative
5. **Monte Carlo Surfer** — O(n) memory, error = O(1/√R), value in RAM-constrained settings
6. **HITS Algorithm** — hub and authority scores, a fundamentally different metric

### Progressive Pruning (Novel Contribution)
After a 10-iteration warmup phase, pages whose rank falls below τ = 0.5/N for three consecutive rounds are **frozen** — they retain their last computed score and stop receiving rank updates, while still passing link weight to active pages. This reduces per-iteration computation cost with **zero accuracy loss**.

### Five SEO Recommendations (Ordered by Impact Sequence)
Applied to the Metro Daily website in an empirically validated order:

| Step | Recommendation | PR Gain |
|---|---|---|
| 1 | Fix dangling pages | +3.2% |
| 2 | Hub-and-spoke structure | +39.4% |
| 3 | Create internal hub page | +41.2% |
| 4 | Multiple inbound links | +0% (saturated) |
| 5 | High-authority external link | +0% (saturated) |

---

## Running the Simulator

```bash
# Install dependencies
pip install streamlit networkx scipy numpy beautifulsoup4 requests plotly pandas

# Run the interactive simulator
streamlit run pr_simulator.py
```

Then navigate to `http://localhost:8501` in your browser.

The simulator supports: BFS web crawling, all six PageRank methods, step-by-step iteration, Progressive Pruning visualisation, TF-IDF search, and Google benchmark comparison.

---

## Running the Experiments

```bash
# All six methods on the toy graph
python all_6_methods.py

# Five simulation experiments (convergence, scalability, accuracy, damping, Monte Carlo)
python Experiments.py

# Large-scale 791-node evaluation
python realistic_data.py

# Progressive Pruning benchmark
python progressive_pruning.py

# SEO recommendations on Metro Daily
python seo_with_recommend.py
```

---

## Dependencies

```
python >= 3.9
networkx >= 3.0
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
streamlit >= 1.28
beautifulsoup4 >= 4.12
requests >= 2.31
plotly >= 5.15
pandas >= 2.0
```

---

## Selected References

- Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web.* Stanford InfoLab.
- Langville, A. N., & Meyer, C. D. (2004). *Deeper inside PageRank.* Internet Mathematics, 1(3), 335–380.
- Bryan, K., & Leise, T. (2006). *The $25,000,000,000 eigenvector: The linear algebra behind Google.* SIAM Review, 48(3), 569–581.
- Avrachenkov, K. et al. (2007). *Monte Carlo methods in PageRank computation.* SIAM Journal on Numerical Analysis, 45(2), 890–904.
- Kleinberg, J. M. (1999). *Authoritative sources in a hyperlinked environment.* Journal of the ACM, 46(5), 604–632.

Full bibliography (23 references) available in the thesis document.

---

## Author

**Hana Ayman Abdelhamid**
Faculty of Media Engineering and Technology — German University in Cairo
Supervised by **Dr. Islam El-maddah**

🌐 [hanaayman33.github.io](https://hanaayman33.github.io)
---
