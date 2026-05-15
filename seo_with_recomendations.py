import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ============================================================================
# METRO DAILY SEO PROGRESSION DATA (from your actual output)
# ============================================================================
steps = ['Baseline', 'Rec 4\nFix Dangling', 'Rec 5\nHub-and-Spoke',
         'Rec 2\nInternal Hub', 'Rec 3\nMultiple Links', 'Rec 1\nExternal Link']
pr_values = [0.194955, 0.201128, 0.280296, 0.395902, 0.395902, 0.395902]
improvements = [0, 3.2, 39.4, 41.2, 0, 0]
cumulative = [0, 3.2, 43.8, 103.0, 103.0, 103.0]

# ============================================================================
# FIGURE 4.3 — Metro Daily SEO Case Study
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Metro Daily Case Study: SEO Recommendations & Diminishing Returns',
             fontsize=14, fontweight='bold', color='#1F3864')

# ──────────────────────────────────────────────────────────────────────────
# TOP-LEFT: PageRank Progression with saturation plateau
# ──────────────────────────────────────────────────────────────────────────
ax1 = axes[0, 0]
colors = ['#C00000' if i == 0 else '#70AD47' if i == 3 else '#2E74B5' for i in range(len(steps))]
bars = ax1.bar(steps, pr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax1.axhline(y=0.396, color='#C00000', linestyle='--', linewidth=1.5,
            label=f'Saturation Ceiling (≈0.396)')
ax1.set_ylabel('Top Story (Page 1) PageRank', fontsize=11, fontweight='bold')
ax1.set_xlabel('SEO Improvement Step', fontsize=11, fontweight='bold')
ax1.set_title('(a) PageRank Progression: +103% Total Improvement', fontsize=11)
ax1.set_ylim(0, 0.45)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars, pr_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ──────────────────────────────────────────────────────────────────────────
# TOP-RIGHT: Diminishing Returns — Marginal gain per step
# ──────────────────────────────────────────────────────────────────────────
ax2 = axes[0, 1]
x_pos = np.arange(len(steps))
colors_marginal = ['gray', '#2E74B5', '#2E74B5', '#70AD47', '#C00000', '#C00000']
bars2 = ax2.bar(x_pos, improvements, color=colors_marginal, alpha=0.8, edgecolor='black', linewidth=1)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Baseline', 'Rec 4\n+3.2%', 'Rec 5\n+39.4%', 'Rec 2\n+41.2%', 'Rec 3\n+0%', 'Rec 1\n+0%'])
ax2.set_ylabel('Marginal Improvement (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('SEO Action', fontsize=11, fontweight='bold')
ax2.set_title('(b) Diminishing Returns: Saturation After Rec 2', fontsize=11)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotations
ax2.annotate('Structural\nImprovements', xy=(1.5, 45), xytext=(1.5, 55),
             ha='center', fontsize=9, color='#2E74B5', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#2E74B5'))
ax2.annotate('Saturation\n(0% gain)', xy=(4.5, 5), xytext=(4.5, 20),
             ha='center', fontsize=9, color='#C00000', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#C00000'))

# ──────────────────────────────────────────────────────────────────────────
# BOTTOM-LEFT: Ranking evolution — How Top Story rises to #1
# ──────────────────────────────────────────────────────────────────────────
ax3 = axes[1, 0]

# Ranking data (position, lower is better)
ranks = {
    'Baseline': {'Top Story': 2, 'Homepage': 1, 'Local News': 3, 'Sports': 4, 'Weather': 5},
    'After Rec4': {'Top Story': 2, 'Homepage': 1, 'Local News': 3, 'Sports': 4, 'Weather': 5},
    'After Rec5': {'Top Story': 1, 'Homepage': 2, 'Local News': 3, 'Sports': 4, 'Weather': 5},
    'After Rec2+': {'Top Story': 1, 'Homepage': 2, 'Local News': 3, 'Sports': 4, 'Weather': 5}
}

pages = ['Top Story', 'Homepage', 'Local News', 'Sports', 'Weather']
colors_pages = ['#C00000', '#1F3864', '#2E74B5', '#2E74B5', '#2E74B5']

x_labels = ['Baseline', 'After Rec4', 'After Rec5', 'After Rec2+']
x_pos_rank = np.arange(len(x_labels))

for i, page in enumerate(pages):
    y_vals = [ranks[stage][page] for stage in x_labels]
    ax3.plot(x_pos_rank, y_vals, 'o-', color=colors_pages[i], linewidth=2,
             markersize=8, label=page)

ax3.set_xticks(x_pos_rank)
ax3.set_xticklabels(x_labels, fontsize=9)
ax3.set_ylabel('Rank (1 = Best)', fontsize=11, fontweight='bold')
ax3.set_xlabel('SEO Stage', fontsize=11, fontweight='bold')
ax3.set_title('(c) Ranking Evolution: Top Story Climbs to #1', fontsize=11)
ax3.invert_yaxis()  # Lower rank is better
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper right', fontsize=9)

# Highlight the flip point
ax3.axvline(x=1.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax3.text(1.5, 1.2, 'Hub-and-Spoke\nApplied', ha='center', fontsize=8, color='gray')

# ──────────────────────────────────────────────────────────────────────────
# BOTTOM-RIGHT: PageRank distribution comparison (Chain vs Hub-and-Spoke)
# ──────────────────────────────────────────────────────────────────────────
ax4 = axes[1, 1]

# Chain structure PR values (simulated based on your chain)
chain_pr = {'Page A (Start)': 0.081, 'Page B': 0.150, 'Page C': 0.209,
            'Page D': 0.259, 'Page E (End)': 0.301}
hub_pr = {'Page A': 0.105, 'Page B': 0.105, 'Page C': 0.105,
          'Page D': 0.105, 'Page E': 0.105}

x_pages = list(chain_pr.keys())
x_pos_pages = np.arange(len(x_pages))
width = 0.35

bars3 = ax4.bar(x_pos_pages - width/2, chain_pr.values(), width,
                label='Chain Structure', color='#C00000', alpha=0.8)
bars4 = ax4.bar(x_pos_pages + width/2, hub_pr.values(), width,
                label='Hub-and-Spoke', color='#2E74B5', alpha=0.8)

ax4.set_xticks(x_pos_pages)
ax4.set_xticklabels(x_pages, fontsize=9, rotation=15)
ax4.set_ylabel('PageRank Score', fontsize=11, fontweight='bold')
ax4.set_xlabel('Page Position', fontsize=11, fontweight='bold')
ax4.set_title('(d) Chain vs Hub-and-Spoke: Rank Distribution', fontsize=11)
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Annotate the improvement
ax4.annotate('Start page +29.8%', xy=(0, 0.081), xytext=(0, 0.14),
             ha='center', fontsize=8, color='#C00000',
             arrowprops=dict(arrowstyle='->', color='#C00000'))
ax4.annotate('End page -65.0%', xy=(4, 0.301), xytext=(3.5, 0.35),
             ha='center', fontsize=8, color='#C00000',
             arrowprops=dict(arrowstyle='->', color='#C00000'))

plt.tight_layout()
plt.savefig('metro_daily_seo_figure.png', dpi=150, bbox_inches='tight')
print("Figure saved: metro_daily_seo_figure.png")
plt.show()

# ============================================================================
# PRINT SUMMARY TABLE FOR THESIS
# ============================================================================
print("\n" + "="*70)
print("METRO DAILY SEO CASE STUDY — SUMMARY TABLE FOR THESIS")
print("="*70)
print(f"\n{'Step':<35} {'Page 1 PR':<12} {'Improvement':<15}")
print("-"*62)
print(f"{'1. Baseline (poor chain + dangling)':<35} {0.194955:<12.6f} {'—':<15}")
print(f"{'2. Rec 4 — Fix dangling pages':<35} {0.201128:<12.6f} {'+3.2%':<15}")
print(f"{'3. Rec 5 — Hub-and-spoke navigation':<35} {0.280296:<12.6f} {'+39.4%':<15}")
print(f"{'4. Rec 2 — Internal hub page':<35} {0.395902:<12.6f} {'+41.2%':<15}")
print(f"{'5. Rec 3 — Multiple inbound links':<35} {0.395902:<12.6f} {'+0.0% (saturated)':<15}")
print(f"{'6. Rec 1 — High-authority external':<35} {0.395902:<12.6f} {'+0.0% (saturated)':<15}")
print("="*70)
print("\nTOTAL CUMULATIVE IMPROVEMENT FROM BASELINE: +103%")
print("="*70)