import os
import networkx as nx

def generate_site(folder_name, link_structure):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pages = [f"page{i}.html" for i in range(15)]
    for i in range(15):
        with open(os.path.join(folder_name, pages[i]), "w") as f:
            f.write(f"<html><body><h1>Page {i}: Metro Daily</h1>")
            targets = link_structure.get(i, [])
            for t in targets:
                f.write(f'<p><a href="{pages[t]}">Link to Page {t}</a></p>')
            f.write("</body></html>")

def compute_all_pageranks(link_structure):
    """Compute PageRank for all nodes and return sorted results"""
    G = nx.DiGraph()
    for u, targets in link_structure.items():
        for v in targets:
            G.add_edge(u, v)
    # Add all nodes (0..14) to ensure no missing nodes
    for i in range(15):
        G.add_node(i)
    pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-9)
    return pr

# ============================================================================
# BASELINE: Metro Daily with poor structure
# ============================================================================
baseline = {
    0: [1],       # Homepage → Top Story
    1: [2],       # Top Story → Local News
    2: [3],       # Local News → Sports
    3: [4],       # Sports → Weather
    4: [0],       # Weather → back to Homepage (weak cycle)
    5: [0],       # About Us → Homepage
    6: [0],       # Contact → Homepage
    7: [0],       # Old Article 1 → Homepage
    8: [0],       # Old Article 2 → Homepage
    9: [0],       # Old Article 3 → Homepage
    10: [0],      # Advertising → Homepage
    11: [0],      # Terms → Homepage
    12: [0],      # Privacy → Homepage
    13: [],       # RSS Feed – DANGLING (no outlinks)
    14: []        # Print Edition – DANGLING
}
generate_site("metro_1_baseline", baseline)

# ============================================================================
# RECOMMENDATION 4: Fix Dangling Pages (Do FIRST - technical foundation)
# ============================================================================
# Fix authority leaks by adding outlinks from dangling pages
rec4 = baseline.copy()
rec4[13] = [0]   # RSS Feed → Homepage
rec4[14] = [0]   # Print Edition → Homepage
generate_site("metro_2_rec4_fix_dangling", rec4)

# ============================================================================
# RECOMMENDATION 5: Hub-and-Spoke Structure (Fix navigation SECOND)
# ============================================================================
# Convert from chain to hub-and-spoke for better authority distribution
rec5 = rec4.copy()
rec5[0] = [1, 2, 3, 4]   # Homepage links directly to ALL major sections
# Remove the chain links that cause damping loss
rec5[1] = []             # Top Story no longer forced to link only to Local
rec5[2] = []             # Local News no longer forced to link to Sports
rec5[3] = []             # Sports no longer forced to link to Weather
rec5[4] = [0]            # Weather links back to Homepage
generate_site("metro_3_rec5_hub_and_spoke", rec5)

# ============================================================================
# RECOMMENDATION 2: Internal Hub Page (Create topical authority THIRD)
# ============================================================================
# Make Page 1 the central hub for all old content
rec2 = rec5.copy()
for i in [7, 8, 9, 10, 11, 12]:
    rec2[i] = [1]      # All old articles + policies → Top Story hub
generate_site("metro_4_rec2_internal_hub", rec2)

# ============================================================================
# RECOMMENDATION 3: Multiple Inbound Links (Accumulate votes FOURTH)
# ============================================================================
# Make most pages point to Page 1 to accumulate authority
rec3 = rec2.copy()
for i in range(15):
    if i != 1:  # except Page 1 itself
        if rec3.get(i) is None:
            rec3[i] = [1]
        else:
            if 1 not in rec3[i]:
                rec3[i].append(1)  # add link to Top Story
generate_site("metro_5_rec3_multiple_inbounds", rec3)

# ============================================================================
# RECOMMENDATION 1: High-Authority Inbound Link (External trust FINALLY)
# ============================================================================
# FIXED: Simulate external authority by adding MULTIPLE links from Homepage
# to represent a high-authority external backlink
rec1 = rec3.copy()
# Add TWO links from Homepage to Top Story (simulating external authority boost)
# This represents getting a backlink from a site like BBC or NYTimes
rec1[0] = [1, 1, 2, 3, 4]  # Two votes to Page 1 from the most authoritative page
generate_site("metro_6_rec1_high_authority_link", rec1)

# ============================================================================
# Print PageRank progression for verification
# ============================================================================
print("\n" + "="*60)
print("METRO DAILY PAGE RANK PROGRESSION")
print("="*60)

baseline_pr = compute_all_pageranks(baseline)
rec4_pr = compute_all_pageranks(rec4)
rec5_pr = compute_all_pageranks(rec5)
rec2_pr = compute_all_pageranks(rec2)
rec3_pr = compute_all_pageranks(rec3)
rec1_pr = compute_all_pageranks(rec1)

print(f"\n{'Step':<30} {'Page1 PR':<12} {'Change':<12}")
print("-"*60)
print(f"{'1. Baseline (poor)':<30} {baseline_pr[1]:<12.6f} {'':<12}")
print(f"{'2. After Rec4 (Fix Dangling)':<30} {rec4_pr[1]:<12.6f} {(rec4_pr[1]-baseline_pr[1])/baseline_pr[1]*100:+.1f}%")
print(f"{'3. After Rec5 (Hub-and-Spoke)':<30} {rec5_pr[1]:<12.6f} {(rec5_pr[1]-rec4_pr[1])/rec4_pr[1]*100:+.1f}%")
print(f"{'4. After Rec2 (Internal Hub)':<30} {rec2_pr[1]:<12.6f} {(rec2_pr[1]-rec5_pr[1])/rec5_pr[1]*100:+.1f}%")
print(f"{'5. After Rec3 (Multiple Inbounds)':<30} {rec3_pr[1]:<12.6f} {(rec3_pr[1]-rec2_pr[1])/rec2_pr[1]*100:+.1f}%")
print(f"{'6. After Rec1 (External Link)':<30} {rec1_pr[1]:<12.6f} {(rec1_pr[1]-rec3_pr[1])/rec3_pr[1]*100:+.1f}%")
print("="*60)

# Print detailed rankings for each stage
def print_detailed_rankings(pr_dict, stage_name):
    print(f"\n{stage_name} - Top 10 Pages:")
    sorted_pr = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (node, pr) in enumerate(sorted_pr[:10], 1):
        page_name = f"Page{node}"
        if node == 0: page_name = "Homepage"
        elif node == 1: page_name = "Top Story"
        elif node == 2: page_name = "Local News"
        elif node == 3: page_name = "Sports"
        elif node == 4: page_name = "Weather"
        print(f"  {i:2}. {page_name:<12} (node{node}): {pr:.6f}")

print_detailed_rankings(baseline_pr, "BASELINE")
print_detailed_rankings(rec4_pr, "AFTER REC4 (Fix Dangling)")
print_detailed_rankings(rec5_pr, "AFTER REC5 (Hub-and-Spoke)")
print_detailed_rankings(rec2_pr, "AFTER REC2 (Internal Hub)")
print_detailed_rankings(rec3_pr, "AFTER REC3 (Multiple Inbounds)")
print_detailed_rankings(rec1_pr, "AFTER REC1 (External Link)")