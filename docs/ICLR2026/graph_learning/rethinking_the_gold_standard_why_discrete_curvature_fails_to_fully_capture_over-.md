---
title: >-
  [Paper Note] Rethinking the Gold Standard: Why Discrete Curvature Fails to Fully Capture Over-squashing in GNNs?
description: >-
  [ICLR 2026][Graph Learning][over-squashing] This paper systematically refutes the "high negative curvature = over-squashing" gold standard in graph learning. By constructing a counterexample graph family, it proves that high negative curvature is a **sufficient but not necessary** condition for over-squashing. The authors propose the MOSR metric to quantify that
tags:
  - ICLR 2026
  - Graph Learning
  - over-squashing
  - MinHash
date: 2026-05-08
content_hash: f9905b5833974872
---
# Rethinking the Gold Standard: Why Discrete Curvature Fails to Fully Capture Over-squashing in GNNs?

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QYtmqCoilk](https://openreview.net/forum?id=QYtmqCoilk)  
**Code**: Anonymous link provided in the paper appendix (to be confirmed)  
**Area**: Graph Learning / GNN Theory / Discrete Curvature / Over-squashing  
**Keywords**: over-squashing, discrete curvature, Forman curvature, graph rewiring, MinHash

## TL;DR
This paper systematically refutes the "high negative curvature = over-squashing" gold standard in graph learning. By constructing a counterexample graph family, it proves that high negative curvature is a **sufficient but not necessary** condition for over-squashing. The authors propose the MOSR metric to quantify that curvature misses 30%–40% of squashed edges and introduce a new weighted curvature, WAF3, along with a linear-time MinHash approximation algorithm (23.6 seconds for a graph with 5 million edges, 133.7x faster than existing methods).

## Background & Motivation

**Background**: Over-squashing is a core pathology in Message Passing Graph Neural Networks (MPNNs). When information flows from a source node to a distant target node across "bottleneck" structures, it is compressed into narrow channels, leading to severe distortion. Topping et al. (2021) proposed a widely accepted criterion: **edges with high negative discrete curvature are the ones creating bottlenecks and causing over-squashing**. This view has catalyzed numerous works, including curvature-based graph rewiring and curvature-inspired GNN architectures, establishing discrete curvature as the "gold standard" for detecting over-squashing.

**Limitations of Prior Work**: However, Topping et al. only proved the direction of "high negative curvature $\implies$ over-squashing" (sufficiency) and never established the reverse (necessity). Consequently, no prior work validated whether "all severely squashed edges exhibit high negative curvature." This subtle but critical gap has been ignored by the community—if necessity does not hold, then many curvature-based screening methods systematically miss a significant portion of truly squashed edges.

**Key Challenge**: Discrete curvature (whether Ollivier-Ricci, Balanced Forman, or Augmented Forman) essentially characterizes how tightly a **first-order neighborhood** of an edge $(u,v)$ is connected, primarily determined by the number of triangles the edge participates in. In contrast, the true measure of over-squashing is the Jacobian norm $\|\partial h_t^{(L)}/\partial h_s^{(0)}\|$, which depends on **multi-hop information propagation paths**. Using a first-order local metric to approximate a multi-hop global phenomenon naturally creates blind spots.

**Goal**: The objectives are divided into four interconnected sub-problems: (1) Does necessity hold theoretically? (Theoretical) (2) How many squashed edges does curvature miss in real-world graphs, and where are they? (Empirical) (3) Can a new curvature be designed with fewer missed detections? (Methodological) (4) Can the new curvature be computed on large-scale graphs with billions of edges? (Algorithmic)

**Key Insight**: The authors start with a counterintuitive observation: as long as the first-order neighbors of edge $(u,v)$ connect to a large number of "fan-out" nodes, information is "diluted" by these nodes at every step of multi-hop propagation, causing severe over-squashing. However, if $(u,v)$ itself is part of many triangles, its curvature remains highly positive. This observation identifies the rift where "local tightness $\neq$ global throughput."

**Core Idea**: Use "weighting" to correct the erroneous counting of high-degree nodes in curvature (as high-degree nodes actually contribute little to localized information flow). The triangle-counting-based Augmented Forman-3 is upgraded to WAF3 and rewritten as a weighted Jaccard similarity to achieve linear acceleration via MinHash.

## Method

### Overall Architecture

This is a "deconstruct then reconstruct" theory and methodology paper. The logical flow is: **Refuting the gold standard with counterexamples $\to$ Quantifying missed detections with metrics $\to$ Repairing with a new curvature $\to$ Implementing via approximation algorithms**.

First (Deconstruction), the authors construct a counterexample graph family $G^c_{n,m}$ to theoretically prove the existence of edges that are severely squashed yet rated highly positive by eight mainstream discrete curvatures. Second (Quantification), they define the Missed Over-Squashing Ratio (MOSR) metric, using the Jacobian norm as the ground truth to count how many truly squashed edges are missed by curvature. In 21 datasets, Ollivier-Ricci misses over 30%, and edge betweenness reveals that curvature only catches "inter-cluster bridge edges" while systematically ignoring intra-cluster squashed edges. Third (Reconstruction), Augmented Forman-3 is rewritten into an equivalent form of "number of nodes in triangles minus number of nodes outside triangles," adding a degree-weighted correction function $f$ to obtain WAF3. Fourth (Implementation), WAF3 is rewritten as a weighted Jaccard similarity, using MinHash to reduce the per-edge calculation to constant complexity $O(H|E|)$.

The following sections expand on the key designs of the counterexample, metric, new curvature, and approximation.

### Key Designs

**1. Counterexample Graph Family $G^c_{n,m}$: Disproving Curvature Necessity via "Fan-out Dilution"**

To disprove the unverified hypothesis that "high negative curvature is a necessary condition for over-squashing," the authors construct a controllable counterexample. $G^c_{n,m}$ contains source $s$, target $t$, $n$ first-order neighbors $N_1=\{u_i\}$ (each $u_i$ connected to both $s$ and $t$), and $n\times m$ second-order neighbors $N_2=\{v_{ij}\}$ (each $u_i$ fans out to $m$ nodes $v_{ij}$). This design creates a divergence: $(s,t)$ participates in $n$ triangles, reaching the maximum possible triangle count, thus $\mathrm{Curv}(s,t)$ is highly positive across nearly all definitions. However, information from $s$ to $t$ must pass through $N_1$, where each $u_i$ dilutes the signal among $m$ neighbors.

Theorem 4 rigorously proves this divergence: as $m\to+\infty$, the Jacobian norm approaches its theoretical lower bound (maximal squashing) at a rate of $O(m^{-1})$:

$$\phi_L(a,b) := \prod_{l=0}^{L-1}\|W^{(l)}\|\left(\tfrac{1}{a+1}+\tfrac{1}{b+1}\right)^{L-1}\frac{\rho}{\sqrt{(a+1)(b+1)}}$$

Yet, eight curvatures including $\alpha$-Ollivier-Ricci, Balanced Forman, and Augmented Forman all satisfy $\mathrm{Curv}(s,t)>c>0$. The root cause is that discrete curvature only considers first-order neighbors, while GNNs require at least two rounds of message passing; first-order metrics cannot capture multi-hop dilution.

**2. MOSR: Quantifying Missed Squashed Edges**

The authors define the Missed Over-Squashing Ratio (MOSR). First, the Jacobian norm $\mathrm{JacoNorm}(u,v):=\|\partial h_v^{(L)}/\partial h_u^{(0)}\|_F$ is used as the ground truth for over-squashing. The set of high negative curvature edges is denoted as $E_q := \{e\in E \mid \mathrm{Curv}(e)\le \mathrm{Percentile}(C^-,q)\}$. Let $J_q:=\max_{(u,v)\in E_q}\mathrm{JacoNorm}(u,v)$ be the "curvature-accepted squashing threshold." MOSR is defined as:

$$\mathrm{MOSR}_q := \frac{\sum_{(u,v)\in E}\mathbf{1}_{\mathrm{Curv}(u,v)\ge 0}\cdot \mathbf{1}_{\mathrm{JacoNorm}(u,v)\le J_q}}{\sum_{(u',v')\in E}\mathbf{1}_{\mathrm{JacoNorm}(u',v')\le J_q}}$$

The numerator counts edges that are non-negative in curvature but are actually squashed, while the denominator counts all truly squashed edges. This metric objectively exposes curvature blind spots by using the model's ground truth. Additionally, the authors use edge betweenness to define BetwIden, BetwAll, and BetwIgno to characterize the topological locations of missed edges.

**3. WAF3: Correcting Miscounted High-Degree Nodes with Weighting**

Finding that Augmented Forman-3 (AF3) has the lowest complexity and few misses, the authors propose an equivalent form:

$$AF3(u,v) = 4 - d_u - d_v + 3\triangle(u,v) = \underbrace{|B(u)\cap B(v)|}_{\text{Nodes in triangles}} - \underbrace{\big(|N(u)/B(v)| + |N(v)/B(u)|\big)}_{\text{Nodes outside triangles}}$$

WAF3 introduces a weighting function $f:\mathbb{R}\to\mathbb{R}$ based on degree:

$$\mathrm{WAF3}_f(u,v) := \sum_{i\in B(u)\cap B(v)} f(d_i) - \Big(\sum_{i\in N(u)/B(v)} f(d_i) + \sum_{i\in N(v)/B(u)} f(d_i)\Big)$$

Theorem 5 proves that as long as $f(+\infty)=0^+$ (weight of high-degree nodes tends to zero), WAF3 resolves the $G^c_{n,m}$ counterexample. The paper adopts a GCN-style weight $f(x)=1/(1+x)$.

**4. MinHash Approximation: Linearizing Complexity**

To overcome the $d_{\max}$ factor in set intersection complexity, Theorem 6 rewrites WAF3 using **Weighted Jaccard Similarity**:

$$\mathrm{Jaccard}_f(N(u),N(v)) := \frac{\sum_{i\in N(u)\cap N(v)} f(d_i)}{\sum_{i\in N(u)\cup N(v)} f(d_i)}$$

By using MinHash to sample $H$ hash functions, single-edge similarity calculation drops to constant $O(H)$, reaching a total graph complexity of $O(H|E|)$. This allows for a new paradigm: quickly screening candidates with the approximation and performing exact calculations only on the candidate set.

## Key Experimental Results

Experiments cover 3 mainstream curvatures across 3 GNNs on 21 datasets.

### Main Results: Missed Over-Squashing Ratio (GCN, $q=25$)

| Dataset | Ollivier-Ricci | Augmented Forman-3 | WAF3 (Ours) |
| :--- | :--- | :--- | :--- |
| Cora | 0.103 | 0.027 | 0.014 |
| Squirrel | 0.723 | 0.137 | 0.039 |
| **Average (GCN, MOSR₁₀/₂₅)** | .271/.307 | .067/.079 | **.036/.045** |

- Observation 1: Discrete curvature systematically misses 6.7%–38.6% of squashed edges at $q=10$, proving it is not a perfect indicator.
- Observation 3: Ollivier-Ricci, despite its high complexity, misses the most edges, while AF3 performs best among legacy metrics.
- WAF3 further reduces the average MOSR for GCN to **3.6%/4.5%**.

### Efficiency and Accuracy (ER Graph, 5M edges)

- Observation 7: For a graph with $5\times10^6$ edges, exact calculations for ORC or BFC are infeasible. MinHash approximation ($H=100$) takes only 23.6 seconds, a **133.7x gain** in speed.
- Observation 8: At $H=100$, the Kendall Tau-b rank correlation between the approximation and the exact value is approximately 95%, which is sufficient for rewiring tasks that only care about the relative order of the lowest curvature edges.

### Key Findings
- Triangle counting is the primary blind spot: high-degree nodes contribute to triangle counts but do not transmit information effectively.
- Curvature is actually a "gold standard for inter-cluster bridge detection" rather than for over-squashing per se, as it systematically ignores intra-cluster squashed edges (Observation 4 & 5).

## Highlights & Insights
- **Refutation over Construction**: The most significant contribution is splitting the community's assumption into sufficiency and necessity, proving the latter false with the minimal $G^c_{n,m}$ model.
- **Metric-Driven Quantification**: MOSR transforms a conceptual debate into a measurable benchmark (Jacobian norm vs. Curvature), providing 350 reproducible data points.
- **Complexity Breakthrough**: Rewriting curvature as weighted Jaccard (Theorem 6) allows the use of MinHash, hitting the theoretical complexity lower bound for large-scale graph analysis.

## Limitations & Future Work
- WAF3 still relies on first-order neighborhoods; while it addresses the "high degree dilution," the inherent tension of using local metrics for multi-hop phenomena remains.
- The Jacobian norm ground truth relies on the assumption of uniform path activation probability, which may not hold exactly after model training.
- The paper focuses on MOSR accuracy; more extensive end-to-end downstream task performance comparisons for WAF3-based rewiring could be explored.

## Related Work & Insights
- **vs. Topping et al. (2021)**: This work redefines curvature as a "bridge-edge detector" rather than an "over-squashing detector," proving their criterion is sufficient but not necessary.
- **vs. Di Giovanni et al. (2023)**: While they use Jacobian norms to characterize over-squashing, this paper focuses on the previously unquantified "missed detection rate" of curvature as a proxy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to disprove curvature necessity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive MOSR data, though downstream benchmarks are secondary.
- Writing Quality: ⭐⭐⭐⭐⭐ Very clear logical flow from theory to empirical to algorithm.
- Value: ⭐⭐⭐⭐⭐ Corrects a long-standing assumption in the community.

<!-- RELATED:START -->
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity](glstm_mitigating_over-squashing_by_increasing_storage_capacity.md)
- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](cords_-_continuous_representations_of_discrete_structures.md)
- [\[ICLR 2026\] Discrete Bayesian Sample Inference for Graph Generation](discrete_bayesian_sample_inference_for_graph_generation.md)
- [\[ICLR 2026\] GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)
- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)

</div>

<!-- RELATED:END -->
