---
title: >-
  [Paper Note] Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion
description: >-
  [ICLR 2026][learning_theory][k-center] This paper generalizes the previous "one representative per component" Metric Forest Completion (MFC) approximation algorithm into MultiRepMFC, which selects "a set of representatives per component." It provides an instance-specific $\alpha$-approximation bound using a cheaply computable cost function, while tightening
tags:
  - ICLR 2026
  - learning_theory
  - k-center
date: 2026-05-08
content_hash: f3808fac4700c9a6
---
# Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TWmS4o41oA](https://openreview.net/forum?id=TWmS4o41oA)  
**Code**: TBD (Extended based on the open-source C++ implementation of the predecessor MFC-Approx)  
**Area**: Learning-Augmented Algorithms / Graphs and Metric Spaces  
**Keywords**: Metric Minimum Spanning Tree, Learning-Augmented Algorithms, Metric Forest Completion, k-center, Approximation Algorithms

## TL;DR
This paper generalizes the previous "one representative per component" Metric Forest Completion (MFC) approximation algorithm into MultiRepMFC, which selects "a set of representatives per component." It provides an instance-specific $\alpha$-approximation bound using a cheaply computable cost function, while tightening worst-case approximation ratios from $2.62$ (MFC) and $2\gamma+1$ (Metric MST) to $2$ and $2\gamma$. The tightness of these bounds is proven, and the algorithm is shown to approach optimal spanning tree quality with minimal extra computation.

## Background & Motivation
**Background**: The Minimum Spanning Tree (MST) is a fundamental primitive for tasks such as clustering, network design, and feature selection. Metric MST is a special case where the input consists of $n$ points and edge weights are defined by a distance metric $d$. The naive approach requires explicitly computing all $O(n^2)$ distances before running Kruskal’s or Borůvka’s greedy algorithms. While fast $o(n^2)$ algorithms exist for Euclidean metrics, Indyk (1999) proved that for **arbitrary metric spaces**, $\Omega(n^2)$ edges must be examined even for approximate solutions. This is the fundamental barrier to achieving large-scale, general-distance, and guaranteed-performance simultaneously.

**Limitations of Prior Work**: The authors' previous work (Veldt et al., 2025) adopted a learning-augmented perspective: treating a "forest obtained by early termination of Kruskal/Borůvka" as a **prediction** provided by an ML heuristic—termed the initial forest. The task of completing this into a full spanning tree is called **Metric Forest Completion (MFC)**. The previous MFC-Approx selected only one representative per component and considered only edges adjacent to these representatives, yielding a $2.62$-approximation for MFC and a $(2\gamma+1)$-approximation for Metric MST (where $\gamma \ge 1$ measures initial forest quality). However, real-world performance was much better than these theoretical bounds, leaving three unanswered questions.

**Key Challenge**: A massive gap exists between theoretical bounds ($2.62$ / $2\gamma+1$) and empirical approximation ratios. Is this gap due to dataset stochasticity, or do pathological instances exist that make the old bounds tight? Can worst-case bounds be tightened while providing guarantees better suited to specific instances? Using "only one representative" clearly discards too much information.

**Goal**: (1) Design an algorithm that interpolates between "one representative" and "all points" (optimal but $\Omega(n^2)$); (2) Provide instance-specific, cheaply computable approximation bounds; (3) Tighten and prove the tightness of worst-case bounds.

**Key Insight**: Relax the "one representative per component" constraint to "a set of representatives $R_i \subseteq P_i$ per component." More representatives lead to better approximations at higher costs, forming a tunable quality-time tradeoff curve.

**Core Idea**: Interpolate between MFC-Approx and exact algorithms using MultiRepMFC with "multiple representatives per component + edges adjacent to representatives." The "optimal representative set selection" is reduced to a **shared-budget multi-instance k-center** problem.

## Method

### Overall Architecture
MFC can be viewed as finding an MST on a **coarsened graph** $G_P=(V_P,E_P)$, where each component $P_i$ is contracted into a super-node $v_i$. The super-edge weight $w^*(v_i,v_j)=d(P_i,P_j)$ is the bichromatic closest pair distance between two components. The difficulty lies in the fact that computing $w^*$ exactly requires $\Omega(n^2)$ distance queries when component sizes are balanced.

The MultiRepMFC workflow is: Given an initial forest $G_t$ (partition $P=\{P_1,\dots,P_t\}$ and intra-component trees $T_i$) $\to$ Select a set of representatives $R_i$ for each component under budget $b$ (using the BESTREPS solver) $\to$ Define an **upper-bound weight function** $\hat w \ge w^*$ using representatives $\to$ Compute MST on the coarsened graph for $\hat w$ $\to$ Map each super-edge back to its corresponding real point pair to get the completion edge set $\hat M$, and merge with initial forest edges $E_t$ to form the full tree $\hat T$. When $R_i$ is a single point, it reduces to the previous MFC-Approx; when $R=X$ (all points), $\hat w=w^*$, reducing to the exact algorithm. In other words, the budget $b$ is a "knob" tuning the same algorithm from the fastest approximation to the optimum.

### Key Designs

**1. MultiRepMFC: Interpolating between approximation and exactness via representative upper-bound weights**

Previous work used a single representative, effectively treating a component as a point to estimate distances, resulting in significant information loss and loose bounds ($2.62$). This paper selects a non-empty representative subset $R_i \subseteq P_i$ for each component. Let $R = \cup_i R_i$. Only edges adjacent to $R$ are allowed. It computes the MST on the coarsened graph for the following weight function:

$$\hat w(v_i,v_j)=\min\{\,d(P_i,R_j),\ d(P_j,R_i)\,\}.$$

This uses the minimum of "distance from all points in $P_i$ to $R_j$" and "distance from all points in $P_j$ to $R_i$" as a proxy for the true closest pair distance $w^*(v_i,v_j)=d(P_i,P_j)$. As representatives become denser, $\hat w$ approaches $w^*$. Since queries are only performed on one side (at the representatives), the query complexity is governed by the total number of representatives rather than $n^2$. Since $\hat w \ge w^*$ always holds, the final tree weight is strictly upper-bounded. This is the source of "interpolation": a knob $|R|$ that smoothly adjusts from $t$ representatives (fastest, bound of 2) to $n$ (exact, $\Omega(n^2)$).

**2. Cost function and instance-specific bound $\alpha$: Turning representative quality into cheap guarantees**

To quantify the quality of MultiRepMFC$(R)$, the cost of component $P_i$ is defined as the maximum distance from any point in the component to its nearest representative:

$$\mathrm{cost}(P_i,R_i)=\max_{x\in P_i}\ \min_{r\in R_i}\ d(x,r),\qquad \mathrm{cost}(P,R)=\sum_{i=1}^{t}\mathrm{cost}(P_i,R_i).$$

This is the k-center radius when $R_i$ serves as the set of cluster centers for $P_i$. **Theorem 1** states that MultiRepMFC$(R)$ is an $\alpha$-approximation for MFC and an $\alpha\gamma$-approximation for Metric MST, where:

$$\alpha=1+\frac{\mathrm{cost}(P,R)}{w_X(E_t)}.$$

The logic: For any optimal coarsened edge $(v_i, v_j)$ corresponding to point pair $(x_a, x_b)$, let $z \in R_i$ be the nearest rep to $x_a$. By triangle inequality, $\hat w(v_i,v_j) \le d(z,x_b) \le d(x_a,x_b) + d(x_a,z) \le w^*(v_i,v_j) + \mathrm{cost}(P_i)$. Summing over edges yields the bound. The significance of $\alpha$ is that it **depends only on cost and initial forest weight**, making it computable during the MultiRepMFC run with nearly zero overhead. This allows for **dynamic budget selection**—adding representatives until $\alpha$ is satisfactory.

**3. Corollary 2 and Tightness: Tightening bounds to 2 / 2γ**

Applying Theorem 1 to the single-representative case yields: MFC-Approx is a $2$-approximation for MFC and a $2\gamma$-approximation for Metric MST, improving the previous $2.62$ and $2\gamma+1$ bounds. The key observation is that for a single representative, $\mathrm{cost}(P_i) \le w_X(T_i)$, thus $\mathrm{cost}(P) \le w_X(E_t)$, leading to $\alpha \le 2$.

**Theorem 3** uses an $\ell_\infty$ norm construction to prove these bounds are tight in the worst-case. Pathological instances do exist, and the reason practice is better is that representatives can be **selected strategically** rather than arbitrarily.

**4. BESTREPS: Selecting representatives as shared-budget multi-instance k-center**

The remaining problem: Given budget $b$ (extra reps beyond one per component), how to choose $R$ to minimize $\mathrm{cost}(P,R)$? This is the Best Representatives (BESTREPS) problem:

$$\min\ \mathrm{cost}(P,R)\quad \text{s.t.}\quad |R_i|\ge1,\ \sum_{i=1}^{t}(|R_i|-1)\le b.$$

This is a generalization of k-center to multiple instances with a shared budget. It is NP-hard. The authors provide a $2$-approximation: run Gonzalez's (1985) greedy k-center for each component to get approximate costs $\hat c_i(j)$, then use **dynamic programming** to minimize $\sum_i \hat c_i(b_i+1)$ under the budget constraint. **Theorem 4** proves this "Greedy k-center + DP allocation" is a 2-approximation for BESTREPS.

### Loss & Training
This is a theoretical and algorithmic engineering paper; it occupies no neural network training. Three algorithm variants:
- **DP-MultiRepMFC**: Uses Theorem 4's DP allocation, $O(nQ_X(b+t)+tb^2)$, the only method with BESTREPS guarantees.
- **Greedy-MultiRepMFC**: Greedily allocates reps to the component that improves the objective most, $O(nQ_X(b+t))$.
- **Fixed($\ell$)-MultiRepMFC**: Assigns $\ell$ reps to each component, $O(nQ_X(b+t))$.

All are $2$-approximations for MFC and $2\gamma$-approximations for Metric MST.

## Key Experimental Results

Experiments focus on the MFC step, evaluating MultiRepMFC against MFC-Approx ($b=0$) and exact MFC-OPT ($b=n$), testing BESTREPS strategies, and comparing $\alpha$ against the worst-case bound of 2.

### Main Results
Using 4 datasets (Cooking, GreenGenes, FashionMNIST, Names-US) with $t=\sqrt{n}$ and averaging over 16 runs.

| Dataset | Distance Metric | $b=0$ (MFC-Approx) $\to$ Increasing Budget | $\alpha$ Performance |
| :--- | :--- | :--- | :--- |
| Cooking | Jaccard | Sharp initial drop; small $b$ significantly beats $b=0$ | $\alpha$ consistently near 1 |
| GreenGenes | Hamming | Quality rapidly approaches optimal | $\alpha$ far better than 2 |
| FashionMNIST | Euclidean | Same as above | $\alpha$ near 1 |
| Names-US | Levenshtein | Highly unbalanced forest; gains are limited for large $b$ | — |

The core conclusion: All variants provide a useful interpolation. Adding just a few representatives significantly improves tree quality, bringing it close to optimal with minor increases in runtime.

### Ablation Study (Comparison of BESTREPS Strategies)

| Configuration | Performance under fixed time budget | Note |
| :--- | :--- | :--- |
| DP-MultiRepMFC | Best real Cost Ratio; $\alpha$ significantly better; fastest gap closure | Only one with BESTREPS guarantees |
| Fixed($\ell$) | Cost Ratio often better than Greedy | Simple but effective |
| Greedy | Improvement plateaus after a while | Too short-sighted to capture long-term gains |

### Key Findings
- **$\alpha$ is an excellent proxy for real approximation ratio**: $\alpha$ is almost always very close to 1, much tighter than the worst-case bound of 2. It enables practical "dynamic budget selection" based on $\alpha$ satisfaction.
- **DP advantage is most visible in $\alpha$**: While real Cost Ratio gains are modest over variants, DP is far superior in minimizing $\alpha$ (the computable guarantee).
- **Unbalanced forests are counter-examples**: On Names-US, where one giant component dominates, the exact MFC-OPT is already cheap, making the interpolation less valuable.

## Highlights & Insights
- **Observability of parameter quality**: The cost function $\to \alpha$ bound makes the quality of representative selection observable at runtime. This is more practical than "post-hoc" approximation ratios.
- **One knob to rule them all**: MultiRepMFC uses a budget knob to connect "fast but coarse" MFC-Approx with "accurate but $\Omega(n^2)$" exact algorithms.
- **Serendipitous sub-problem**: Selecting optimal representatives led to "shared-budget multi-instance k-center"—a novel generalization of k-center with independent value.
- **Stronger through simplification**: The new analysis provides tighter $2$/$2\gamma$ bounds and is more concise than the previous $2.62$/$2\gamma+1$ logic.

## Limitations & Future Work
- **Reliance on $\gamma$-overlap**: The Metric MST approximation is tied to $\gamma$; other quality parameters for predictions could be explored.
- **Breaking the bound of 2**: Whether sub-quadratic algorithms can achieve $<2$ worst-case approximation for MFC remains an open question.
- **Lack of universal lower bounds**: It is unclear if a lower bound on approximation ratios exists for all sub-quadratic algorithms.
- **Diminishing returns in unbalanced forests**: The method’s "sweet spot" is when components are relatively balanced ($t \approx \sqrt{n}$).

## Related Work & Insights
- **vs. Predecessor MFC-Approx (Veldt et al., 2025)**: This work generalizes the single-rep approach to multi-rep, tightens bounds to $2$/$2\gamma$ as a corollary, and adds the instance-specific $\alpha$ bound.
- **vs. Classic Metric MST**: Naive methods check $O(n^2)$ distances. This work follows the learning-augmented path to achieve sub-quadratic complexity with provable guarantees.
- **vs. Standard k-center (Gonzalez, 1985)**: BESTREPS generalizes k-center to multiple instances, reusing the 2-approximation greedy algorithm as a subroutine within a DP framework.

## Rating
- Novelty: ⭐⭐⭐⭐ (Multi-rep interpolation + instance bounds + multi-instance k-center is novel, though built on prior framework.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Evaluated on 4 metrics and 3 strategies; focused on algorithm validation.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Excellent progression from theorems to tightness constructions; concise and general analysis.)
- Value: ⭐⭐⭐⭐ (Provides tight bounds and runtime guarantees for large-scale metric MST; practical with clear open problems.)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Online Rounding and Learning Augmented Algorithms for Facility Location](online_rounding_and_learning_augmented_algorithms_for_facility_location.md)
- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](../../ICML2026/learning_theory/parsimonious_learning-augmented_online_metric_matching.md)
- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICLR 2026\] ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling](atlas_alibaba_dataset_and_benchmark_for_learning-augmented_scheduling.md)

</div>

<!-- RELATED:END -->
