---
title: >-
  [Paper Note] Guaranteed Optimal Compositional Explanations for Neurons
description: >-
  [ICML 2026][Neuron explanations] Compositional explanations typically use beam search to find the "logical formula that best aligns with neuron activations…
tags:
  - "ICML 2026"
  - "Neuron explanations"
  - "IoU decomposition"
  - "optimal compositional explanations"
  - "heuristic search"
  - "beam search"
date: 2026-05-08
content_hash: 1662a422e2b82220
---

# Guaranteed Optimal Compositional Explanations for Neurons

**Conference**: ICML 2026  
**arXiv**: [2511.20934](https://arxiv.org/abs/2511.20934)  
**Code**: Provided in the paper ("We release the code at the following repository", see the original text for the specific link)  
**Area**: Interpretability / Neuron Explanation / Compositional Explanations  
**Keywords**: Neuron explanations, IoU decomposition, optimal compositional explanations, heuristic search, beam search  

## TL;DR
Compositional explanations typically use beam search to find the "logical formula that best aligns with neuron activations," but beam search lacks optimality guarantees. This paper proposes a precise decomposition of IoU (dIoU) + an admissible heuristic + a best-first optimal algorithm. Within a runtime comparable to beam search, it **guarantees the global optimal solution for the first time**, revealing that 10–40% of explanations in past literature are actually suboptimal.

## Background & Motivation

**Background**: Compositional explanations (Mu & Andreas 2020) are a class of methods designed to characterize "which concepts a CNN neuron aligns with spatially." The output consists of propositional logic formulas such as `((Cat OR Car) AND White)`, and the alignment quality is quantified using IoU. Compared to early Network Dissection, which provides only a single concept label, this approach better reflects the true behavior of "polysemantic neurons" and serves as a pillar of mechanistic interpretability.

**Limitations of Prior Work**: The size of the full state space across the candidate concept set $L^1$ and formula length $n$ is $\sum_{k=1}^{n}n_o^{k-1}\prod_{i=0}^{k-1}(|L^1|-i)$. Under the standard settings of Mu & Andreas, this reaches $2.8\times 10^{14}$ operations, making exhaustive enumeration impossible. Previous methods relied on beam search with small widths and additional assumptions such as "distinct concepts" or "layer-wise incremental concatenation." The cost of beam search is the **lack of optimality guarantees**—the returned solution may not be truly optimal, and the gap to the optimal solution remains unknown. This has left the field in an awkward position for years: explanations look appealing, but it is unclear whether they represent reality or are merely artifacts of beam search.

**Key Challenge**: Massive state space prevents enumeration vs. beam search lacks optimality guarantees $\rightarrow$ Ground truth is unknown $\rightarrow$ Impossible to judge the approximation quality of existing algorithms or systematically develop better heuristics. Executing a direct BFS on medium-to-high complexity datasets would take $\sim 4\times 10^{8}$ hours, which is clearly infeasible.

**Goal**: (i) Define a set of **fundamental quantities** to decompose IoU into terms that can be independently estimated and combined via logical operators; (ii) Design an **admissible heuristic** providing a $[dIoU_{\min}, dIoU_{\max}]$ interval to prune the state space sufficiently; (iii) Construct an optimal algorithm with time complexity in the same order of magnitude as beam search.

**Key Insight**: The authors noted that compositional explanations only use three 0-preserving operators (OR, AND, AND NOT), and formulas can always be decomposed into "left sub-formula $\oplus$ right atomic concept" per Assumption 2. This implies that if IoU can be expressed as "local terms accumulated across samples $x$ that can be derived from sub-formula quantities to parent formulas," a classic A*-style optimal search can be performed.

**Core Idea**: Rewrite IoU as $dIoU=\frac{\sum_x|I^U(L)_x|+|I^C(L)_x|}{|^1N|+\sum_x|E^U(L)_x|+|E^C(L)_x|}$, where $I^{U/C}$ (unique/common intersection) and $E^{U/C}$ (unique/common extras) are decomposable terms partitioned by "whether a position is labeled by multiple concepts simultaneously." Based on this decomposition, min/max estimates are derived and integrated into a best-first search, implementing the first **optimal** compositional explanation algorithm.

## Method

### Overall Architecture
The method consists of three parts. The first part (Sec. 3.1) defines decomposable quantities: dataset positions $(x,j)$ are divided into **unique** $U$ and **common** $C$ categories based on how many concepts label them simultaneously. Neuron activations, intersections with concepts, and the "remaining labeled but non-active" portions are split by $U/C$ to obtain six fundamental quantities: $N^{U}, N^{C}, I^{U}(k), I^{C}(k), E^{U}(k), E^{C}(k)$. The second part (Sec. 3.2) provides the heuristic: based on a **Disjoint Matrix** $D$ that determines if the two sides of a sub-formula are disjoint in their labeling, min/max recursions for $I^C$ and $E^C$ are derived for OR/AND/AND NOT. Top-$n$/bottom-$n$ estimates for each concept across each quantity are used to estimate the "maximum/minimum gain brought by adding $n$ more concepts." The third part (Sec. 3.3) embeds the heuristic into a best-first search to achieve the optimal algorithm.

### Key Designs

1.  **IoU Precise Decomposition (dIoU) + Fundamental Quantities**:
    - **Function**: Rewrites the global metric $IoU(L,N,M)=|^1N\cap{}^1M_L|/|^1N\cup{}^1M_L|$ into a form that is **independently cumulative per sample and recursive from sub-formula quantities**, allowing pruning on formula prefixes.
    - **Mechanism**: The core distinction is whether a position $(x,j)$ belongs to $U$ (labeled by exactly one concept) or $C$ (labeled by $\geq 2$ concepts). For 0-preserving logic operators (OR, AND, AND NOT), the behavior of unique elements can be exactly derived via their truth tables (Observation 1): OR sums the unique counts of both sides, AND clears unique counts, and AND NOT equals the left side's unique count. Common elements require checking if sub-formulas are disjoint: when disjoint, they are calculated exactly like unique elements; when overlapping, only intervals can be provided. Equivalence is strictly guaranteed by Lemma 3.6: $dIoU=IoU$ if and only if all operators are 0-preserving, which covers all commonly used operators in the literature.
    - **Design Motivation**: Original IoU can only be calculated after the formula is **fully assembled**, preventing prefix pruning. By introducing the $U/C$ split, the "unique part" is exact and the "common part" can be bounded, making an admissible heuristic possible (a necessary condition is upper bound $\geq$ true value), which is the theoretical foundation of the optimal algorithm.

2.  **min/max Heuristic + Multi-step Path Estimation**:
    - **Function**: For any current prefix $L\in L^i$ ($i\leq n$), provides the $[dIoU_{\min}, dIoU_{\max}]$ achievable by appending up to $n-i$ atomic concepts along OR/AND/AND NOT "exclusive paths."
    - **Mechanism**: Two layers. **Layer 1 (Single-step estimation)**: Appending a new concept $k$ to the current prefix, using the Disjoint Matrix $D$ to distinguish disjoint/overlap cases and calculating min/max for $I^C/E^C$ via Eqs. (7)–(10); unique parts utilize Observation 1. **Layer 2 (Multi-step path estimation)**: Pre-calculating $\mathrm{Top}_k$ and $\mathrm{Bott}_k$ (sum of top/bottom $k$ concept quantities per sample) and using Eqs. (11)–(14) to provide final $|I_{\min}|, |I_{\max}|, |\mathrm{Union}_{\min}|, |\mathrm{Union}_{\max}|$ for the three exclusive paths. Non-exclusive paths between operators take max/min across paths (preserving admissibility). Finally, $dIoU_{\max}=\frac{\sum_x|I_{\max}(L)_x|}{\sum_x|\mathrm{Union}_{\min}(L)_x|}$, with $dIoU_{\min}$ being symmetric.
    - **Design Motivation**: Classic A* requires an admissible heuristic to guarantee optimality—here, it is sufficient that $dIoU_{\max}$ never falls below the true value. The authors also provide an "aggregated computation" version: per-sample quantities are summed before min/max operations, slightly reducing precision but calculating only once per prefix—far cheaper than sample-wise. The frontier uses aggregated estimates for coarse screening, upgrading to sample-wise when popped, and calculating exact values only at the end. This "tiered refinement" is key to achieving both heuristic affordability and search termination.

3.  **Best-first Optimal Search + Sub-label Backpropagation**:
    - **Function**: Maintains candidate prefixes in a max-heap frontier, popping the node with the highest current $dIoU_{\max}$. If it is an aggregated estimate, it is refined and re-inserted; if sample-wise, it is expanded (append atomic concept $\times$ operator) or evaluated for true value. A global $dIoU_{\min}^*$ is maintained to prune nodes where $dIoU_{\max} < dIoU_{\min}^*$.
    - **Mechanism**: Four key tricks make the search feasible: (a) **Frontier initialization** only includes seeds with $dIoU_{\max} >$ global $dIoU_{\min}^*$ to prevent immediate explosion; (b) **Aggregated $\rightarrow$ sample-wise $\rightarrow$ exact refinement** applies expensive calculations only to promising nodes; (c) **Sub-label backpropagation**: Exact quantities generated during path evaluation are stored; nodes in the search frontier sharing the same sub-formulas use these exact quantities to replace estimates, tightening bounds; (d) **Logical equivalence pruning + buffer** removes degenerated or redundant expressions like `cat AND NOT cat` or `A OR A`. Theoretically, since $dIoU_{\max}$ is an admissible upper bound and the algorithm exhausts all nodes with "upper bounds higher than the current best lower bound," the returned solution is globally optimal (proof in Appendix F).
    - **Design Motivation**: The fundamental flaw of beam search is its "inability to backtrack," potentially stacking explanations that rely on unverified scenarios to fix early errors (the paper cites `(ball_pit OR flower) AND NOT dining_room` as a counter-example: `ball_pit` and `dining_room` never co-occur, making the constraint invalid). Best-first + admissible heuristics fix this: if a higher true value solution exists, one of its prefixes must have a high $dIoU_{\max}$ and will never be pruned incorrectly.

### Loss & Training
No training—this is an analysis/explanation algorithm. Inputs are a pre-trained CNN, a concept-labeled dataset, and neuron activation ranges $[\tau_1, \tau_2]$; the output is the logical formula best aligned with the neuron.

## Key Experimental Results

### Main Results
The feasibility of the algorithm was evaluated across three levels of data complexity: Cityscapes (25 concepts, all disjoint, Low); Ade20K-Detectron2 (847 concepts, no overlap, Medium); Broden (1198 concepts, frequent overlap, High). 50 neurons from the final layer of ResNet were sampled for each, with activation ranges at the top 0.5%.

| Complexity | Algorithm | Visited | Expanded | Estimated | Sec/Unit |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Low (Cityscapes) | Optimal (Ours) | 1 | 101 | 778 | 0.08 |
| Low | Beam (Ours, Heuristic) | 6 | 14 | 639 | 0.17 |
| Low | MMESH beam | 121 | 15 | 697 | 10.37 |
| Low | Vanilla beam | 716 | 15 | – | 2.77 |
| Med (Ade20K-D2) | Optimal (Ours) | 1 | 4915 | 106 | 90.57 |
| Med | Beam (Ours, Heuristic) | 10 | 15 | 37956 | 11.55 |
| Med | MMESH beam | 39 | 15 | 37956 | 38.42 |
| Med | Vanilla beam | 37979 | 15 | – | 450 |
| High (Broden) | Optimal (Ours) | 47 | 105 | 108 | 5768 |
| High | Beam (Ours, Heuristic) | 27 | 15 | 53752 | 123.33 |
| High | MMESH beam | 43 | 15 | 53752 | 102.35 |
| High | Vanilla beam | 53775 | 15 | – | 5929 |

The **optimal algorithm** completed across all three complexity levels, with runtimes in the same order of magnitude as vanilla beam search (both $\sim$ 5800 s/unit at high complexity), vastly smaller than the BFS estimate of $4\times 10^8$ hours. **Beam search guided by this paper's heuristic** had the fewest visited nodes and wall-clock times comparable to or faster than MMESH—a single heuristic provides both an optimal algorithm and optimization for beam search.

### Ablation Study
**How poor are the solutions found by beam search?**

| Model | Suboptimal Ratio | Cat 1 (Diff Concepts + IoU) | Cat 2 (Same Concepts, Diff Logic $\rightarrow$ Diff IoU) | Cat 3 (Same IoU, Diff Logic) |
| :--- | ---: | ---: | ---: | ---: |
| ResNet | 9% | 76% | 6% | 17% |
| AlexNet | 23% | 93% | 5% | 2% |
| DenseNet | 39% | 73% | 0% | 27% |

At high complexity, 10–40% of beam search explanations deviate from the optimal. Cat 1 (most severe) suboptimal solutions mostly involve AND / AND NOT, indicating that beam search struggles with "polysemantic neurons requiring complex negation/intersection for precise characterization."

### Key Findings
- **Expanded nodes < 0.1% of state space**: The optimal algorithm's frontier expanded only 105 nodes under Broden's high complexity (vs. hundreds of millions in the state space), proving the pruning efficiency of sub-label backpropagation + min/max dual estimation; "exploring the entire space" is compressed into "exploring a sparse A* tree."
- **Aggregated estimation is key to feasibility**: The number of estimated nodes is much larger than expanded nodes (37,956 vs. 4,915 at medium complexity), which means many nodes are rejected immediately because their upper bounds are too low. This is the payoff of the tiered refinement strategy.
- **Beam variant of this heuristic is hyperparameter-insensitive**: As explanation length increases from 3 to 20 and beam from 5 to 20, runtime only drifts between 0.19–0.42 min/unit; MMESH jumps from 0.62 to 25 min/unit in the same setting. This means researchers no longer need to compromise between beam size and exploration depth.
- **DenseNet has the highest suboptimality rate (39%)**, with Cat 3 (same IoU, different expression) accounting for 27%—suggesting neurons formed by dense connections are particularly sensitive to the specific form of logical expressions, prompting beam search to produce "IoU-correct but semantically fragile" pseudo-solutions (such as the `AND NOT dining_room` example).

## Highlights & Insights
- **First admissible heuristic**: It is not that no one wanted to do optimal search in compositional explanations, but rather no one could formulate an affordable admissible upper bound. Once dIoU correctly isolated "exact unique parts" and "bounded common parts," A* became viable. This is a model for aligning "theoretical analyzability" with "algorithmic computability."
- **MMESH uses spatial info, transferability follows binary bits**: Unlike MMESH, this heuristic only uses binary occupancy information and can thus generalize to domains without spatial structures (NLP, tabular), which MMESH cannot.
- **Exposing the flaws of beam search**: The `(ball_pit OR flower) AND NOT dining_room` example demonstrates that beam search can construct explanations that are "semantically absurd but numerically attractive in IoU." The lack of co-occurrence between `ball_pit` and `dining_room` makes the `AND NOT` constraint uninformative. Such spurious explanations were invisible without ground truth.

## Limitations & Future Work
- Feasibility relies heavily on having a sufficient number of "unique elements." In NLP datasets (where labels are almost entirely overlapping), quantities degrade into large uncertainty intervals for common estimates, failing to collapse the state space. The authors noted this but provided no solution.
- At high complexity, the optimal algorithm takes ~96 min/unit—reproducible but not ideal for interactive use.
- When neurons are inherently uninterpretable (IoU < 0.04) or over-generalized, the frontier can explode; the paper suggests a two-stage "beam first, then optimal for interpretable units" strategy, but it is not integrated.
- Only validated on CNNs + visual concepts. Transformations for Transformer multi-token structures or vectorized concepts (e.g., Probe-derived) haven't been explored.
- Default max length is 3, which is usually sufficient but may not capture extremely complex polysemantic neurons; at greater lengths, top vectors dominate and search may degrade toward BFS.

## Related Work & Insights
- **vs. Mu & Andreas 2020 (vanilla beam)**: Uses the same formulas, operators, and assumptions, but provides the missing "optimality guarantee." Vanilla beam requires 37,979 visited nodes at medium complexity to match results, whereas the optimal search expands only 4,915 nodes.
- **vs. MMESH (La Rosa et al., 2023)**: MMESH is currently the strongest informed beam search, relying on spatial info. This heuristic uses only binary info but matches or beats its wall-clock time and is hyperparameter-insensitive. Both find the same solutions on ResNet (limited by the beam search space), but this paper reveals these solutions are suboptimal for 9% of units.
- **vs. Network Dissection (Bau et al., 2017)**: Network Dissection gives single-concept explanations; compositional explanations expand this to logic. This "optimal" framework can audit whether Network Dissection’s explanations are truly optimal for single concepts.
- **vs. Probabilistic IoU Approximation (Nowozin 2014; Li et al. 2013)**: Those works use expected IoU approximations for structured prediction; this paper performs deterministic precise decomposition suited for search. Both share the spirit of rewriting non-decomposable global metrics into manageable local terms.
- **Insights for Mechanistic Interpretability**: Compositional explanations finally have a ground-truth benchmark. This means future developments of new heuristics or algorithms (e.g., for transformer neurons or SAE feature explanations) have a gold standard for comparison; the field can now discuss "approximation ratios" and "suboptimality rates" as in optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First optimality guarantee for compositional explanations; dIoU decomposition + admissible heuristic is a clean theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three complexity tiers + three backbones + comprehensive comparison with beam search; detailed analysis, though restricted to visual CNNs.
- Writing Quality: ⭐⭐⭐⭐ Numerous formulas but a clear hierarchy; Definition/Observation/Lemma structure is rigorous; Fig. 1 visualization is very helpful.
- Value: ⭐⭐⭐⭐⭐ Provides a ground-truth metric for the entire compositional explanation direction, reassesses conclusions in previous literature, and lays the foundation for future algorithmic research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Regularization for Performative Learning](optimal_regularization_for_performative_learning.md)
- [\[AAAI 2026\] Formal Abductive Latent Explanations for Prototype-Based Networks](../../AAAI2026/others/formal_abductive_latent_explanations_for_prototype-based_networks.md)
- [\[ICCV 2025\] On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations](../../ICCV2025/others/on_the_complexity-faithfulness_trade-off_of_gradient-based_explanations.md)
- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](../../ICLR2026/others/compositional_diffusion_long_horizon_planning.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](../../CVPR2026/others/simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)

</div>

<!-- RELATED:END -->
