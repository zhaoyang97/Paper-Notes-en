---
title: >-
  [Paper Note] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation
description: >-
  [AAAI 2026][over-squashing] From an information-theoretic perspective, this paper models spectral GNNs as communication channels and proposes the Channel Capacity Constrained Estimation (C3E) framework, which formalizes the selection of GNN hidden dimensions and depth as a nonlinear programming problem. The framework estimates optimal architectural parameters prior to training, effectively mitigating over-squashing and consistently improving representation learning across 9 datasets.
tags:
  - AAAI 2026
  - over-squashing
  - graph neural networks
  - channel capacity
  - information theory
  - hidden dimension estimation
date: 2026-05-08
content_hash: 1800ca65343b595e
---

# How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation

**Conference**: AAAI 2026
**arXiv**: [2511.06443](https://arxiv.org/abs/2511.06443)
**Code**: [GitHub](https://github.com/pixelhero98/C3E)
**Area**: Others
**Keywords**: over-squashing, graph neural networks, channel capacity, information theory, hidden dimension estimation

## TL;DR

From an information-theoretic perspective, this paper models spectral GNNs as communication channels and proposes the Channel Capacity Constrained Estimation (C3E) framework, which formalizes the selection of GNN hidden dimensions and depth as a nonlinear programming problem. The framework estimates optimal architectural parameters prior to training, effectively mitigating over-squashing and consistently improving representation learning across 9 datasets.

## Background & Motivation

Graph neural networks (GNNs) achieve strong performance on various graph learning tasks, but suffer from severe performance degradation as propagation depth increases. This degradation is typically attributed to two phenomena: **over-smoothing** (node representations converge to identical values) and **over-squashing** (information is compressed into fixed-size vectors, causing catastrophic information loss).

Over-smoothing has been effectively addressed by normalization and residual techniques, whereas the theoretical understanding of over-squashing remains limited. Existing approaches primarily focus on modifying graph structure (e.g., graph rewiring) or node features; however, a growing body of theoretical work demonstrates that **the network architecture itself—hidden dimension and depth—is the critical factor**. Loukas (2020) showed that GNNs can propagate information effectively only when the product of depth and hidden dimension exceeds a polynomial in the graph size; Di Giovanni (2023) further proved that larger hidden dimensions alleviate over-squashing.

Despite theoretical knowledge that specific hidden dimensions and depths can mitigate over-squashing, **no existing method can estimate these optimal parameters prior to training**. In practice, GNN hidden dimensions and depths are chosen entirely by heuristics (e.g., 16/32/64/128/256), often leading to suboptimal results or severe information loss. The core idea of this paper is to model spectral GNNs as communication channels, leverage Shannon's theorem and the maximum entropy principle, and reformulate optimal architectural parameter selection as a nonlinear programming problem with well-defined constraints.

## Method

### Overall Architecture

The C3E framework consists of three core steps: (1) deriving a theoretical channel capacity expression for spectral GNNs modeled as communication channels; (2) analyzing the relationship between information compression and over-squashing by introducing the representation compression ratio as a metric; (3) constraining the effective channel capacity to a reasonable range and constructing a nonlinear programming problem to solve for optimal hidden dimension and depth. The entire process is completed **prior to training**, requiring only the graph structure and propagation matrix.

### Key Designs

1. **Theoretical Channel Capacity of Spectral GNNs (Theorem 1)**:

    - Function: Derives a theoretical upper bound on the amount of information a spectral GNN can propagate.
    - Mechanism: The layer-wise representation learning of a spectral GNN is $\mathbf{H}_l = \Delta(\mathbf{S}_l \mathbf{H}_{l-1} \mathbf{W}_l)$, where $\mathbf{S}_l$ is the propagation matrix. Assuming a perfect encoder mapping such that the conditional entropy $H(\mathcal{G}'|f(\mathcal{G})) = 0$, and by the maximum entropy principle, the channel capacity is: $\phi = \max\left[\frac{1}{2}\ln(2\pi e) + \frac{1}{2}\sum_{l=1}^{L}\ln(n w_{l-1} \sigma^2_{\mathbf{S}_l})\right]$
    - Design Motivation: The formula directly relates hidden dimension $w_l$, depth $L$, graph structure (via propagation matrix variance $\sigma^2_{\mathbf{S}_l}$), and channel capacity. Small hidden dimensions cause channel capacity to decrease with depth, resulting in information loss.

2. **Representation Compression Ratio and Information Compression Analysis (Corollary 2)**:

    - Function: Quantifies the degree of information compression and reveals the dual role of width and depth.
    - Mechanism: The representation compression ratio is defined as $\theta = \phi / \bar{w}$, where $\bar{w} = (\prod_{l=1}^{L} w_l)^{1/L}$ is the geometric mean of hidden dimensions. Partial derivative analysis of $\theta$ reveals: (a) for fixed depth, $\theta$ attains its maximum at a threshold $\bar{w}^*$, and increasing width reduces compression; (b) for fixed width, the effect of depth depends on the comparison between $\ln(\bar{w})$ and $-\bar{K}$—deepening a wide network exacerbates compression ($\theta$ increases), while deepening a narrow network leads to information vanishing ($\phi$ decreases).
    - Design Motivation: Reveals that over-squashing is fundamentally **cumulative information compression in the representation matrix**, not merely a consequence of graph structure. It also explains why naively increasing depth or width is not always effective.

3. **Effective Channel Capacity and Constrained Optimization (C3E Formulation)**:

    - Function: Accounts for inter-layer architectural bottlenecks to construct a practically applicable nonlinear programming problem.
    - Mechanism: An effective channel capacity $\phi_0$ is introduced (capturing the information bottleneck effect due to width differences between adjacent layers), and Shannon's theorem is used to establish the constraint $\ln(n) \leq \phi_0 \leq \frac{1}{\eta}\ln(n)$, where $\eta \in (0,1]$ is a regularization hyperparameter. The resulting optimization problem is: $\max_{\mathbf{w}^{(L)}, L} \phi$, subject to $\bar{w} > \bar{w}^*$ (preventing over-squashing), $\ln(\bar{w}) > -\bar{K}$ (preventing information vanishing), and the channel capacity range constraint.
    - Design Motivation: Avoids trivial solutions of unboundedly increasing width or depth. In $\phi_0$, the numerator captures the architectural bottleneck (width differences between adjacent layers) and the denominator reflects the cumulative attenuation effect. The SLSQP solver yields solutions within seconds.

### Loss & Training

C3E is a pre-training architectural design method and does not involve a training loss. The propagation matrix variance $\sigma^2_{\mathbf{S}_l}$ is computed via population variance through sparse pre-computation before training. Hidden dimensions are obtained as continuous values and then rounded to integers. Solving time ranges from 3.7 seconds (Cora) to 879.6 seconds (ogbn-papers100M), far below the 2.45–120 hours required by trial-and-error methods.

## Key Experimental Results

### Main Results

Evaluation on 9 datasets across 8 spectral GNNs; ⋆ denotes architectures with parameters estimated by C3E:

| Model | Cora | Citeseer | Chameleon | Squirrel | ogbn-arxiv | ogbn-papers100M |
|-------|------|----------|-----------|----------|------------|-----------------|
| GCN | 0.808 | 0.707 | 0.381 | 0.311 | 0.714 | 0.733 |
| GCN⋆ | **0.837** | **0.723** | **0.432** | **0.346** | **0.729** | **0.760** |
| S2GC | 0.829 | 0.718 | 0.398 | 0.312 | 0.707 | 0.715 |
| S2GC⋆ | **0.841** | 0.724 | **0.435** | **0.352** | **0.726** | **0.753** |
| JacobiConv | 0.827 | 0.722 | 0.423 | 0.328 | 0.718 | 0.722 |
| JacobiConv⋆ | **0.841** | **0.729** | **0.469** | 0.351 | **0.730** | **0.759** |

Across all 8 models and 9 datasets, C3E-estimated variants achieve superior average performance, with most results being statistically significant (t-test, $p < 0.05$).

### Ablation Study

Relationship between representation compression ratio and model performance (Citeseer dataset, GCN):

| Depth $L$ | Baseline $\bar{w}$ | Baseline $\theta$ | Baseline Acc. | C3E $\bar{w}$ | C3E $\theta$ | C3E Acc. |
|-----------|-------------------|-------------------|---------------|---------------|-------------|---------|
| 1 | 16 | 0.558 | 0.707 | 32765 | 0.000 | - |
| 2 | 16 | 0.856 | 0.663 | 3960.89 | 0.004 | 0.716 |
| 4 | 16 | 1.450 | 0.612 | 3203.64 | 0.010 | **0.723** |
| 7 | 16 | 2.347 | 0.550 | 2399.07 | 0.021 | 0.714 |

### Key Findings

- The essence of over-squashing is **cumulative information compression in the representation matrix**: as depth increases, $\theta$ increases monotonically and model performance degrades consistently.
- Increasing hidden dimension effectively reduces $\theta$ and mitigates information compression; C3E-estimated models maintain extremely low $\theta$ (0.004–0.021 vs. baseline 0.558–2.347).
- Models with heuristically chosen dimensions (green points) scatter across the parameter space at suboptimal locations, while C3E-estimated solutions (red points) consistently fall in high-performance regions.
- The representational entropy of naively stacked deep models rapidly collapses to near zero, whereas C3E models maintain high entropy across all layers.
- Depth plays a dual role: beneficial for low-compression models (helping compress high-dimensional signals) and detrimental for high-compression models (exacerbating information loss).

## Highlights & Insights

- This is the first work to transform GNN hidden dimension × depth selection into a **theoretically grounded automated process** from an information-theoretic perspective, replacing trial-and-error.
- The representation compression ratio $\theta$ is a concise and powerful diagnostic tool: it converts over-squashing from a difficult-to-measure phenomenon into a computable scalar.
- The derivation of effective channel capacity $\phi_0$ is elegant: the numerator captures inter-layer bottlenecks and the denominator reflects information attenuation, jointly characterizing actual information flow.
- The method is highly practical: second-level solve time, no modification of propagation mechanisms or graph structure, and plug-and-play compatibility with any spectral GNN.
- The analysis in Corollary 2 reveals a previously unclearly articulated insight: deep-and-wide GNNs may over-squeeze due to cumulative compression, while deep-and-narrow GNNs fail due to information vanishing—two distinct failure modes requiring different remediation strategies.

## Limitations & Future Work

- The theory is based on the maximum entropy principle, which may overestimate the actual channel capacity of practical networks, providing an upper bound rather than a tight bound.
- The framework currently applies only to spectral GNNs (where propagation can be folded into a matrix operator) and has not been extended to spatial GNNs or Graph Transformers.
- While improvements are observed on heterophilic graphs (Chameleon, Squirrel), the gains are smaller than on homophilic graphs, indicating room for improvement in the framework's handling of heterophily.
- Although the $\eta$ hyperparameter is robust, it still requires tuning; automatic selection of optimal $\eta$ is a direction for future work.
- The element-wise definition of matrix entropy ignores higher-order correlations among entries, potentially underestimating the actual information content.

## Related Work & Insights

- Orthogonal to graph rewiring methods: C3E mitigates over-squashing from an architectural perspective and can be combined with graph rewiring approaches.
- The information bottleneck principle in DNNs (Tishby 2000) inspired the channel-based perspective of this paper; this work goes further by providing an actionable architectural selection method.
- Sun et al. (2021) applied the maximum entropy principle to automatically design CNN architectures; this paper extends that idea to GNNs and directly addresses over-squashing.
- Implications for NAS (neural architecture search): the information-theoretic perspective can serve as a prior constraint for search, substantially reducing the search space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[AAAI 2026\] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)
- [\[ICLR 2026\] The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](../../ICLR2026/others/the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)

</div>

<!-- RELATED:END -->
