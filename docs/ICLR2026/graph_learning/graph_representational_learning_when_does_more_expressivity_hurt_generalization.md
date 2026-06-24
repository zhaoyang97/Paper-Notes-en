---
title: >-
  [Paper Note] Graph Representational Learning: When Does More Expressivity Hurt Generalization?
description: >-
  [ICLR 2026][Graph Learning][GNN Expressivity] This paper proposes a family of pseudometrics $\zeta$-TMD parameterized by graph invariants and derives a data-dependent generalization bound based on "train-test graph structural similarity + model complexity + training set size." It theoretically explains "when higher expressivity GNNs generalize worse"—only when the added expressivity aligns with the structure-label correlation of the task is it beneficial; otherwise…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "GNN Expressivity"
  - "Generalization Bounds"
  - "Tree Mover's Distance"
  - "Pseudometric"
  - "PAC-Bayes"
  - "Structure-Label Correlation"
date: 2026-05-08
content_hash: 7d8844913956fcf6
---

# Graph Representational Learning: When Does More Expressivity Hurt Generalization?

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=C6vpifaZvU](https://openreview.net/forum?id=C6vpifaZvU)  
**Code**: Open source (GitHub, see original paper for link)  
**Area**: Graph Representational Learning / GNN Generalization Theory  
**Keywords**: GNN Expressivity, Generalization Bounds, Tree Mover's Distance, Pseudometric, PAC-Bayes, Structure-Label Correlation  

## TL;DR
This paper proposes a family of pseudometrics $\zeta$-TMD parameterized by graph invariants and derives a data-dependent generalization bound based on "train-test graph structural similarity + model complexity + training set size." It theoretically explains "when higher expressivity GNNs generalize worse"—only when the added expressivity aligns with the structure-label correlation of the task is it beneficial; otherwise, it merely increases complexity and degrades generalization.

## Background & Motivation
**Background**: A major line of GNN research focuses on continuously increasing the "expressivity" of models, i.e., the ability to distinguish non-isomorphic graphs, typically measured by the Weisfeiler-Lehman (WL) hierarchy. From standard message-passing networks (MPNN, limited to 1-WL) to k-GNNs, subgraph GNNs, and high-order folklore GNNs, expressivity has been pushed progressively higher.

**Limitations of Prior Work**: The relationship between expressivity and actual predictive performance remains unclear. On one hand, stronger models indeed perform better on many tasks, even if they do not distinguish more non-isomorphic graphs on those specific benchmarks. On the other hand, Bechler-Speicher et al. found that models discarding graph structure entirely (weaker than 1-WL) can outperform carefully designed GNNs on certain tasks. Classical complexity measures (VC dimension, Rademacher complexity) fail to explain this phenomenon in the over-parameterized regime—networks can memorize arbitrary labels, but they only generalize when there is real correlation between data and labels.

**Key Challenge**: Expressivity itself is neither absolutely harmful nor absolutely beneficial. The authors observed in a synthetic cycle-counting task that F4-MPNN (MPNN enhanced with cycle counts), which has moderate expressivity, generalizes best, while the strongest LF-GNN overfits. On BZR/MUTAGENICITY/NCI109, even with added label noise, training loss converges to zero, but test error spikes once the graph-label correlation is destroyed.

**Goal**: To answer two key questions—(1) When is increasing expressivity beneficial, and when is it harmful? (2) If a stronger GNN perform better, is it truly due to its increased expressivity in terms of graph distinguishability?

**Core Idea (Task-Aligned Pseudometrics + Data-Dependent Bound)**: Formalize the empirical intuition that "generalization depends on structure-label correlation." Introduce a family of **$\zeta$-TMD pseudometrics**, each parameterized by a graph invariant (degree distribution, k-WL colors, motif counts, etc.) to characterize expressivity at a specific level. By assuming labels are strongly correlated with a fixed $\zeta$-TMD, the generalization bound naturally decomposes into a "model complexity term" and a "train-test structural similarity term," precisely characterizing the trade-offs between expressivity and generalization.

## Method

### Overall Architecture
The framework consists of three steps: first, unify "graph invariants/color refinement algorithms" as CRA that are strongly simulatable by 1-WL, and use this to generalize Tree Mover's Distance (TMD) into **$\zeta$-TMD** pseudometrics; second, prove that the corresponding $\zeta$-MPNN is Lipschitz continuous with respect to this pseudometric; finally, use the PAC-Bayes mechanism to translate Lipschitzness into a data-dependent generalization bound and analyze when "expressivity exceeding task requirements" raises the bound.

```mermaid
flowchart TD
    A[Graph Invariants / CRA ζ<br/>Degree Dist·k-WL·Motif Counts] --> B[Strong Simulation Transform Rζ<br/>Simulate ζ via 1-WL]
    B --> C[ζ-TMD Pseudometric<br/>TMD on Rζ(G),Rζ(H)]
    C --> D[ζ-MPNN w.r.t. ζ-TMD<br/>Lipschitz Continuous Thm3.4]
    D --> E[PAC-Bayes Bound Thm4.1<br/>Complexity Term + Structural Similarity ξζ]
    E --> F[When is Expressivity Harmful? Thm5.1<br/>Comparison of F'⊊F⊊F̃]
```

### Key Designs

**1. $\zeta$-TMD: Embedding Expressivity into Comparable Pseudometrics.** Standard TMD is the Wasserstein distance between the distributions of rooted trees of depth $t$ for two graphs: $\mathrm{TMD}_t(G,H)=\mathrm{Wasserstein}(\mathcal{T}^t(G),\mathcal{T}^t(H))$, which only characterizes 1-WL expressivity. The authors observed that many stronger architectures (k-GNNs, subgraph GNNs, motif-enhanced GNNs) can be strongly simulated via "input transformation $R_\zeta$ + standard message passing" (Jogl et al. 2024). Running $t$ steps of CRA $\zeta$ on graph $G$ is equivalent to running $t$ steps of 1-WL on the transformed graph $R_\zeta(G)$. Thus, define $\zeta\text{-TMD}_t(G,H):=\mathrm{TMD}_t(R_\zeta(G),R_\zeta(H))$. The elegance of this construction is that it is a valid pseudometric (Prop 3.2), and if $G,H$ are distinguished by $\zeta$ within $T$ steps, then $\zeta\text{-TMD}_{T+1}>0$ (Prop 3.3). This translates "discrete distinguishability" into "continuous structural similarity," allowing different expressivity levels to be measured on a unified scale.

**2. Lipschitz Continuity of $\zeta$-MPNN.** This is the bridge connecting expressivity and generalization. Theorem 3.4 proves: if a $\zeta$-MPNN has $T$ layers, message/update function Lipschitz constants are bounded by $L_{g^{(t)}},L_{f^{(t)}}$, followed by global sum pooling and a Lipschitz classifier, then $\|h(G)-h(H)\|\le L\cdot \zeta\text{-TMD}_{T+1}(G,H)$, where $L=L_c 2^T\prod_{t=1}^T L_{f^{(t)}}L_{g^{(t)}}$. Intuitively: structurally similar graphs (small $\zeta$-TMD) must be mapped to similar representations. For F-MPNN (node features enhanced by motif/cycle counts), Corollary 3.5 provides $\|h(G)-h(H)\|\le L\cdot F\text{-TMD}_{T+1}(G,H)$.

**3. Data-Dependent Generalization Bound and Structure-Label Alignment.** The key assumption is that "labels are strongly correlated with a specific pseudometric": for each class $k$, there exists a Lipschitz function $\eta_k$ such that $\Pr(y_G=k\mid G)=\eta_k(G)$, denoted as $y\sim pm$. Define the train-test structural distance $\xi_{pm}=pm(G_{te},G_{tr})=\max_{G\in G_{te}}\min_{H\in G_{tr}}pm(G,H)$. Under $y\sim\zeta\text{-TMD}_{T+1}$, Theorem 4.1 uses PAC-Bayes to give the bound:
$$L^0_{te}(\tilde h)\le \hat L^\gamma_{tr}(\tilde h)+O\Big(\underbrace{\tfrac{\mathrm{MC}(\mathcal C\circ\mathcal E_\zeta)}{N_{tr}^{2\alpha}\gamma^{1/D}\delta}}_{\text{Complexity Term}}+\underbrace{C\,\xi_\zeta}_{\text{Structural Similarity Term}}\Big).$$
The complexity term $\mathrm{MC}$ aggregates weight spectral norms, hidden dimension $b$, and maximum degree $d$. Structural similarity $\xi_\zeta$ measures the alignment of train-test graphs under $\zeta$-TMD. This decomposition is the core insight: good generalization occurs when the model maps the "structural similarities relevant to the labels" into representation similarities while controlling capacity.

**4. Conditions for Negative Effects of Expressivity.** Let labels be strongly correlated with $F\text{-TMD}_{T+1}$, and consider three nested expressivity levels $\mathcal F'\subsetneq\mathcal F\subsetneq\tilde{\mathcal F}$. Theorem 5.1 shows that moving from $\mathcal F'$ to $\mathcal F$ (which precisely captures task-relevant structure) keeps the bound stable as it is controlled by the same $\xi_F$. However, increasing expressivity to $\tilde{\mathcal F}$ (capturing extra task-irrelevant structure) increases structural discrepancy ($\xi_{\tilde F}\ge \xi_F$, Lemma G.1), significantly raising the generalization bound. The conclusion is clear: **the optimal GNN expressivity should exactly match the level required to capture task-relevant structure-label correlation—no more, no less.**

## Key Experimental Results

### Main Results (Task 1: Synthetic graphs classified by median "3-cycle + 4-cycle counts")

| Model | Expressivity | Test Accuracy |
|-------|--------------|---------------|
| MPNN | 1-WL | 0.8490 ± 0.0045 |
| F3-MPNN | +3-cycle | 0.8657 ± 0.0085 |
| **F4-MPNN** | +4-cycle (Task Aligned) | **0.9793 ± 0.0068** |
| F7-MPNN | +Long cycles | 0.9660 ± 0.0065 |
| Sub-G (Subgraph GNN) | Stronger | 0.8623 ± 0.0058 |
| L-G (Local 2-GNN) | Stronger | 0.8543 ± 0.0063 |
| LF-G (Local Folklore 2-GNN) | Strongest | 0.8450 ± 0.0135 |

All models achieve train accuracy >0.99, but F4-MPNN, which aligns with the task (4-cycles), has the highest test accuracy. Stronger models like LF-G perform worse, directly validating Theorem 5.1.

### Ablation Study (Task 2: TUDatasets, Accuracy vs. TMD Distance to Train Set)

| Dataset | Phenomenon | Related Theory |
|---------|------------|----------------|
| BZR | Accuracy decreases monotonically as TMD increases | $\xi_\zeta$ term in Theorem 4.1 |
| MUTAGENICITY | Theory curve closely fits empirical generalization error | Bound ≈ Measured Gap |
| NCI109 | Same as above | Same as above |

Label noise experiments: training loss still converges to 0, but test error spikes after graph-label correlation is destroyed, showing that memorization capacity ≠ generalization capacity.

### Key Findings
- **Task Alignment > Absolute Expressivity**: F4-MPNN (exactly capturing 4-cycles) beats all stronger models.
- **Structural Distance Governs Generalization**: Accuracy decreases as test graphs move further from the training set (higher TMD), consistent across datasets.
- **Tight Bounds**: Compared to standard PAC-Bayes bounds (magnitude $\approx 10^{16}$), this bound is at $10^4$, tighter by over 10 orders of magnitude and aligning with empirical gap curves.

## Highlights & Insights
- **Translating Discrete Expressivity to Continuous Pseudometrics**: $\zeta$-TMD allows different expressivity levels (1-WL, k-WL, motifs) to be quantified on the same scale for the first time.
- **Instructive Two-Term Decomposition**: The complexity vs. structural similarity decomposition explains the "double-edged sword" of expressivity—increasing it can either decrease $\xi_\zeta$ (beneficial) or increase it (harmful).
- **Actionable Design Principles**: Optimal Expressivity = Just enough to capture task correlation, echoing findings that TMD-based data augmentation improves generalization (Southern et al. 2025).
- **Theoretical-Empirical Loop**: Synthetic tasks (controlled correlation), real data (TMD-accuracy curves), and label noise experiments cross-validate each other.

## Limitations & Future Work
- **Bounds not tight enough for model comparison**: Different models correspond to different TMDs; the bound is a qualitative guide rather than a precise ranking tool.
- **Relevant pseudometrics are unknown/expensive**: Identifying which $\zeta$-TMD labels correlate with is difficult, and TMD (Wasserstein) is computationally expensive.
- **Strong Assumptions**: Relies on Lipschitz modeling of structure-label correlations, which real data may not strictly satisfy.
- **Future Work**: Explore graph augmentation techniques (synthetic graphs) that improve train-test structural similarity to systematically tighten the bound.

## Related Work & Insights
- **Expressivity**: 1-WL upper bound (Xu 2019), k-GNN (Morris 2020), Subgraph GNN (Frasca 2022), and the unified "input transform + standard message passing" view (Jogl et al. 2024).
- **Generalization**: VC dimensions (Scarselli 2018), Rademacher (Garg 2020), PAC-Bayes (Liao 2021). Closest are Ma et al. 2021 (feature alignment but node-level only) and Vasileiou et al. 2025 (Resilience/Tree distance but limited to standard MPNN).
- **Insights**: This work uses "pseudometrics + structure-label correlation" as the core tool for GNN analysis, suggesting that future work should return to the first principle of "task alignment" rather than blindly chasing WL levels.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Connects expressivity and generalization via a unified $\zeta$-TMD framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic and 6 real datasets with label noise and bound comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, though dense theoretical notation may be a barrier.
- **Value**: ⭐⭐⭐⭐⭐ Answers a long-standing open question in the GNN community with actionable design principles.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Trade-off Between Expressivity and Privacy in Graph Representation Learning](on_the_trade-off_between_expressivity_and_privacy_in_graph_representation_learni.md)
- [\[ICLR 2026\] Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion](glance_for_context_learning_when_to_leverage_llms_for_node-aware_gnn-llm_fusion.md)
- [\[ICLR 2026\] Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization](adaptive_mixture_of_disentangled_experts_for_dynamic_graph_out-of-distribution_g.md)
- [\[ICLR 2026\] GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](../../ICML2026/graph_learning/when_do_graph_foundation_models_transfer_a_data-centric_theory.md)

</div>

<!-- RELATED:END -->
