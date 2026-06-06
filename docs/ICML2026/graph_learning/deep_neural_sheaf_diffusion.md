---
title: >-
  [Paper Note] Deep Neural Sheaf Diffusion
description: >-
  [ICML2026][Graph Learning][Neural Sheaf Diffusion] This paper identifies that Neural Sheaf Diffusion (NSD) loses its theoretical anti-collapse capability in deep layers because the "disagreement signal" of the sheaf Lapl…
tags:
  - "ICML2026"
  - "Graph Learning"
  - "Neural Sheaf Diffusion"
  - "Deep GNN"
  - "Sheaf Adjacency Operator"
  - "Oversmoothing"
  - "Graph Foundation Models"
date: 2026-05-08
content_hash: 40647f2cff598155
---

# Deep Neural Sheaf Diffusion

**Conference**: ICML2026  
**arXiv**: [2605.19021](https://arxiv.org/abs/2605.19021)  
**Code**: https://github.com/remibourgerie/deep-neural-sheaf-diffusion  
**Area**: graph_learning  
**Keywords**: Neural Sheaf Diffusion, Deep GNN, Sheaf Adjacency Operator, Oversmoothing, Graph Foundation Models

## TL;DR
This paper identifies that Neural Sheaf Diffusion (NSD) loses its theoretical anti-collapse capability in deep layers because the "disagreement signal" of the sheaf Laplacian vanishes as diffusion converges. DNSD replaces the Laplacian with a **sheaf adjacency operator** and incorporates LayerNorm, odd activations, and per-stalk gating. This allows sheaf architectures to be stably stacked up to 16 layers for the first time, achieving up to 30 pp improvement over GNN/NSD baselines on synthetic long-range tasks and consistent leads on real-world heterophilic benchmarks.

## Background & Motivation

**Background**: Standard GNNs (GCN, GAT, etc.) pass information through layer-wise "neighbor weighted averaging." Theoretically, more layers increase the receptive field; however, in practice, deep GNNs generally fail to train. Literature summarizes this predicament as *oversmoothing* (convergence of node representations) and *oversquashing* (compression of long-distance signals). The root cause is that message passing is essentially a "convex combination," where repeated iterations inevitably level out differences.

**Limitations of Prior Work**: Bodnar et al. (2022) proposed Neural Sheaf Diffusion (NSD), using **cellular sheaves** to assign a learnable linear mapping $\mathcal{F}_{v\trianglelefteq e}$ to each edge. This constructs a sheaf Laplacian $\Delta_\mathcal{F}$ to replace the standard graph Laplacian. They theoretically proved that under suitable restriction maps, the steady state of sheaf diffusion can separate almost any label configuration, thus **not** collapsing due to depth. However, this paper empirically finds that this guarantee **fails in practice**—NSD performance still collapses as layers get deeper.

**Key Challenge**: The essence of the sheaf Laplacian is "measuring the disagreement between adjacent stalks," i.e., $\Delta_\mathcal{F}\mathbf{X}$ measures "how much has not yet aligned." The goal of the diffusion process is precisely to eliminate disagreement. Consequently, $\Delta_\mathcal{F}\mathbf{X}$ **monotonically tends toward 0** as the number of layers increases. Deep networks are forced to update on increasingly faint residuals, making the loss almost insensitive to deep parameters. Combined with the asymmetric truncation of ReLU, cross-layer scale drift, and uniform propagation of noise components, a significant gap opens between theory and practice.

**Goal**: To translate the "theoretical depth" of NSD into "practical depth," making sheaf architectures a stackable backbone for graph foundation models.

**Key Insight**: Since the problem lies in "using disagreement to drive updates," the update operator should be switched to "using dependency to drive updates"—specifically, replacing the sheaf Laplacian $\Delta_\mathcal{F}$ with the sheaf **adjacency** operator $A_\mathcal{F}$. This sheaf convolutional operator actually appeared in the original derivation by Bodnar et al. (their Eq. 4), but in the final architecture, they only applied non-linearity to the Laplacian term and discarded the identity component, thereby introducing the side effect of "signals vanishing with depth." DNSD restores this discarded path and stacks standard components for deep training (LayerNorm + odd activation + gating) to form a coherent recipe for "depth-available sheaf networks."

**Core Idea**: Change sheaf diffusion from "subtracting a vanishing disagreement" to "aggregating matrix-valued dependency," using LayerNorm/odd activation/per-stalk gating to stabilize deep dynamics.

## Method

### Overall Architecture
The input consists of a standard graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ and node features. The model constructs a set of $d\times f$ stalk representations $\mathbf{X}_v^{(l)}$ for each node. Each layer performs three tasks: (a) **learning** the restriction maps $\mathcal{F}^{(l)}_{v\trianglelefteq e}$ for that layer from endpoint node representations to obtain the current sheaf operator; (b) aggregating neighbors using the sheaf **adjacency** operator $A_\mathcal{F}^{(l)}$ instead of the Laplacian; (c) applying LayerNorm after residual addition and selectively filtering aggregated components via a gate $\mathbf{G}^{(l)}$. The final update rule is:

$\mathbf{X}^{(l+1)} = \mathrm{LN}\!\big((1+\epsilon^{(l)})\mathbf{X}^{(l)} - (\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot \sigma_{\mathrm{odd}}(A_\mathcal{F}^{(l)}\mathbf{X}^{(l)} W_1^{(l)}) W_2^{(l)}\big)$

The output is projected back into the task space for node classification. Compared to the original NSD update (which used $\Delta_\mathcal{F}$, ReLU, no LayerNorm, and no gating), DNSD replaces all four elements. The **adjacency replacement** is validated as the most critical factor.

### Key Designs

1.  **Sheaf Adjacency Operator Instead of Laplacian**:
    - **Function**: Replaces the aggregation operator in each layer from $\Delta_\mathcal{F}=D_\mathcal{F}^{-1/2} L_\mathcal{F} D_\mathcal{F}^{-1/2}$ to sheaf adjacency $A_\mathcal{F}$, whose block entries are $(A_\mathcal{F})_{uv}=\mathcal{F}_{u\trianglelefteq e}^\top \mathcal{F}_{v\trianglelefteq e}$.
    - **Mechanism**: The Laplacian measures "how much component is not yet aligned between neighbors," which diffusion pushes toward 0. Thus, $\sigma(\Delta_\mathcal{F}\mathbf{X} W_1)W_2$ in deep layers is equivalent to feeding "near-zero signals" to the non-linearity, causing deep parameters to receive almost no gradient. With the adjacency operator, the update term becomes $\sigma(A_\mathcal{F}\mathbf{X} W_1)W_2$, i.e., "aggregating overall neighbor representations using matrix-valued edge functions"—a "dependency signal" that does not vanish with diffusion and remains informative during both initialization and deep stages.
    - **Design Motivation**: Directly fixes the root cause of the "theoretical guarantee ↔ practical collapse" gap. Simultaneously, the authors explain this from a graph attention perspective in Section 4: GAT is also adjacency-based but uses scalar softmax; DNSD effectively "replaces scalar attention scores in GAT with matrix-valued edge mappings and moves normalization from attention scores to node representations."

2.  **Stabilization via LayerNorm + Odd Activation**:
    - **Function**: Uses row-wise LayerNorm to normalize each stalk $\tilde{\mathbf{X}}_u^{(l)}\in\mathbb{R}^{d\times f}$ across the feature dimension $f$ ($\mu_u,\sigma_u\in\mathbb{R}^d$), and replaces ReLU with a bounded odd function $\sigma_\mathrm{odd}=\tanh$.
    - **Mechanism**: While adjacency prevents signal vanishing, two new problems arise in deep layers: (i) Cross-layer scale drift: under continuous residual and non-linear stacking, magnitudes vary across layers, making optimization unstable. LayerNorm normalizes each stalk independently and re-stretches them with learnable affine parameters $\gamma^{(l)},\beta^{(l)}\in\mathbb{R}^f$, stabilizing both forward and backward passes. (ii) The asymmetric truncation of ReLU in a "residual - message" structure leads to "adjusting in only one direction," causing feature geometry drift over multiple layers. The odd function $\tanh$ preserves positive-negative symmetry and controls update magnitude via boundedness.
    - **Design Motivation**: Both follow empirical established practices from deep Transformers/ResNets but are organized by stalk dimension—BatchNorm across node vectors would disrupt sheaf structural information.

3.  **Per-node Per-stalk Gating**:
    - **Function**: Learns a scalar gate $[(\mathbf{G}^{(l)})_u]_s\in[0,1]$ for each node $u$ and each stalk dimension $s$. It performs channel-wise selection on the "aggregated and activated update term" via $(\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot(\cdot)$.
    - **Mechanism**: The gate is computed from the concatenation of the current stalk representation $\mathbf{X}_{u,s}^{(l)}$ and the "aggregated but not yet activated" intermediate value $\bar{\mathbf{X}}_{u,s}^{(l)}$ through $\mathrm{sigmoid}(w_g^{(l)}[\cdot;\cdot]+b_g^{(l)})$, where $w_g^{(l)}\in\mathbb{R}^{1\times 2f}$ is shared across all stalks with minimal parameters.
    - **Design Motivation**: Even with LN and adjacency, repeated weighted aggregation can accumulate noise components (similar to the *attention sink* phenomenon). Per-stalk gating allows the model to **selectively** let certain dimensions "update less" or be "completely gated," limiting noise accumulation and protecting representation quality.

## Key Experimental Results

### Main Results
Synthetic long-range tasks G0–G10 (3-class community detection, incrementally reconnecting 10% of homophilic edges to cross-community edges):

| Dataset (level) | Metric | DNSD-diag (adj+odd+gate) | NSD-diag | Gain |
|---|---|---|---|---|
| G4 (L12) | acc % | **86.1 ± 1.8** | 51.2 ± 2.1 | +34.9 pp |
| G5 (L12) | acc % | **81.5 ± 5.5** | 51.2 ± 0.7 | +30.3 pp |
| G6 (L16) | acc % | **75.6 ± 4.7** | 49.1 ± 1.7 | +26.5 pp |
| G7 (L12) | acc % | **63.4 ± 4.4** | 49.1 ± 1.2 | +14.3 pp |
| G10 (L16) | acc % | **96.2 ± 1.3** | 85.5 ± 4.7 | +10.7 pp |

DNSD-full (adj+odd+gate) reaches **97.5 ± 0.8** on G10 (vs. 84.0 ± 4.0 for NSD-full). Optimal performance consistently occurs at L12–L16, whereas the optimal points for NSD/MPNN/GAT almost all stay at L2–L4—directly validating that "DNSD can use depth, while NSD cannot."

### Ablation Study
Incremental ablation on synthetic datasets (diag, optimal depth):

| Config (diag) | G4 acc | G6 acc | Description |
|---|---|---|---|
| Original NSD | 51.2 | 49.1 | Baseline, fails at depth |
| + adj | 53.5 → 60+ | 60.4 | **Adding adj alone** pushes G5–G6 to 60+, the primary factor |
| + adj + odd | **86.1**(G4) | 75.6 | Adding odd activation completes deep stability |
| + adj + gate (no odd) | 83.5 | 74.4 | Gate is secondary but synergistic |
| + adj + odd + gate (full) | **75.0**(G5,L16) | 64.6(L16) | Full map set reaches 97.5% on G10 |

Real-world heterophilic benchmarks (Roman Empire, Amazon Ratings, Minesweeper, Tolokers, Questions, Penn94; diag, L≤8): DNSD ranks in the top tier on every dataset. Although depth was limited to ≤8 due to computation, the trend aligns with synthetic experiments.

### Key Findings
- **Adjacency replacement is the primary factor**: Adding adj alone pulls deep accuracy from ~50% to 60–80%; odd activation and gating serve as "performance-enhancing stabilizers."
- **Optimal depth for DNSD centers around L12–L16**, while GNN/NSD baselines peak at L2–L4—providing direct evidence of "deep usability."
- **Theoretical guarantees $\neq$ Engineering availability**: NSD's anti-collapse theorem is sound, but due to signal decay, ReLU truncation, scale drift, and noise accumulation—four "low-level details"—the architecture still collapses at depth. This suggests that for any "deeply scalable" graph model theory, one must question its effectiveness under engineering implementations.
- **DNSD $\approx$ matrix-valued GAT with representation normalization**: Table 1 aligns GAT/NSD/DNSD along four axes—Update Operator (Dependency vs. Disagreement), Edge Transformation (Scalar vs. Matrix), Normalization (Attention Scores vs. Representations), and Deep Behavior (Averaging vs. Vanishing vs. Mitigated Decay)—providing a highly insightful unified perspective.

## Highlights & Insights
- **"The discarded term is key"**: DNSD invents almost no new operators—the sheaf convolution $A_\mathcal{F}$ already appeared in Bodnar et al.'s original discretization derivation, only to be discarded in the final NSD architecture. Restoring the "discarded identity component" makes deep sheaf networks viable. This research path of "returning to intermediate steps of previous papers for answers" is highly valuable for analyzing any "theoretically strong but practically poor" work.
- **Unified GAT/NSD/DNSD Comparison**: Using the "Dependency vs. Disagreement + Scalar vs. Matrix + Normalization Location" axes provides a clear coordinate for "what kind of graph layers are suitable as backbones for foundation models."
- **Transferable Design Patterns**: Treating LayerNorm + odd activation + gating as a "standard trio for deep stackable graph models"—this recipe is worth trying in any architecture involving "repeated layer-wise weighted aggregation" (not just sheaves, but standard GNNs, hypergraphs, or operators on simplicial complexes).

## Limitations & Future Work
- The authors acknowledge that **orthogonal restriction maps** are difficult to train stably at depth and explicitly leave this to future work. This is the most expressive parameterization of the sheaf framework and could further improve performance if tamed.
- Real-world data experiments were constrained to **L≤8** due to computational limits; the benefits of DNSD at greater depths (L12–L16) on real graphs are not fully validated.
- The "deep receptive field requirement" of the synthetic G datasets is based on "k-NN + reconnection," which is somewhat artificial. Re-verifying DNSD’s advantage on complex long-range dependencies in large real-world graphs (e.g., social networks) is required.
- Computational overhead: Matrix-valued edge mappings are more expensive than GAT's scalar attention. The paper lacks in-depth wall-clock analysis, which is an unavoidable engineering hurdle for graph foundation models.
- Natural extension: Porting DNSD's "adjacency + trio" concept back to higher-order diffusions on **simplicial/cellular complexes**, or combining it with multi-hop sheaves (Bamberger et al.), could yield both deep and wide graph backbones.

## Related Work & Insights
- **vs. NSD (Bodnar et al., 2022)**: DNSD shares the cellular sheaf mathematical framework but changes the update operator from Laplacian to adjacency and adds LN/odd/gating. NSD stalls at L2–L4, while DNSD remains stable up to L16; the fundamental difference is whether updates stem from disagreement or dependency.
- **vs. GAT (Veličković et al., 2017; Brody et al., 2021)**: Both are adjacency-based, but GAT uses scalar attention + softmax normalization, leading to convex aggregation and oversmoothing at depth. DNSD uses matrix-valued edge mappings + representation-level LN, allowing non-convex aggregation to prevent collapse. DNSD can be viewed as a "matrix-valued, normalization-shifted version of GAT."
- **vs. Multi-hop / Attention-based Sheaf Extensions (Barbero 2022a/b, Zaghen 2024, Bamberger 2024)**: These works use precomputed maps, attention, or multi-hop connections but do not directly address the "signal vanishing at depth." DNSD is the first sheaf architecture to prioritize "deep stackability" and provide a systematic solution.
- **vs. Transformer Deep Training Techniques (ResNet/LN/Residuals)**: DNSD's use of LayerNorm and residuals stems from general deep learning expertise, but the authors emphasize that "normalization must be done per stalk dimension rather than per node"—applying general tricks to a sheaf framework still requires structural awareness.

## Rating
- Novelty: ⭐⭐⭐⭐ Does not construct a new mathematical object but discovers that the "discarded term" in the original paper is key, combined with standard deep training components. High insight value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically scanned 5 depths × 11 perturbation levels × multiple configurations on synthetic data; covered 6 heterophilic benchmarks on real data. Real graph depth restricted to L≤8 is the only minor regret.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical chain, progressing from "theoretical guarantee → practical failure → four mechanisms → four fixes → GAT unified explanation." The three-axis comparison in Table 1 is exemplary.
- Value: ⭐⭐⭐⭐ Provides a well-founded candidate for "deep stackable backbones for graph foundation models" and forces all subsequent sheaf work to confront the "adj vs. Laplacian" choice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](../../ICLR2026/graph_learning/cooperative_sheaf_neural_networks.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)

</div>

<!-- RELATED:END -->
