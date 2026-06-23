---
title: >-
  [Paper Note] Deep Neural Sheaf Diffusion
description: >-
  [ICML 2026][Graph Learning][Paper Note] This paper identifies that Neural Sheaf Diffusion (NSD) loses its theoretically guaranteed resistance to collapse at deep layers because the "disagreement signal" of the sheaf Laplacian vanishes as diffusion converges. DNSD replaces the Laplacian with a **sheaf adjacency operator** and incorporates LayerNorm, odd activ
tags:
  - ICML 2026
  - Graph Learning
date: 2026-05-08
content_hash: 835a35e995475a0f
---
# Deep Neural Sheaf Diffusion

**Conference**: ICML2026  
**arXiv**: [2605.19021](https://arxiv.org/abs/2605.19021)  
**Code**: https://github.com/remibourgerie/deep-neural-sheaf-diffusion  
**Area**: Graph Learning  
**Keywords**: Neural Sheaf Diffusion, Deep GNN, Sheaf Adjacency Operator, Oversmoothing, Graph Foundation Models

## TL;DR
This paper identifies that Neural Sheaf Diffusion (NSD) loses its theoretically guaranteed resistance to collapse at deep layers because the "disagreement signal" of the sheaf Laplacian vanishes as diffusion converges. DNSD replaces the Laplacian with a **sheaf adjacency operator** and incorporates LayerNorm, odd activation functions, and per-stalk gating. This allows the sheaf architecture to be stably stacked up to 16 layers for the first time, achieving up to a 30 pp improvement over GNN/NSD baselines on synthetic long-range tasks and consistent leads on real-world heterophilic graph benchmarks.

## Background & Motivation

**Background**: Standard GNNs (GCN, GAT, etc.) propagate information through layer-wise "weighted neighbor averaging." Theoretically, deeper layers provide a larger receptive field; however, in practice, deep GNNs generally fail to train. Literature summarizes this predicament as *oversmoothing* (convergence of node representations) and *oversquashing* (compression of long-distance signals). The root cause is that message passing is essentially a "convex combination," where repeated iterations inevitably smooth out differences.

**Limitations of Prior Work**: Neural Sheaf Diffusion (NSD), proposed by Bodnar et al. (2022), uses a **cellular sheaf** to assign a learnable linear map $\mathcal{F}_{v\trianglelefteq e}$ to each edge, constructing a sheaf Laplacian $\Delta_\mathcal{F}$ to replace the standard graph Laplacian. It theoretically proves that under appropriate restriction maps, the steady state of sheaf diffusion can separate almost any label configuration, thus **not** collapsing due to depth. However, this paper empirically finds that this guarantee **fails in practice**—NSD performance also collapses as the number of layers increases.

**Key Challenge**: The essence of the sheaf Laplacian is to "measure the disagreement between adjacent stalks," i.e., $\Delta_\mathcal{F}\mathbf{X}$ measures "how much has not yet been aligned." The goal of the diffusion process is precisely to eliminate disagreement, so $\Delta_\mathcal{F}\mathbf{X}$ **monotonically tends to 0** as the number of layers increases. Deep networks end up updating on increasingly faint residuals, and the loss becomes nearly insensitive to deep parameters. Combined with the asymmetric truncation of ReLU, cross-layer scale drift, and uniform noise propagation, a gap opens between theory and practice.

**Goal**: To transform the "theoretical depth" of NSD into "practical depth," making the sheaf architecture a stackable backbone for graph foundation models.

**Key Insight**: Since the problem lies in "using disagreement to drive updates," the update operator is replaced with one "driven by dependencies"—specifically, using the sheaf **adjacency** operator $A_\mathcal{F}$ instead of the Laplacian $\Delta_\mathcal{F}$. This sheaf convolution operator appeared in the original derivation by Bodnar et al. (their Eq. 4), but was discarded in the final architecture in favor of applying non-linearity only to the Laplacian term. DNSD restores this discarded path and overlays standard deep training components (LayerNorm + odd activation + gating) to form a coherent recipe for a "deep-ready sheaf network."

**Core Idea**: Change sheaf diffusion from "subtracting a vanishing disagreement" to "aggregating matrix-valued dependencies," and stabilize deep dynamics with LayerNorm, odd activations, and per-stalk gating.

## Method

### Overall Architecture
DNSD addresses the gap where NSD is theoretically resistant to collapse but practically fails at depth by switching the driving operator from "disagreement measurement" to "dependency measurement" and adding standard stability components for deep networks. Each layer maintains a $d\times f$ stalk representation $\mathbf{X}_v^{(l)}$ for each node. Restriction maps $\mathcal{F}^{(l)}_{v\trianglelefteq e}$ for that layer are learned from node representations, and the sheaf **adjacency** operator $A_\mathcal{F}^{(l)}$ (rather than the Laplacian) aggregates neighbors. This is followed by an odd activation, per-stalk gating filtration, residual addition, and LayerNorm. The complete layer update is:

$$\mathbf{X}^{(l+1)} = \mathrm{LN}\!\big((1+\epsilon^{(l)})\mathbf{X}^{(l)} - (\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot \sigma_{\mathrm{odd}}(A_\mathcal{F}^{(l)}\mathbf{X}^{(l)} W_1^{(l)}) W_2^{(l)}\big)$$

The output is then projected back to the task space for node classification. Compared to the original NSD: NSD uses $\Delta_\mathcal{F}$, uses ReLU, lacks LayerNorm, and lacks gating. DNSD replaces all four, with the adjacency replacement validated as the most critical factor.

### Key Designs

**1. Replacing the Laplacian with the Sheaf Adjacency Operator: Blocking the Root Cause of Signal Vanishing**

The root of NSD's failure is that its aggregation operator $\Delta_\mathcal{F}=D_\mathcal{F}^{-1/2} L_\mathcal{F} D_\mathcal{F}^{-1/2}$ measures the "unaligned components between neighbors." Diffusion iterations push this disagreement toward 0, so in deep layers, $\sigma(\Delta_\mathcal{F}\mathbf{X} W_1)W_2$ repeatedly feeds "near-zero small signals" into the non-linearity, leaving deep parameters with almost no gradient. DNSD replaces this with the sheaf adjacency $A_\mathcal{F}$, whose block matrix elements are $(A_\mathcal{F})_{uv}=\mathcal{F}_{u\trianglelefteq e}^\top \mathcal{F}_{v\trianglelefteq e}$. The update term becomes $\sigma(A_\mathcal{F}\mathbf{X} W_1)W_2$—aggregating the **entire** representation of neighbors using matrix-valued edge functions rather than their difference. This "dependency signal" does not vanish with diffusion convergence, maintaining information at initialization and when stacked to 16 layers. The authors also provide a unified explanation from a graph attention perspective: GAT is also adjacency-based but uses scalar softmax attention, while DNSD replaces scalar attention scores with matrix-valued edge maps and shifts normalization from attention scores to node representations.

**2. LayerNorm + Odd Activation: Stabilizing New Issues from Adjacency**

While switching to adjacency prevents signal loss, it exposes two new problems at depth. First is scale drift of representations: repeated residual stacking and non-linearities lead to inconsistent magnitudes, destabilizing optimization. DNSD uses row-wise LayerNorm to normalize each stalk $\tilde{\mathbf{X}}_u^{(l)}\in\mathbb{R}^{d\times f}$ along the feature dimension $f$ ($\mu_u,\sigma_u\in\mathbb{R}^d$), then re-stretches them with learnable affine parameters $\gamma^{(l)},\beta^{(l)}\in\mathbb{R}^f$, stabilizing both forward and backward passes. Second is the asymmetric truncation of ReLU: in a "residual - message" subtraction structure, it can only adjust in one direction, causing feature geometry to drift over multiple layers. DNSD employs the bounded odd function $\sigma_\mathrm{odd}=\tanh$ to maintain symmetry and control update magnitudes. These ideas are borrowed from deep Transformers/ResNets but must be organized by stalk—standard BatchNorm across all nodes would destroy sheaf structural information.

**3. Per-node Per-stalk Gating: Restricting Noise Accumulation with Depth**

Even with adjacency and LN, repeated weighted aggregation can cause noise components (similar to attention sinks) to accumulate. DNSD learns a scalar gate $[(\mathbf{G}^{(l)})_u]_s\in[0,1]$ for each node $u$ and stalk dimension $s$. The update term, after aggregation and non-linearity, is filtered via $(\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot(\cdot)$. The gate is computed by concatenating the current stalk representation $\mathbf{X}_{u,s}^{(l)}$ with the "aggregated but not yet activated" intermediate value $\bar{\mathbf{X}}_{u,s}^{(l)}$ and passing it through $\mathrm{sigmoid}(w_g^{(l)}[\cdot;\cdot]+b_g^{(l)})$, where $w_g^{(l)}\in\mathbb{R}^{1\times 2f}$ is shared across all stalks. This allows the model to selectively "under-update" or "gate out" certain dimensions, constraining noise accumulation and protecting representation quality.

### Loss & Training
The task follows the node classification setup from NSD (synthetic G0–G10 and 6 real heterophilic benchmarks) using cross-entropy loss. Restriction maps use either diagonal or full parameterizations (orthogonal is difficult to train at depth and left for future work). Layer counts are swept through $\{2,4,8,12,16\}$, and NSD results are reproduced under identical hyperparameter budgets for fairness.

## Key Experimental Results

### Main Results
On synthetic long-range tasks G0–G10 (3-class community detection where 10% of homophilic edges are progressively rewired as inter-community edges):

| Dataset (level) | Metric | DNSD-diag (adj+odd+gate) | NSD-diag | Gain |
|---|---|---|---|---|
| G4 (L12) | acc % | **86.1 ± 1.8** | 51.2 ± 2.1 | +34.9 pp |
| G5 (L12) | acc % | **81.5 ± 5.5** | 51.2 ± 0.7 | +30.3 pp |
| G6 (L16) | acc % | **75.6 ± 4.7** | 49.1 ± 1.7 | +26.5 pp |
| G7 (L12) | acc % | **63.4 ± 4.4** | 49.1 ± 1.2 | +14.3 pp |
| G10 (L16) | acc % | **96.2 ± 1.3** | 85.5 ± 4.7 | +10.7 pp |

DNSD-full (adj+odd+gate) reaches **97.5 ± 0.8** on G10 (NSD-full is only 84.0 ± 4.0). The optimal number of layers generally appears at L12–L16, whereas optimal points for NSD/MPNN/GAT almost entirely stop at L2–L4—directly verifying that "DNSD can use depth, while NSD cannot."

### Ablation Study
"Step-by-step addition" ablation on synthetic datasets (diag, optimal depth):

| Configuration (diag) | G4 acc | G6 acc | Description |
|---|---|---|---|
| Original NSD only | 51.2 | 49.1 | Baseline, fails at depth |
| + adj | 53.5 → 60+ | 60.4 | **Adding adj alone** pushes G5–G6 to 60+, the main driver |
| + adj + odd | **86.1**(G4) | 75.6 | Odd activation completes deep stability |
| + adj + gate (no odd) | 83.5 | 74.4 | Gate is secondary but synergetic |
| + adj + odd + gate (full) | **75.0**(G5,L16) | 64.6(L16) | Full maps with all three components, 97.5% on G10 |

On real heterophilic benchmarks (Roman Empire, Amazon Ratings, Minesweeper, Tolokers, Questions, Penn94, diag, $L\le8$): DNSD ranks among the top three in every dataset. Although depth was constrained to $L\le8$ due to computation, the trend remains consistent with synthetic experiments.

### Key Findings
- **Adjacency replacement is the primary factor**: Ablations show that adding adj alone pulls deep accuracy from ~50% to 60–80%; odd activation and gating are "stabilizing icing on the cake."
- **Optimal depth for DNSD is concentrated at L12–L16**, whereas GNN/NSD baselines peak at L2–L4—this is the most direct evidence of "practical depth."
- **Theoretical guarantees $\neq$ Engineering usability**: NSD's anti-collapse theorem is true, but due to signal decay, ReLU truncation, scale drift, and noise accumulation, the architecture still collapses. This suggests that the engineering validity of any "theoretically deep-scalable" graph model must be scrutinized.
- **DNSD $\approx$ matrix-valued GAT with representation normalization**: Table 1 aligns GAT/NSD/DNSD along four axes: update operator (dependency vs. difference), edge transform (scalar vs. matrix), normalization (attention scores vs. representation), and deep behavior (averaging vs. signal vanishing vs. decay mitigation).

## Highlights & Insights
- **"The discarded term is the key"**: DNSD invents almost no new operators—the sheaf convolution $A_\mathcal{F}$ was already present in the discretization derivation of Bodnar et al., but was omitted from the original NSD architecture. Restoring the "discarded identity component" makes deep sheaf networks usable. This research path of "finding answers in the intermediate steps of previous papers" is highly instructive for analyzing other work that is "theoretically strong but practically weak."
- **Unified GAT/NSD/DNSD Comparison Table**: Using three axes (dependency vs. difference, scalar vs. matrix, normalization position) provides a clear coordinate system for what future graph foundation model backbones should look like.
- **Transferable Design Patterns**: Treating LayerNorm + odd activation + gating as a "standard trio for deep stackable graph models"—this recipe is worth trying in any architecture involving repeated weighted aggregation (not just sheaves, but standard GNNs, hypergraphs, or simplicial complexes).

## Limitations & Future Work
- The authors admit that **orthogonal restriction maps** are difficult to train stably at depth and are explicitly left for future work. This is one of the most expressive parameterizations of the sheaf framework and could yield further improvements if tamed.
- Real-world experiments were constrained to **$L\le8$** due to computational limits; the benefits of DNSD at even deeper levels (L12–L16) on real graphs are not fully verified.
- The "deep receptive field requirement" of the synthetic G datasets is constructed via "k-NN + rewiring," which is somewhat artificial. Real-world long-range dependencies in large graphs (e.g., social networks) may be more complex.
- Computational overhead: Matrix-valued edge maps are more expensive than GAT's scalar attention. The paper does not provide an in-depth wall-clock analysis, which is an unavoidable engineering challenge for graph foundation models.
- Natural extensions: Migrating the "adjacency + trio" concept back to higher-order diffusion on **simplicial/cellular complexes** or combining it with multi-hop sheaves could produce deep and wide graph backbones.

## Related Work & Insights
- **vs NSD (Bodnar et al., 2022)**: DNSD shares the mathematical skeleton of cellular sheaves with NSD but changes the update operator from Laplacian to adjacency and adds LN/odd/gating. While NSD stalls at L2–L4, DNSD is stable to L16. The fundamental difference is whether updates stem from disagreement or dependency.
- **vs GAT (Veličković et al., 2017; Brody et al., 2021)**: Both are adjacency-based. However, GAT uses scalar attention + softmax normalization, leading to convex aggregation and oversmoothing. DNSD uses matrix-valued edge maps + representation-level LN, allowing non-convex aggregation and preventing collapse. DNSD can be viewed as a "matrix-valued, normalization-shifted version of GAT."
- **vs Multi-hop/Attention-based Sheaf Extensions (Barbero 2022a/b, Zaghen 2024, Bamberger 2024)**: These works add attention or multi-hop components but do not directly address the "vanishing signal at depth" problem. DNSD is the first sheaf architecture to prioritize "deep stackability" and provide a systematic solution.
- **vs Transformer Deep Training Techniques (ResNet/LN/Residuals)**: DNSD’s use of LayerNorm and residuals stems from general deep learning wisdom, but the authors emphasize that normalization "must be done per stalk rather than per node"—applying general techniques to the sheaf framework still requires structural awareness.

## Rating
- Novelty: ⭐⭐⭐⭐ Not inventing a new mathematical object, but discovering that the "discarded term" is the key and assembling deep training components; very high insight value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically swept 5 depths × 11 perturbation levels × multiple configurations on synthetic data; covered 6 real benchmarks; depth limit of $L\le8$ on real graphs is the only minor drawback.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is exceptionally clear, progressing through "theory guarantee → practical failure → four mechanisms → four fixes → unified GAT explanation." The triple-axis comparison in Table 1 is exemplary.
- Value: ⭐⭐⭐⭐ Directly provides a grounded candidate for deep-stackable graph foundation model backbones and forces all subsequent sheaf work to confront the "adj vs Laplacian" choice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](../../ICLR2026/graph_learning/cooperative_sheaf_neural_networks.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] What Makes a Desired Graph for Relational Deep Learning?](what_makes_a_desired_graph_for_relational_deep_learning.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)

</div>

<!-- RELATED:END -->
