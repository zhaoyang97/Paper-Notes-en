---
title: >-
  [Paper Note] Deep Neural Sheaf Diffusion
description: >-
  [ICML 2026][Graph Learning][Paper Note] This paper identifies that Neural Sheaf Diffusion (NSD) loses its theoretically guaranteed anti-collapse capability in deep layers because the "disagreement signal" of the sheaf Laplacian vanishes as diffusion converges. DNSD replaces the Laplacian with the **sheaf adjacency operator**, combined with LayerNorm, odd act
tags:
  - ICML 2026
  - Graph Learning
date: 2026-05-08
content_hash: 68c0a7584c8aeceb
---
# Deep Neural Sheaf Diffusion

**Conference**: ICML2026  
**arXiv**: [2605.19021](https://arxiv.org/abs/2105.19021)  
**Code**: https://github.com/remibourgerie/deep-neural-sheaf-diffusion  
**Area**: Graph Learning  
**Keywords**: Neural Sheaf Diffusion, Deep GNN, Sheaf Adjacency Operator, Oversmoothing, Graph Foundation Model

## TL;DR
This paper identifies that Neural Sheaf Diffusion (NSD) loses its theoretically guaranteed anti-collapse capability in deep layers because the "disagreement signal" of the sheaf Laplacian vanishes as diffusion converges. DNSD replaces the Laplacian with the **sheaf adjacency operator**, combined with LayerNorm, odd activations, and per-stalk gating. This allows the sheaf architecture to be stably stacked up to 16 layers for the first time, achieving up to a 30 pp improvement over GNN/NSD baselines on synthetic long-range tasks and consistent leads on real-world heterophilic graph benchmarks.

## Background & Motivation

**Background**: Standard GNNs (GCN / GAT, etc.) propagate information through layer-wise "weighted neighborhood averaging." Theoretically, deeper layers provide a larger receptive field. However, in practice, deep GNNs often fail during training, a predicament summarized in literature as *oversmoothing* (convergence of node representations) and *oversquashing* (compression of long-distance signals). Rooted in message passing being essentially a "convex combination," repeated iterations inevitably flatten differences.

**Limitations of Prior Work**: Neural Sheaf Diffusion (NSD), proposed by Bodnar et al. (2022), uses **cellular sheaves** to assign a learnable linear mapping $\mathcal{F}_{v\trianglelefteq e}$ to each edge. This constructs a sheaf Laplacian $\Delta_\mathcal{F}$ to replace the standard graph Laplacian. It theoretically proves that under appropriate restriction maps, the steady state of sheaf diffusion can separate almost any label configuration, thus **avoiding** collapse due to depth. However, this paper empirically finds that this guarantee **fails in practice**—NSD performance also collapses when the network becomes deeper.

**Key Challenge**: The essence of the sheaf Laplacian is to "measure the disagreement between adjacent stalks," i.e., $\Delta_\mathcal{F}\mathbf{X}$ measures "how much has not yet aligned." Since the goal of the diffusion process is specifically to eliminate disagreement, $\Delta_\mathcal{F}\mathbf{X}$ **monotonically tends to 0** as the number of layers increases. Deep networks end up performing updates on increasingly weak residuals, making loss functions nearly insensitive to deep parameters. Combined with the asymmetric truncation of ReLU, scale drift across layers, and the uniform propagation of noise, a significant gap opens between theory and practice.

**Goal**: To transform the "theoretical depth" of NSD into "practical depth," making the sheaf architecture a stackable backbone for graph foundation models.

**Key Insight**: Instead of "driving updates with disagreement," change the update operator to "drive updates with dependencies"—specifically, replacing the sheaf Laplacian $\Delta_\mathcal{F}$ with the sheaf **adjacency** operator $A_\mathcal{F}$. This sheaf convolution operator actually appeared in the original derivation by Bodnar et al. (their Eq. 4), but the final architecture only applied non-linearity to the Laplacian term and discarded the identity component, thereby introducing the side effect of signal vanishing with depth. DNSD restores this discarded path and layers it with standard components for deep training (LayerNorm + odd activation + gating) to form a coherent recipe for a "depth-ready sheaf network."

**Core Idea**: Shift sheaf diffusion from "subtracting a vanishing disagreement" to "aggregating matrix-valued dependencies," and stabilize deep dynamics with LayerNorm, odd activations, and per-stalk gating.

## Method

### Overall Architecture
DNSD addresses the gap where NSD is theoretically resistant to collapse but practically collapses at depth. It replaces the "disagreement-measuring" operator with a "dependency-measuring" one and integrates standard stabilizer components for deep networks. Each layer maintains a $d\times f$ stalk representation $\mathbf{X}_v^{(l)}$ for each node. Restriction maps $\mathcal{F}^{(l)}_{v\trianglelefteq e}$ are learned from node representations, and neighbors are aggregated using the sheaf **adjacency** operator $A_\mathcal{F}^{(l)}$ (instead of the Laplacian). This is followed by an odd activation and per-stalk gating filtering. Finally, a residual sum is performed followed by LayerNorm. The complete layer update is written as:

$$\mathbf{X}^{(l+1)} = \mathrm{LN}\!\big((1+\epsilon^{(l)})\mathbf{X}^{(l)} - (\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot \sigma_{\mathrm{odd}}(A_\mathcal{F}^{(l)}\mathbf{X}^{(l)} W_1^{(l)}) W_2^{(l)}\big)$$

The output is then projected back to the task space for node classification. Compared point-by-point with the original NSD update: NSD uses $\Delta_\mathcal{F}$, ReLU, no LayerNorm, and no gating; DNSD replaces all four, with the adjacency substitution validated as the most critical factor.

### Key Designs

**1. Sheaf Adjacency Operator as a Substitute for Laplacian: Resolving the Root Cause of Signal Vanishing**

The root of NSD's failure is that its aggregation operator $\Delta_\mathcal{F}=D_\mathcal{F}^{-1/2} L_\mathcal{F} D_\mathcal{F}^{-1/2}$ measures the "unaligned components between neighbors." Diffusion iterations push this disagreement toward zero. Consequently, in deep layers, $\sigma(\Delta_\mathcal{F}\mathbf{X} W_1)W_2$ repeatedly feeds "near-zero small signals" into the non-linearity, and deep parameters receive almost no gradient. DNSD replaces this with the sheaf adjacency $A_\mathcal{F}$, whose block matrix elements are $(A_\mathcal{F})_{uv}=\mathcal{F}_{u\trianglelefteq e}^\top \mathcal{F}_{v\trianglelefteq e}$. The update term becomes $\sigma(A_\mathcal{F}\mathbf{X} W_1)W_2$—aggregating the **overall** representation of neighbors using matrix-valued edge functions rather than their differences. This "dependency signal" does not vanish as diffusion converges, maintaining information content during initialization and even at 16 layers, thus directly fixing the rupture between theoretical guarantees and practical collapse. The authors also provide a unified explanation from the perspective of graph attention: GAT is also adjacency-based but uses scalar softmax attention, whereas DNSD replaces GAT's scalar attention scores with matrix-valued edge mappings and moves normalization from attention scores to node representations.

**2. LayerNorm + Odd Activation: Stabilizing New Issues from Adjacency Substitution**

While substituting adjacency prevents signal vanishing, it exposes two new issues in deep layers that require stabilization. First is the scale drift of representations across layers: magnitudes across layers become inconsistent after continuous residual stacking and non-linearities, leading to optimization instability. DNSD uses row-wise LayerNorm to standardize each stalk $\tilde{\mathbf{X}}_u^{(l)}\in\mathbb{R}^{d\times f}$ along the feature dimension $f$ ($\mu_u,\sigma_u\in\mathbb{R}^d$), then re-scales using learnable affine parameters $\gamma^{(l)},\beta^{(l)}\in\mathbb{R}^f$, stabilizing both forward and backward passes. Second is the asymmetric truncation of ReLU: in a "residual - message" subtraction structure, it can only adjust in one direction, causing feature geometry to drift over many layers. DNSD instead uses a bounded odd function $\sigma_\mathrm{odd}=\tanh$, which preserves positive-negative symmetry and controls update magnitudes through boundedness. These strategies are borrowed from deep Transformer/ResNet architectures but must be organized by stalk dimension—applying BatchNorm to a flattened vector of all nodes would scatter the sheaf structural information.

**3. Per-node Per-stalk Gating: Restricting Noise Accumulation with Depth**

Even with adjacency and LN, repeated weighted aggregation can cause certain noise components (similar to attention sinks) to accumulate with depth. DNSD learns a scalar gate $[(\mathbf{G}^{(l)})_u]_s\in[0,1]$ for each node $u$ and each stalk dimension $s$. This is used in $(\mathbf{G}^{(l)}\otimes \mathbf{1}_f^\top)\odot(\cdot)$ to perform channel-wise filtering of the "aggregated and activated update term." This gate is derived by concatenating current stalk representations $\mathbf{X}_{u,s}^{(l)}$ and the "aggregated but non-activated" intermediate values $\bar{\mathbf{X}}_{u,s}^{(l)}$, then passing them through $\mathrm{sigmoid}(w_g^{(l)}[\cdot;\cdot]+b_g^{(l)})$, where $w_g^{(l)}\in\mathbb{R}^{1\times 2f}$ is shared across all stalks. This allows the model to selectively "reduce updates" or "completely gate out" certain dimensions, constraining noise accumulation and protecting representation quality.

### Loss & Training
The task follows the node classification setup of NSD (synthetic G0–G10 and 6 real heterophilic graph benchmarks) using cross-entropy loss. Restriction maps are parameterized as either diagonal or full (orthogonal maps were difficult to train at depth and are left for future work). Layer depth was searched across $\{2, 4, 8, 12, 16\}$, re-implementing NSD under the same hyperparameter budget for a fair comparison.

## Key Experimental Results

### Main Results
Synthetic long-range tasks G0–G10 (3-class community detection, with 10% of homophilic edges gradually rewired to cross-community edges):

| Dataset (level) | Metric | DNSD-diag (adj+odd+gate) | NSD-diag | Gain |
|---|---|---|---|---|
| G4 (L12) | acc % | **86.1 ± 1.8** | 51.2 ± 2.1 | +34.9 pp |
| G5 (L12) | acc % | **81.5 ± 5.5** | 51.2 ± 0.7 | +30.3 pp |
| G6 (L16) | acc % | **75.6 ± 4.7** | 49.1 ± 1.7 | +26.5 pp |
| G7 (L12) | acc % | **63.4 ± 4.4** | 49.1 ± 1.2 | +14.3 pp |
| G10 (L16) | acc % | **96.2 ± 1.3** | 85.5 ± 4.7 | +10.7 pp |

DNSD-full (adj+odd+gate) reached **97.5 ± 0.8** on G10 (Ours) vs 84.0 ± 4.0 (NSD-full). Furthermore, the best layer depth for DNSD generally appeared at L12–L16, while the best points for NSD/MPNN/GAT almost entirely stayed within L2–L4—directly verifying that "DNSD can use depth, while NSD cannot."

### Ablation Study
"Add-one-by-one" ablation on synthetic datasets (diag, best depth):

| Config (diag) | G4 acc | G6 acc | Description |
|---|---|---|---|
| Original NSD only (no adj/odd/gate) | 51.2 | 49.1 | Baseline, fails at depth |
| + adj | 53.5 → 60+ | 60.4 | **Adding adj alone** pushes G5–G6 to 60+, the primary factor |
| + adj + odd | **86.1**(G4) | 75.6 | Adding odd activation completes deep stability |
| + adj + gate (no odd) | 83.5 | 74.4 | Gate is secondary alone but synergistic |
| + adj + odd + gate (full) | **75.0**(G5,L12) | 64.6(L16) | Full map with trio, G10 reaches 97.5% |

Real-world heterophilic benchmarks (Roman Empire / Amazon Ratings / Minesweeper / Tolokers / Questions / Penn94, diag, L≤8): DNSD consistently placed among the top ranks for each dataset. Although depth was limited to ≤8 (due to computational constraints), the trend aligned with synthetic experiments.

### Key Findings
- **Adjacency substitution is the primary factor**: Ablations show that adding adjacency alone pulls deep accuracy from ~50% to 60–80%; odd activation and gating are "stabilizers" that further improve the results.
- **DNSD's best depth is concentrated at L12–L16**, whereas GNN/NSD baselines peak at L2–L4—this is the most direct evidence of "practical depth" for DNSD.
- **Theoretical Guarantee $\neq$ Engineering Availability**: NSD's anti-collapse theorem is true, but due to signal decay, ReLU truncation, scale drift, and noise accumulation, the architecture still collapses at depth. This suggests that any "theoretically deep" graph model should be scrutinized for effectiveness in engineering implementation.
- **DNSD $\approx$ matrix-valued GAT with representation normalization**: Table 1 aligns GAT / NSD / DNSD along four axes: update operator (dependency vs. difference), edge transform (scalar vs. matrix-valued), normalization (attention scores vs. representations), and deep behavior (averaging vs. signal vanishing vs. mitigation). This provides a very insightful unified perspective.

## Highlights & Insights
- **"The discarded term is key"**: DNSD introduces almost no new operators—the sheaf convolution $A_\mathcal{F}$ was already present in the discretization derivation of Bodnar et al., but was discarded in the original NSD architecture. Restoring the "discarded identity component" makes deep sheaf networks viable. This research path—returning to intermediate steps of prior papers for answers—is highly instructive for analyzing any work that is "theoretically strong but practically poor."
- **Unified Comparison Table of GAT/NSD/DNSD**: Using the three axes of "dependency vs. difference + scalar vs. matrix + normalization location" provides a clear coordinate system for determining what kind of graph layers are suitable as backbones for future foundation models.
- **Transferable Design Patterns**: Treating LayerNorm + odd activation + gating as a "standard trio for deep stackable graph models"—this recipe is worth testing in any architecture involving repeated weighted aggregation (not just sheaves, but also standard GNNs, hypergraphs, or operators on simplicial complexes).

## Limitations & Future Work
- The authors admit that **orthogonal restriction maps** are difficult to train stably at depth, leaving this for future work. This is actually the most expressive parameterization in the sheaf framework; taming it could lead to further improvements.
- Real-world experiments were constrained to **L≤8** due to compute; the benefits of DNSD at greater depths (L12–L16) on real graphs have not been fully verified.
- The "deep receptive field requirement" of synthetic G datasets is based on "k-NN + rewiring," which is somewhat artificial. Long-range dependence structures in real large-scale graphs (e.g., social networks) might be more complex, and more validation is needed to see if DNSD maintains its edge.
- Computational overhead: Matrix-valued edge mappings are more expensive than GAT's scalar attention. The paper does not analyze wall-clock time in depth, which remains an engineering issue for graph foundation models.
- Natural extension: Applying the "adjacency + trio" concept of DNSD back to higher-order diffusions on **simplicial / cellular complexes**, or combining it with multi-hop sheaves (Bamberger et al.), might yield graph backbones that are both deep and wide.

## Related Work & Insights
- **vs NSD (Bodnar et al., 2022)**: DNSD shares the mathematical skeleton of cellular sheaves with NSD but changes the update operator from Laplacian to adjacency and adds LN/odd/gating. NSD stalls at L2–L4, while DNSD can stabilize to L16. The fundamental difference lies in whether updates originate from disagreement or dependency.
- **vs GAT (Veličković et al., 2017; Brody et al., 2021)**: Both are adjacency-based. However, GAT uses scalar attention + softmax normalization leading to convex aggregation and averaging at depth. DNSD uses matrix-valued edge mappings + representation-level LN, allowing non-convex aggregation and preventing collapse at depth. DNSD can be viewed as a "matrix-valued, normalization-shifted version of GAT."
- **vs Multi-hop / Attention-based Sheaf Extensions (Barbero 2022a/b, Zaghen 2024, Bamberger 2024)**: These works use precomputed mappings, attention, or multi-hops but do not directly solve the "signal vanishing at depth" problem. DNSD is the first sheaf architecture to systematically provide a solution with "deep stackability" as its primary goal.
- **vs Transformer Deep Training Techniques (ResNet / LN / Residuals)**: The LayerNorm and residual structures in DNSD are directly inspired by general deep network experiences, but the authors emphasize that "normalization must be done per stalk rather than per node"—applying general techniques to the sheaf framework still requires structural awareness.

## Rating
- Novelty: ⭐⭐⭐⭐ Not about constructing a new mathematical object, but discovering that the "abandoned terms" in the original paper are key and adding deep training standards; the insight value is high.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically scanned 5 depths × 11 perturbation levels × multiple configurations on synthetic data; real-world data covered 6 heterophilic benchmarks. The only minor drawback is the depth limit (L≤8) on real graphs.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain of the argument is very clear, progressing from "theoretical guarantee → practical failure → four mechanisms → four fixes → unified GAT perspective." The triple-axis comparison in Table 1 is exemplary.
- Value: ⭐⭐⭐⭐ Directly provides a grounded candidate for a deep stackable backbone for graph foundation models and forces all subsequent sheaf work to confront the "adj vs. Laplacian" choice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](../../ICLR2026/graph_learning/cooperative_sheaf_neural_networks.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)

</div>

<!-- RELATED:END -->
