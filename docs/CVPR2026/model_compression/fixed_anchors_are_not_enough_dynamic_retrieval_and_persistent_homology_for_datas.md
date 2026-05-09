---
title: >-
  [Paper Note] Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation
description: >-
  [CVPR 2026][Model Compression][dataset distillation] RETA decouples two failure modes in residual matching for dataset distillation—the fit-complexity gap and the pull-to-anchor effect—by employing Dynamic Retrieval Connection (DRC) to adaptively select real patch anchors and Persistent Topology Alignment (PTA) to preserve intra-class diversity. The method achieves 64.3% (+3.1% vs. FADRM) on ImageNet-1K with ResNet-18 at IPC=50.
tags:
  - CVPR 2026
  - Model Compression
  - dataset distillation
  - residual matching
  - persistent homology
  - topology alignment
  - dynamic retrieval
date: 2026-05-08
content_hash: d1a697726442d4ec
---

# Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation

**Conference**: CVPR 2026
**arXiv**: [2602.24144](https://arxiv.org/abs/2602.24144)
**Code**: To be confirmed
**Area**: Dataset Distillation / Model Compression
**Keywords**: dataset distillation, residual matching, persistent homology, topology alignment, dynamic retrieval

## TL;DR
RETA decouples two failure modes in residual matching for dataset distillation—the fit-complexity gap and the pull-to-anchor effect—by employing Dynamic Retrieval Connection (DRC) to adaptively select real patch anchors and Persistent Topology Alignment (PTA) to preserve intra-class diversity. The method achieves 64.3% (+3.1% vs. FADRM) on ImageNet-1K with ResNet-18 at IPC=50.

## Background & Motivation

**Background**: Dataset distillation (DD) aims to compress large datasets into a small set of synthetic images such that models trained on the synthetic set approach the performance of full-dataset training. Decoupled DD methods (e.g., SRe2L, EDC, FADRM) separate the supervision objective and distribution alignment into two optimization streams, achieving improved stability and scalability. FADRM further introduces residual matching—periodically injecting real patches into synthetic images via residual connections to prevent information collapse caused by pure pixel-level updates.

**Limitations of Prior Work**: FADRM relies on fixed, pre-selected real patches as anchors, giving rise to two coupled failure modes:
   - **(i) Fit-Complexity Gap**: Fixed patches may be misaligned with the current synthetic features in teacher space (large fit gap), or the patches themselves may introduce high-frequency noise due to textural complexity (complexity inflation), both of which harm the generalization bound.
   - **(ii) Pull-to-Anchor Effect**: Each residual connection pulls the synthetic features toward the nearest real sample in teacher space; repeated application causes intra-class synthetic feature distances to contract as $\|y_i' - y_j'\| \leq \alpha\|y_i - y_j\| + (1-\alpha)\|a_i - a_j\|$, leading to premature merging of distinct clusters and loss of intra-class diversity.

**Key Challenge**: Fixed anchors can neither adaptively minimize the fit gap across stages nor reliably control complexity—anchor selection is locally suboptimal, and repeated anchoring globally degrades the class topological structure.

**Goal**
   - How to adaptively select residual anchors at each stage to simultaneously control the fit gap and complexity?
   - How to preserve intra-class feature diversity and topological structure under repeated residual injection?

**Key Insight**: Starting from a generalization bound decomposition (Theorem 4.1), the post-connection risk is decomposed into a fit term and a complexity term, providing theoretical guidance for anchor selection; from the perspective of topological data analysis (TDA), persistent homology is used to quantify class topology discrepancy and construct a differentiable regularization.

**Core Idea**: Replace fixed anchors with dynamic retrieval to resolve the fit-complexity trade-off, and counter class topology collapse caused by the pull-to-anchor effect via persistent homology topology alignment regularization.

## Method

### Overall Architecture
RETA operates on top of the decoupled DD framework. Given a large-scale training set $\mathcal{D}$, it produces a compact synthetic set $\tilde{\mathcal{C}}$. The optimization objective $\min_{\tilde{\mathcal{C}}} \mathcal{L}_{sup}(f_\theta; \tilde{\mathcal{C}}) + \beta \mathcal{R}_{align}(\tilde{\mathcal{C}}; \mathcal{D}, \mathcal{T})$ divides the total budget $B$ into $k+1$ blocks, each of $b = \lfloor B/(k+1) \rfloor$ steps executing the Eq. 1 optimization, with residual connections applied at the end of the first $k$ blocks. RETA introduces two modules at the residual connection step: DRC determines *which anchor to connect*, while PTA constrains *what global structure should be preserved after connection*.

### Key Designs

1. **Dynamic Retrieval Connection (DRC)**

    - **Function**: At each stage, dynamically retrieves the optimal real patch as a residual anchor for each synthetic image in each class.
    - **Mechanism**: A pool $p_c$ is constructed for each class $c$ (one $1 \times 1$ patch per real image). The frozen teacher $\phi(\cdot)$ encodes the synthetic images and all candidate patches. A fit-complexity score is defined as:
    $J(o|\tilde{x}_t) = (1-\lambda)\|q(\tilde{x}_t) - z(o)\|_2^2 + \lambda \cdot c(o)$
      where $q(\tilde{x}_t) = \text{Norm}(\phi(\tilde{x}_t))$ and $z(o) = \text{Norm}(\phi(o))$. The first term is the fit gap (distance between synthetic features and candidate patches in teacher space); the second term $c(o) = \text{Var}_{u \in \Omega_{D_t}}(\|\nabla(G_\sigma * o)(u)\|_2^2)$ is a complexity score—the spatial variance of gradient magnitudes after Gaussian smoothing, where large values indicate severe high-frequency spatial fluctuations. $\lambda$ balances fit and complexity. The optimal anchor is retrieved as $o^* = \arg\min_{o \in p_c} J(o|\tilde{x}_t)$, resampled to the current resolution, and used in the residual update $\tilde{x}_t \leftarrow \alpha \tilde{x}_t + (1-\alpha) \text{Resample}(o^*, D_t)$.
    - **Design Motivation**: Theorem 4.1 proves that the post-connection generalization bound contains both the fit gap $\Delta$ and the complexity gap $\mathfrak{R}_n(H \circ O) - \mathfrak{R}_n(H \circ \tilde{\mathcal{C}}_{pre})$. Fixed anchors cannot optimize both simultaneously. DRC operationalizes these two levers through per-stage dynamic retrieval—adaptively reducing $\Delta$ while regularizing complexity via $c(o)$.

2. **Persistent Topology Alignment (PTA)**

    - **Function**: Constructs a differentiable persistent homology regularization to align the topological structure (connected components and loops) of real and synthetic feature sets, countering the intra-class diversity collapse induced by the pull-to-anchor effect.
    - **Mechanism**: For each class $c$, the union $Z_c$ of synthetic features $Z_c^{syn} = \{\phi(\tilde{x}_i)\}$ and real features $Z_c^{real} = \{\phi(x): x \in p_c\}$ is formed. A class-balanced mutual $k$-NN graph is constructed, and persistent homology is run to obtain persistence diagrams $\mathcal{D}_c^{(q)}$ ($q=0$: connected components; $q=1$: loops). Each diagram is mapped to a persistence image (PI):
    $I^{(q)}(Z_c)[m] = \sum_{(b_j, p_j) \in \mathcal{D}_c^{(q)}} w_q(p_j) \exp\left(-\frac{\|u_m - (b_j, p_j)\|_2^2}{2\sigma^2}\right)$
      The alignment loss is:
    $\mathcal{L}_{topo} = \sum_c \left(\|I^{(0)}(Z_c^{syn}) - I^{(0)}(Z_c^{real})\|_2^2 + \gamma \|I^{(1)}(Z_c^{syn}) - I^{(1)}(Z_c^{real})\|_2^2\right)$
    - **Design Motivation**: The pull-to-anchor effect manifests in Betti curves as a leftward shift of $\mathcal{B}_0^{syn}$ (premature merging of connected components) and suppression of $\mathcal{B}_1^{syn}$ (premature disappearance of loops). PIs provide stable, differentiable topological summaries, allowing gradients from $\mathcal{L}_{topo}$ to flow to the synthetic inputs (through $\phi$, which is frozen but retained in the computation graph), thereby constraining synthetic features to maintain multi-scale connectivity and loop structure consistent with real data.

3. **Complementarity of DRC and PTA**

    - DRC optimizes the fit-complexity trade-off at the local level (individual anchor selection).
    - PTA preserves topological structure at the global level (the feature geometry of entire classes).
    - The two are jointly optimized: $\mathcal{L} \leftarrow \mathcal{L} + \lambda_{topo} \mathcal{L}_{topo}$.

### Loss & Training
- Based on the FADRM protocol: Adam optimizer, lr=0.25, $(\beta_1, \beta_2) = (0.5, 0.9)$.
- Optimization budget $B \in \{300, 2000\}$ (dataset-dependent), merge ratio $\alpha = 0.5$.
- DRC: $\lambda = 0.1$ (default fit-complexity trade-off).
- PTA: $\lambda_{topo} = 0.5$, $k \approx 10\text{-}20$, $32 \times 32$ PI grid.
- Feature cache refreshed periodically (every $T$ steps) to reduce computational overhead.
- Runs on a single RTX 4090.

## Key Experimental Results

### Main Results (ResNet-18, IPC=10/50)

| Dataset | IPC | SRe2L | RDED | EDC | FADRM+ | **RETA** | **Δ** |
|--------|-----|-------|------|-----|--------|----------|-------|
| CIFAR-100 | 10 | 27.0 | 56.9 | 63.7 | 67.9 | **70.3** | +2.4 |
| CIFAR-100 | 50 | 50.2 | 66.8 | 68.6 | 67.9 | **70.3** | +1.7 |
| Tiny-ImageNet | 10 | 16.1 | 41.9 | 51.2 | 52.8 | **56.2** | +3.4 |
| ImageNette | 10 | 29.4 | 61.4 | - | 69.0 | **72.5** | +3.5 |
| ImageNet-1K | 10 | 21.3 | 42.0 | 48.6 | 50.9 | **53.2** | +2.3 |
| ImageNet-1K | 50 | 46.8 | 56.5 | 58.0 | 61.2 | **64.3** | +3.1 |

RETA achieves the best results across all datasets × all IPC values × all backbones (ResNet-18/50/101).

### Ablation Study (ResNet-18, IPC=10)

| DRC | PTA | CIFAR-100 | Tiny-ImageNet | ImageNette | ImageNet-1K |
|-----|-----|-----------|---------------|------------|-------------|
| × | × | 67.9 | 52.8 | 69.0 | 50.9 |
| ✓ | × | 69.0 (+1.1) | 54.5 (+1.7) | 70.9 (+1.9) | 51.8 (+0.9) |
| × | ✓ | 68.5 (+0.6) | 53.7 (+0.9) | 69.8 (+0.8) | 51.6 (+0.7) |
| ✓ | ✓ | **70.3 (+2.4)** | **56.2 (+3.4)** | **72.5 (+3.5)** | **53.2 (+2.3)** |

### Key Findings

- **DRC contributes more**: DRC alone yields larger gains than PTA alone across all datasets (1.1–1.9 vs. 0.6–0.9), indicating that anchor selection is the more critical bottleneck.
- **Super-additive complementarity**: The combined gain of DRC+PTA exceeds the sum of their individual contributions (e.g., Tiny-ImageNet: 1.7+0.9=2.6, actual +3.4), demonstrating that the two modules address the coupled problem from complementary levels.
- **Cross-architecture generalization**: RETA surpasses FADRM+ on EfficientNet/MobileNet/ShuffleNet/Swin-Tiny/DenseNet (+0.9–3.4), confirming it is not architecture-specific.
- **Robustness**: Average +3.2% on ImageNet-Subset-C (15 corruption types × 5 severity levels), jointly contributed by DRC's noisy anchor filtering and PTA's stable topology preservation.
- **Moderate computational overhead**: Approximately 20% more time (1.31s vs. 1.09s/image) and 22% more memory (13.4GB vs. 11.0GB) relative to FADRM+, far lower than methods such as G-VBSM/EDC.
- **$\lambda$ sensitivity**: $\lambda=0.1$ is optimal (52.9%); $\lambda=0$ yields 51.7% and $\lambda=1.0$ yields 50.8%—a modest degree of complexity control is beneficial but should not be excessive.
- **$\lambda_{topo}$ sensitivity**: $\lambda_{topo}=0.5$ is optimal (52.5%); too small a value fails to counter the pull-to-anchor effect, while too large a value competes with the feature-matching objective.

## Highlights & Insights

- **Theory-driven method design**: Theorem 4.1's generalization bound decomposition directly leads to DRC's fit-complexity score, rather than being an ad-hoc design. The use of persistent homology to quantify class topology discrepancy also rests on a clear mathematical foundation—the paradigm of "theory → problem diagnosis → method design" is exemplary.
- **First application of TDA to dataset distillation**: Using persistence images from persistent homology as a differentiable topological regularization signal elegantly resolves the inherent non-differentiability of PH. This idea is transferable to any generation or compression task that requires preserving the topological structure of a dataset.
- **Simple yet effective complexity score**: $c(o) = \text{Var}(\|\nabla(G_\sigma * o)\|_2^2)$ captures "residual sharp edges" via the spatial variance of gradient magnitudes after Gaussian smoothing—simple, interpretable, and computationally lightweight.

## Limitations & Future Work

- **Acknowledged limitations**: The method relies on a frozen teacher, per-class retrieval pools, and multiple topological hyperparameters ($k$-NN construction, PI grid, $\lambda_{topo}$). The complexity proxy in DRC is hand-crafted, and PTA incurs additional overhead during distillation.
- **Learnable complexity proxy**: The current $c(o)$ is based on gradient statistics; a small learned network could serve as a more capable complexity predictor.
- **Layer-wise topology alignment**: Topology alignment is currently computed only in the final-layer teacher space; aligning topological structure across multiple feature layers—preserving structure in shallow layers and semantics in deep layers—warrants exploration.
- **Online pool updates**: The per-class pool is currently fixed; dynamically updating the pool (adding/removing candidate patches) during distillation could further improve retrieval quality.

## Related Work & Insights

- **vs. FADRM**: FADRM identifies the information collapse problem and introduces fixed residual connections, but anchor selection is random and invariant throughout training. RETA directly addresses FADRM's two systematic deficiencies (fit-complexity gap + pull-to-anchor effect), yielding +2–3.5% across all settings.
- **vs. RDED/EDC/SRe2L**: These methods do not employ a residual connection mechanism and perform markedly worse than FADRM/RETA on large-scale datasets. RETA's design builds on the residual matching paradigm and does not directly apply to non-residual methods.
- **vs. Trajectory Matching (MTT, etc.)**: Trajectory matching methods require extensive inner-loop training, incurring high computational cost. As a decoupled method, RETA maintains efficiency (single-GPU feasible) and substantially outperforms MTT in corruption robustness.

## Rating
- **Novelty**: ⭐⭐⭐⭐ In-depth problem diagnosis (two failure modes), a novel application of persistent homology to DD, and tight integration of theory and method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets × three IPC values × three backbones, plus cross-architecture evaluation, corruption robustness, ablation, hyperparameter sensitivity, continual learning application, and visualizations—comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with a coherent narrative from theory to method to experiments; the TDA section may pose a reading barrier for readers unfamiliar with the topic.
- **Value**: ⭐⭐⭐⭐ Achieves consistent state-of-the-art performance in dataset distillation; the design ideas behind DRC and PTA (adaptive anchor selection + topology preservation) demonstrate good generality.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](../../ACL2026/model_compression/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[CVPR 2026\] HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)

<!-- RELATED:END -->
