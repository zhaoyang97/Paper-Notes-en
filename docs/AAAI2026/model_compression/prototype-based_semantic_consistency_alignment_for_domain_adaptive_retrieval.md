---
title: >-
  [Paper Note] Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval
description: >-
  [AAAI2026][Model Compression][domain adaptive retrieval] This paper proposes PSCA, a two-stage framework that establishes class-level semantic connections via orthogonal prototypes…
tags:
  - "AAAI2026"
  - "Model Compression"
  - "domain adaptive retrieval"
  - "hashing"
  - "prototype learning"
  - "pseudo-label correction"
  - "semantic alignment"
date: 2026-05-08
content_hash: 4200958fcc33dd5e
---

# Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval

**Conference**: AAAI2026
**arXiv**: [2512.04524](https://arxiv.org/abs/2512.04524)
**Code**: Not released
**Area**: Model Compression
**Keywords**: domain adaptive retrieval, hashing, prototype learning, pseudo-label correction, semantic alignment

## TL;DR
This paper proposes PSCA, a two-stage framework that establishes class-level semantic connections via orthogonal prototypes, dynamically corrects pseudo-label reliability through geometric-semantic consistency alignment, and learns hash codes on reconstructed features, achieving substantial improvements over existing methods on multiple cross-domain retrieval benchmarks.

## Background & Motivation

### State of the Field

Hash-based retrieval is widely adopted in image retrieval owing to its compact storage and efficient computation. Domain Adaptive Retrieval (DAR) extends this setting to cross-domain scenarios, transferring knowledge from a labeled source domain to an unlabeled target domain and supporting both cross-domain and single-domain retrieval. Prior methods such as PWCF employ focal-triplet constraints, DAPH applies MMD-based distribution alignment to mitigate domain discrepancy, while TSS, SGHL, and DCS-LSG introduce pseudo-labels for semantically guided alignment.

### Limitations of Prior Work

Existing DAR methods suffer from three key deficiencies: (1) **Over-reliance on pair-wise alignment** — methods such as PWCF, TSS, and SGHL minimize distributional discrepancy between semantically consistent sample pairs, incurring $O(n^2)$ complexity and failing to cover the full data distribution; (2) **Insufficient handling of pseudo-label reliability** — pseudo-label errors in the unlabeled target domain lead to biased alignment and degraded hash code quality; DCS-LSG relies solely on semantic consensus without cross-validation from geometric knowledge; (3) **Direct quantization of domain-shifted features** — mapping original features that contain domain shift directly into Hamming space introduces substantial quantization error.

### Root Cause

A fundamental tension exists between the semantic predictions of pseudo-labels and the geometric structure of the feature space. When the two are consistent they mutually reinforce one another, but blindly trusting semantic predictions under conflict causes error propagation. Existing methods either ignore this conflict entirely or assess label reliability from only a single perspective.

### Paper Goals & Starting Point

**Goal**: Design a framework that adaptively evaluates pseudo-label reliability by jointly leveraging geometric neighborhood structure and semantic predictions. **Key Insight**: Replace pair-wise alignment with class-level semantic connections established via orthogonal prototypes, and compare geometric distances against semantic predictions in prototype space to dynamically adjust label weights. **Core Idea**: After correcting pseudo-labels through geometric-semantic consistency alignment, perform hash encoding on reconstructed features rather than raw features, fundamentally improving encoding quality.

## Method

### Overall Architecture
PSCA adopts a two-stage framework. Stage 1 learns orthogonal prototypes and performs semantic consistency alignment to obtain a reliable soft membership matrix. Stage 2 uses the membership matrix and prototypes to reconstruct semantically enriched features, on which hash codes are then learned. The input consists of labeled source domain data $\mathcal{D}_s$ and unlabeled target domain data $\mathcal{D}_t$; the output is a unified binary hash code matrix $\mathbf{B} \in \{-1,1\}^{r \times n}$.

### Key Designs

1. **MMD Marginal Distribution Alignment + Orthogonal Prototype Learning**:

    - **Function**: Establish a domain-shared subspace and learn class-level semantic centroids.
    - **Mechanism**: A projection matrix $\mathbf{P}$ maps both domains into a shared $q$-dimensional subspace and reduces marginal distribution discrepancy via MMD. Simultaneously, $c$ orthogonal class prototypes $\mathbf{O} \in \mathbb{R}^{q \times c}$ (constrained by $\mathbf{O}^\top \mathbf{O} = \mathbf{I}_c$) are learned to compact intra-class samples while maximizing inter-class separation. The orthogonality constraint ensures that distinct prototypes are mutually orthogonal, guaranteeing maximal separability.
    - **Design Motivation**: Compared with pair-wise alignment ($O(n^2)$ complexity), class-level alignment maintains only $c$ prototypes, yielding computational efficiency while covering the complete class distribution.

2. **Semantic Consistency Alignment**:

    - **Function**: Dynamically assess and correct pseudo-label reliability to produce a soft membership matrix.
    - **Mechanism**: A soft membership matrix $\mathbf{R} \in \mathbb{R}^{n_t \times c}$ is constructed by fusing geometric distance $d_{ij} = \|\mathbf{P}^\top \mathbf{x}_{t_i} - \mathbf{o}_j\|_2^2$ with semantic pseudo-label information. An adaptive weight $\alpha_i$ dynamically modulates the contribution of the two signals: when the geometrically nearest prototype $k_\text{geo}$ agrees with the semantic prediction $k_\text{sem}$, the weight is adjusted according to the ratio of semantic to geometric margins; under conflict, the semantic contribution is down-weighted proportionally to the disagreement $|\pi_{i,k_\text{geo}} - \pi_{i,k_\text{sem}}|$.
    - **Design Motivation**: Addresses pseudo-label error propagation — mutual reinforcement when geometric and semantic signals agree, and automatic suppression of unreliable signals under conflict to prevent error accumulation.

3. **Feature Reconstruction Hashing**:

    - **Function**: Learn high-quality hash codes on semantically enriched reconstructed features.
    - **Mechanism**: Prototypes $\mathbf{O}$ and membership matrix $\mathbf{R}$ from Stage 1 are used to reconstruct semantically enhanced features $\widetilde{\mathbf{X}}$ (for the target domain: $\widetilde{\mathbf{x}}_{t_i} = \sum_m r_{im} \mathbf{o}_m^\top$), which are concatenated with projected features to form $\mathbf{D} \in \mathbb{R}^{2q \times n}$. Two domain-specific orthogonal quantization functions $\mathbf{W}_s, \mathbf{W}_t$ are then learned under a mutual approximation constraint $\|\mathbf{W}_s - \mathbf{W}_t\|_F^2$ to generate unified binary hash codes.
    - **Design Motivation**: Directly quantizing domain-shifted raw features introduces large quantization errors. Features reconstructed via prototypes possess stronger semantic discriminability while preserving the geometric structure of the projected space.

### Loss & Training
The overall objective includes an MMD alignment term, a prototype clustering term, an $\ell_{2,1}$-norm row-sparsity regularizer for feature selection, and a quantization loss. An alternating optimization strategy is employed to update $\mathbf{P}$, $\mathbf{O}$, $\mathbf{R}$, $\mathbf{W}_s$, $\mathbf{W}_t$, and $\mathbf{B}$ in turn. During the out-of-sample stage, a linear regressor $\mathbf{\Phi}$ efficiently maps new samples to hash codes.

## Key Experimental Results

### Main Results

Evaluation is conducted on MNIST→USPS, COIL1→COIL2, Office-31 (A→D, A→W), and Office-Home (6 cases) with code lengths ranging from 16 to 128 bits.

| Method | MNIST→USPS | COIL1→COIL2 | A→D | A→W |
|---|---|---|---|---|
| DCS-LSG | 59.88% | 85.70% | 64.59% | 57.13% |
| TSS | 73.88% | 87.55% | 45.23% | 53.23% |
| SGHL | 71.46% | 83.00% | 59.91% | 55.64% |
| **PSCA** | **88.71%** | **90.76%** | **67.41%** | **65.78%** |

Compared with deep methods at 128-bit: PSCA outperforms COUPLE by 15.89% on MNIST→USPS and exceeds CPH by an average of 1.98% on Office-Home.

### Ablation Study

| Variant | MAP@128bit (MNIST→USPS) | Description |
|------|------------------------|------|
| PSCA-v1 (w/o semantic fusion) | 61.38% | Geometry only; −27.33% |
| PSCA-v2 (w/o consistency alignment) | 83.24% | Hard pseudo-labels; −5.47% |
| PSCA-v3 (w/o prototypes) | 44.56% | No class-level structure; −44.15% |
| PSCA-v4 (w/o feature reconstruction) | 80.92% | Direct quantization of projected features; −7.79% |
| **PSCA (full model)** | **88.71%** | — |

### Key Findings
- **Prototype learning contributes most**: removing it (v3) causes MAP to drop from 88.71% to 44.56%, confirming that class-level semantic structure is the cornerstone of the framework.
- **Semantic fusion signal is indispensable**: v1 relying solely on geometric distance achieves only 61.38%, demonstrating the necessity of dual geometric-semantic signals.
- **Feature reconstruction yields significant gains**: v4 directly quantizing projected features loses 7.79%, validating the advantage of encoding on reconstructed features.
- Sensitivity analysis shows stable performance for $\lambda_1, \lambda_3$ within $[10^2, 10^4]$.

## Highlights & Insights
- **Class-level alignment over pair-wise alignment**: Orthogonal prototypes efficiently model semantic structure, eliminating $O(n^2)$ computational overhead while providing more robust semantic anchors than individual sample pairs.
- **Dual-signal pseudo-label correction**: Adaptive re-weighting under both agreement and conflict scenarios is a generalizable design principle applicable to any setting requiring multi-source signal fusion.
- **Two-stage bridging via feature reconstruction**: Reconstruction elegantly connects prototype learning with hash encoding, preventing information loss. This "correct-then-encode" paradigm offers transferable insights for other tasks requiring discretization.

## Limitations & Future Work
- The method is based on traditional machine learning rather than deep learning; linear projection may limit representational capacity on very large-scale, high-dimensional data.
- Single-domain retrieval improvement on A→D is modest (2.25%), suggesting that domain-shared prototypes may over-smooth fine-grained feature differences in the target domain.
- Hyperparameters $\lambda_1, \lambda_2, \lambda_3$ as well as $\sigma$ and $\alpha$ require per-dataset tuning and are stable only within $[10^2, 10^4]$.
- Comprehensive comparison with recent deep DAR methods (PEACE, CPH, COUPLE) across all code lengths and datasets is not provided.

## Related Work & Insights
- **vs. DCS-LSG**: Assesses pseudo-label reliability using semantic consensus alone, without geometric cross-validation. PSCA's dual-signal consistency alignment yields a 28.83% improvement on MNIST→USPS.
- **vs. TSS/SGHL**: Pair-wise semantic alignment is computationally expensive and sensitive to outliers; PSCA's class-level alignment is more efficient and robust.
- **vs. COUPLE (deep method)**: Uses graph flow diffusion for cross-domain knowledge transfer; PSCA, as a traditional ML method, surpasses this deep approach on several datasets, demonstrating the effectiveness of the prototype mechanism.
- The adaptive weighting design based on geometric-semantic consistency is transferable to pseudo-label selection in semi-supervised learning.

## Rating
- Novelty: ⭐⭐⭐⭐ — The geometric-semantic consistency adaptive pseudo-label correction mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four datasets, multiple code lengths, cross-domain and single-domain settings, ablation studies, and comparison with deep methods are all comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear; the two-stage logic is coherent.
- Value: ⭐⭐⭐ — Solid contribution to the DAR field, though the application scope is relatively narrow.

---

## Related Papers

- [\[NeurIPS 2025\] Mitigating Semantic Collapse in Partially Relevant Video Retrieval](../../NeurIPS2025/model_compression/mitigating_semantic_collapse_in_partially_relevant_video_retrieval.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[NeurIPS 2025\] AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees](../../NeurIPS2025/model_compression/admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)
- [\[AAAI 2026\] Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation](earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](../../ACL2025/model_compression/aligndistil_token_level_alignment.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Semantic Collapse in Partially Relevant Video Retrieval](../../NeurIPS2025/model_compression/mitigating_semantic_collapse_in_partially_relevant_video_retrieval.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[NeurIPS 2025\] AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees](../../NeurIPS2025/model_compression/admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)
- [\[AAAI 2026\] Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation](earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)
- [\[ICLR 2026\] Modality-free Graph In-context Alignment](../../ICLR2026/model_compression/modality-free_graph_in-context_alignment.md)

</div>

<!-- RELATED:END -->
