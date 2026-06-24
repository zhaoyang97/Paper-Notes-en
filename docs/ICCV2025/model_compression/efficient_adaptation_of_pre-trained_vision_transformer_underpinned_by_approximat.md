---
title: >-
  [Paper Note] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory
description: >-
  [ICCV 2025][Model Compression][parameter-efficient fine-tuning] This paper identifies that the row/column vectors of pre-trained ViT weight matrices exhibit approximate orthogonality, whereas the projection matrices learned by LoRA/Adapter do not. The authors propose AOFT, a strategy that generates approximately orthogonal down/up projection matrices from a single learnable vector, aligning the adaptation modules with the properties of the backbone network. This reduces the g…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "parameter-efficient fine-tuning"
  - "approximate orthogonality"
  - "LoRA"
  - "Adapter"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 90a9ca41bdd32b1c
---

# Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory

**Conference**: ICCV 2025
**arXiv**: [2507.13260](https://arxiv.org/abs/2507.13260)  
**Code**: [Google Drive](https://drive.google.com/file/d/1rg3JYfkmeLGDbRWXspO22wxVspbtnthV/view?usp=drive_link)  
**Area**: Model Compression
**Keywords**: parameter-efficient fine-tuning, approximate orthogonality, LoRA, Adapter, Vision Transformer

## TL;DR
This paper identifies that the row/column vectors of pre-trained ViT weight matrices exhibit approximate orthogonality, whereas the projection matrices learned by LoRA/Adapter do not. The authors propose AOFT, a strategy that generates approximately orthogonal down/up projection matrices from a single learnable vector, aligning the adaptation modules with the properties of the backbone network. This reduces the generalization error bound and achieves competitive performance on FGVC and VTAB-1k with fewer parameters.

## Background & Motivation
Parameter-efficient fine-tuning (PEFT) has become the dominant paradigm for adapting large-scale pre-trained ViTs to downstream tasks. Methods such as LoRA and Adapter approximate weight increments via low-rank down-projection–up-projection matrices, requiring updates to only a small number of parameters.

Through careful analysis of pre-trained ViT weight matrices $\mathbf{W}_q, \mathbf{W}_v$, etc., the authors observe an important and previously underexploited phenomenon:

**The row/column vectors of pre-trained backbone matrices exhibit approximate orthogonality**—their angular distribution concentrates near 90°.

**The down/up projection matrices learned by LoRA/Adapter do not possess this property**—their angular distribution is dispersed, far from orthogonal.

Orthogonality mathematically implies independence among vectors. From a generalization-theoretic perspective, orthogonal weight matrices have smaller L2 norms, which in turn reduces the generalization error bound given by Rademacher complexity.

The core question is: can endowing projection matrices with approximate orthogonality improve the generalization of fine-tuned models? AOFT answers affirmatively.

## Method

### Overall Architecture
AOFT is a general projection matrix substitution strategy that can be inserted into existing PEFT frameworks such as LoRA, Adapter, and VPT. The core idea is to use a single learnable vector $\vec{q} \in \mathbb{R}^N$ to generate an approximately orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{N \times N}$, from which the first $d$ columns are extracted as the down/up projection matrices.

### Key Designs
1. **Approximately Orthogonal Matrix Construction**

    - Function: Construct an orthogonal matrix $\mathbf{Q}$ from a single vector $\vec{q} = (q_0, q_1, \cdots, q_N)^\top$.
    - Mechanism: The construction of $\mathbf{Q}$ is based on a generalization of the Householder transformation. The $(i,j)$-th element of $\mathbf{Q}$ is defined as follows:
        - First row: $q_0, -q_1, -q_2, \cdots, -q_N$
        - Remaining rows: diagonal elements $1 - \frac{q_i q_i}{1+q_0}$, off-diagonal elements $-\frac{q_j q_i}{1+q_0}$
    - When the normalization constraint $\sum_{i=1}^N |q_i|^2 = 1$ is satisfied, $\mathbf{Q}$ is strictly orthogonal.
    - **Key relaxation**: This normalization is not strictly enforced, keeping the column vectors "approximately" orthogonal and enhancing model flexibility.
    - Operation definition: $\text{AO}(\vec{q}) = \mathbf{Q}[:, 0:d]$, taking the first $d$ columns.

2. **Integration of AOFT with Different PEFT Methods**

    - **LoRA + AOFT**: $\mathbf{X}_{FT}^{(l-1)} = \mathbf{X}^{(l-1)}(\mathbf{W}^{(l)} + \text{AO}(\vec{q}_{down}) \cdot \text{AO}(\vec{q}_{up})^\top)$
    - **Adapter + AOFT**: Appends $\text{AO}(\vec{q}_{down}^{MHA}) \cdot \text{AO}(\vec{q}_{up}^{MHA})^\top$ after each MHA and FFN block.
    - **VPT + AOFT**: Replaces prompt tokens with approximately orthogonal matrices.
    - Design Motivation: Since AOFT does not introduce additional parameters as the bottleneck dimension increases (only a single $N$-dimensional vector is required), the bottleneck size can be adjusted flexibly.

3. **AOFT* Variant: Learnable Scaling**

    - Function: Introduces a learnable scaling vector $\vec{\lambda}$ to further enhance flexibility.
    - Implementation: $(\mathbf{W}_{down} \odot \vec{\lambda}^\top) \mathbf{W}_{up}$
    - Provides independent scaling control over each rank component.

### Theoretical Analysis of Generalization Error
The generalization error bound is analyzed via Rademacher complexity:

$$\mathbb{E}\left[\frac{1}{m} \sup_{\|\mathbf{W}\| \leq \gamma} \left\| \sum_{i=1}^m \xi_i \mathbf{W} \vec{x}_i \right\|\right] \leq \frac{\gamma}{m} \mathbb{E}\left[\left\| \sum_{i=1}^m \xi_i \vec{x}_i \right\|\right]$$

where $\gamma$ is the L2 norm of the weight matrix. The projection matrices of AOFT exhibit significantly smaller L2 norms than those of LoRA/Adapter, yielding a tighter generalization error bound.

## Key Experimental Results

### Main Results
FGVC benchmark (5 datasets, ViT-B/16):

| Method | CUB-200 | NABirds | Flowers | Dogs | Cars | Mean | Params (M) |
|--------|---------|---------|---------|------|------|------|------------|
| Full fine-tuning | 87.3 | 82.7 | 98.8 | 89.4 | 84.5 | 88.5 | 85.98 |
| Adapter | 87.1 | 84.3 | 98.5 | 89.8 | 68.6 | 85.7 | 0.41 |
| **Adapter+AOFT*** | **89.0** | **84.5** | **99.5** | **92.0** | **85.2** | **90.1** | **0.20** |
| LoRA | 88.3 | 85.6 | 99.2 | 91.0 | 83.2 | 89.5 | 0.44 |
| **LoRA+AOFT** | **88.8** | 84.2 | **99.4** | **92.0** | **85.1** | **89.9** | **0.22** |
| VPT-Deep | 88.5 | 84.2 | 99.0 | 90.2 | 83.6 | 89.1 | 0.85 |
| **VPT-Deep+AOFT** | **88.7** | 82.8 | **99.5** | **91.5** | 84.1 | **89.5** | **0.15** |

### Ablation Study
VTAB-1k benchmark (19 datasets, 3 groups, ViT-B/16, partial results):

| Method | Natural Mean | Specialized Mean | Structured Mean | Overall Mean | Params (M) |
|--------|-------------|-----------------|----------------|--------------|------------|
| Full fine-tuning | 75.9 | 83.4 | 47.6 | 65.6 | 85.80 |
| Adapter | 79.0 | 84.1 | 58.5 | 71.4 | 0.16 |
| Adapter+AOFT | 79.3 | 84.2 | 60.6 | 72.5 | 0.06 |
| Adapter+AOFT* | 81.4 | 83.9 | 59.4 | 72.7 | 0.06 |
| LoRA | 79.5 | 84.9 | - | - | - |
| VPT-Deep+AOFT | 80.3 | 84.7 | 55.4 | 70.7 | 0.05 |

### Key Findings
- **Significant parameter efficiency gains**: Adapter+AOFT surpasses the original Adapter with only 0.20M parameters versus 0.41M, achieving higher accuracy (90.1 vs. 85.7).
- **Generalization validation**: After applying AOFT, the angular distribution of projection matrix column vectors concentrates near 90°, consistent with the pre-trained backbone.
- **Substantially reduced L2 norms**: The L2 norms of AOFT projection matrices are far smaller than those of vanilla LoRA/Adapter, empirically corroborating the theoretically predicted generalization advantage.
- **Flexible bottleneck adjustment**: AOFT introduces no additional parameters (only a single vector is needed), allowing adaptive bottleneck dimensions for different tasks.
- **Cross-framework generality**: All three PEFT frameworks—LoRA, Adapter, and VPT—benefit from AOFT.

## Highlights & Insights
- **A complete chain from observation to theory to method**: The discovery of the orthogonality phenomenon → Rademacher complexity theoretical analysis → AOFT method design → empirical validation forms a remarkably coherent research narrative.
- **"One vector generates one matrix"**: This minimalist design substantially reduces parameter count while preserving sufficient expressive power.
- **Universal enhancement for PEFT methods**: AOFT functions as a plug-and-play module that improves multiple PEFT methods.
- **No strict orthogonality enforcement**: Relaxing the normalization constraint allows the model to balance orthogonality and flexibility, reflecting sound engineering judgment.

## Limitations & Future Work
- The construction of orthogonal matrices relies on a specific mathematical form (generalized Householder transformation); alternative constructions may be worth exploring.
- LoRA+AOFT slightly underperforms vanilla LoRA on some datasets (e.g., NABirds), suggesting that the approximate orthogonality constraint may be overly restrictive in certain scenarios.
- Validation is limited to image classification tasks and has not been extended to dense prediction tasks such as detection and segmentation.
- A thorough theoretical comparison with methods that also employ orthogonal transformations, such as OFT and BOFT, is lacking.

## Related Work & Insights
- OFT preserves pre-trained semantics via orthogonal transformations, whereas AOFT introduces orthogonality from the perspective of generalization error, representing a distinct viewpoint.
- The finding that pre-trained weight matrices exhibit approximate orthogonality may hold theoretical value for understanding the training dynamics of large models.
- The idea of generating a matrix from a single vector may inspire other scenarios requiring structured parameterization.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation of approximate orthogonality and the single-vector approach to generating orthogonal matrices are genuinely original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 24 datasets across FGVC and VTAB-1k with three PEFT frameworks, though dense prediction experiments are absent.
- Writing Quality: ⭐⭐⭐ The theoretical analysis section is somewhat verbose, and notation could be made more concise.
- Value: ⭐⭐⭐⭐ Provides a general enhancement strategy for PEFT methods with both theoretical and practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](../../CVPR2025/model_compression/bhvit_binarized_hybrid_vision_transformer.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
