---
title: >-
  [Paper Note] Learning Generalizable Shape Completion with SIM(3) Equivariance
description: >-
  [NeurIPS 2025][LLM Evaluation][shape completion] This paper proposes SIMECO, the first SIM(3)-equivariant shape completion network. Through a three-stage modular design — feature canonicalization → similarity-invariant geometric reasoning → transformation recovery — SIMECO outperforms all augmentation-based and equivariant baselines under an unbiased evaluation protocol, achieving a 17% MMD reduction on KITTI and a 14% CD-$\ell_1$ reduction on OmniObject3D. Notably, SIMECO under the stricter protocol still surpasses competing methods evaluated under their own biased settings.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - shape completion
  - SIM(3) equivariance
  - vector neurons
  - point cloud
  - cross-domain generalization
date: 2026-05-08
content_hash: 9b31b49c2e8f5829
---

# Learning Generalizable Shape Completion with SIM(3) Equivariance

**Conference**: NeurIPS 2025
**arXiv**: [2509.26631](https://arxiv.org/abs/2509.26631)
**Code**: [Project Page](https://sime-completion.github.io)
**Area**: LLM Evaluation
**Keywords**: shape completion, SIM(3) equivariance, vector neurons, point cloud, cross-domain generalization

## TL;DR
This paper proposes SIMECO, the first SIM(3)-equivariant shape completion network. Through a three-stage modular design — feature canonicalization → similarity-invariant geometric reasoning → transformation recovery — SIMECO outperforms all augmentation-based and equivariant baselines under an unbiased evaluation protocol, achieving a 17% MMD reduction on KITTI and a 14% CD-$\ell_1$ reduction on OmniObject3D. Notably, SIMECO under the stricter protocol still surpasses competing methods evaluated under their own biased settings.

## Background & Motivation
**Background**: 3D shape completion aims to reconstruct complete shapes from partial point cloud observations. Dominant methods (PoinTr, AdaPoinTr, SeedFormer, etc.) perform well on benchmarks such as PCN, but all training data are pre-aligned to a canonical coordinate frame with fixed pose and scale.

**Limitations of Prior Work**:
- **Implicit bias leakage**: Pre-alignment allows networks to memorize absolute positions in the coordinate system rather than inferring intrinsic geometry; performance collapses once alignment is removed.
- **SO(3)/SE(3) equivariant methods are insufficient**: Existing equivariant methods still rely on ground-truth centroids and scales for input normalization, which is essentially explicit canonicalization rather than true equivariance.
- **Data augmentation is a symptomatic fix**: Random transformation augmentation approximates invariance but couples extrinsic transformations with intrinsic geometry, leaving ambiguity at test time.

**Key Challenge**: True generalization requires complete invariance to rotation, translation, and scale (the SIM(3) group), yet no existing shape completion architecture achieves full SIM(3) equivariance.

**Core Idea**: Enforce SIM(3) equivariance at every layer via a three-stage module: canonicalize to remove translation and scale → perform geometric reasoning in the invariant space → recover the original transformation.

## Method

### Overall Architecture
The input partial point cloud (2048 points) is passed through VN-DGCNN for local geometric feature extraction, followed by $L$ SIM(3)-equivariant Transformer layers (each comprising three stages: canonicalization $\mathcal{C}^l$ → geometric reasoning $\mathcal{A}^l$ → transformation recovery $\mathcal{R}^l$), and outputs a complete point cloud (16384 points). The backbone follows AdaPoinTr, with all layers replaced by their equivariant counterparts.

### Key Designs

1. **Feature Canonicalization $\mathcal{C}^l$ (translation and scale removal)**

    - **Function**: Transforms VN features into a translation- and scale-invariant canonical space.
    - **Mechanism**: Extends Layer Normalization — first subtracts the channel mean $\bar{V}_i$ to remove translation, then divides by the norm to remove scale, and finally applies standard LayerNorm on the norm for training stability. The formula is: $V'_i = \text{layernorm}(\|V_i - \bar{V}_i\|_2) \cdot \frac{V_i - \bar{V}_i}{\|V_i - \bar{V}_i\|_2}$
    - **Design Motivation**: Eliminating transformation variance prior to geometric reasoning ensures that attention weights depend solely on intrinsic geometry.

2. **SIM(3)-Invariant Geometric Reasoning $\mathcal{A}^l$**

    - **Function**: Reasons about missing geometry using Transformer attention in the canonicalized invariant space.
    - **Mechanism**: Employs rotation-invariant attention weights from VN-Transformer: $a_{ij} = \text{softmax}(\frac{1}{\sqrt{3D}} \langle W_Q V'_i, W_K V'_j \rangle_F)$. The Frobenius inner product is invariant to joint rotations of $V'_i$ and $V'_j$; combined with the canonicalization that removes translation and scale, the overall mechanism is SIM(3)-invariant.
    - **Design Motivation**: Attention weights reflect only the relative geometric relationships between points, fully decoupling intrinsic shape features from extrinsic transformations.

3. **Transformation Recovery $\mathcal{R}^l$**

    - **Function**: Maps reasoning results from the canonical space back to the original sensor coordinate frame.
    - **Mechanism**: Propagates pose and scale information from the input via residual connections. $V^{l+1} = V^l + \Phi(\mu^l Z)$, where $\mu^l$ is a global scale statistic computed from the average norm of centered input features, and $\Phi$ is a VN linear layer.
    - **Design Motivation**: SIM(3) equivariance only guarantees relative transformation consistency, but downstream tasks require absolute coordinates. Layer-wise recovery ensures that outputs reside in the original coordinate frame.

### Loss & Training
- Chamfer Distance $\ell_1$ loss (permutation-invariant)
- No data augmentation required during training (guaranteed by the equivariant architecture)
- Network depth and loss configuration follow AdaPoinTr for fair comparison

## Key Experimental Results

### Main Results (PCN Benchmark, Unbiased Evaluation)

| Method | Transform Setting | Mean CD-$\ell_1$ ↓ | F1 ↑ |
|--------|------------------|-------------------|------|
| AdaPoinTr (no augmentation) | I/SIM(3) | Collapse | — |
| AdaPoinTr + SIM(3) augmentation | SIM(3)/SIM(3) | ~9.2 | ~0.72 |
| EquivPCN (SO(3)) | I/SO(3) | Better under biased setting | — |
| ESCAPE (SE(3)) | I/SE(3) | Better under biased setting | — |
| **SIMECO (Ours)** | **I/SIM(3)** | **Best** | **Best** |

SIMECO improves over AdaPoinTr + augmentation by 10% in CD-$\ell_1$ and 8% in F1.

### Cross-Domain Generalization (Direct Transfer from PCN Training)

| Dataset | Metric | SIMECO | Best Baseline | Gain |
|---------|--------|--------|--------------|------|
| KITTI (real driving scans) | MMD ↓ | Best | — | **-17%** |
| OmniObject3D (indoor scans) | CD-$\ell_1$ ↓ | Best | — | **-14%** |

### Key Findings
- SIMECO under the stricter unbiased protocol (I/SIM(3)) outperforms competing methods evaluated under their own biased settings (e.g., I/SO(3), I/SE(3)), demonstrating that baseline methods exploit leaked alignment information rather than learning genuine geometry.
- Cross-domain generalization requires no additional canonicalization or fine-tuning — the generalization capacity of the equivariant architecture transfers directly.
- Qualitative results show that SIMECO recovers sharper geometric details (aircraft wings, lamp stems, table legs), whereas augmentation-based methods produce blurry or distorted shapes.

## Highlights & Insights
- **The most valuable contribution lies in problem identification**: The paper reveals that the "high performance" of existing shape completion methods is partly attributable to alignment information leakage rather than genuine geometric understanding. The introduction of an unbiased evaluation protocol is itself a significant contribution.
- **Elegance of the three-stage module**: The design paradigm of canonicalization → invariant reasoning → recovery is general and can in principle be applied to any 3D Transformer architecture requiring equivariance.
- **Striking conclusion — stricter protocol outperforms relaxed one**: SIMECO under the I/SIM(3) protocol surpasses baselines even under their biased settings, strongly demonstrating that architectural equivariance is superior to data augmentation.

## Limitations & Future Work
- The Vector Neurons-based architecture is relatively shallow, which may limit its capacity to model highly complex geometry.
- VN-DGCNN local feature extraction may not be sufficiently robust for extremely sparse point clouds.
- SIM(3) equivariance introduces additional implementation complexity and computational overhead; inference time is not reported.
- Training is conducted only on PCN (ShapeNet 8 categories), limiting the breadth of category-level generalization.

## Related Work & Insights
- **vs. AdaPoinTr + augmentation**: Augmentation can approximate invariance but introduces ambiguity and lacks theoretical guarantees; SIMECO achieves equivariance by design, yielding 10%+ improvement.
- **vs. SO(3)/SE(3) equivariant methods**: EquivPCN, ESCAPE, and similar methods still rely on ground-truth scale normalization, which is essentially explicit canonicalization rather than true equivariance.
- **vs. SCARP**: The two-stage approach of estimating canonical pose before completion is fragile under partial observations, and estimation errors propagate to the completion stage.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First complete SIM(3)-equivariant shape completion; the unbiased evaluation protocol is an additional contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on PCN, KITTI, and OmniObject3D with systematic comparison against diverse baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem motivation is clear, method derivation is rigorous, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Practically meaningful for real-world 3D completion; the unbiased evaluation protocol has the potential to advance the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](../../CVPR2026/llm_evaluation/unified_primitive_proxies_for_structured_shape_completion.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[ICCV 2025\] Combinative Matching for Geometric Shape Assembly](../../ICCV2025/llm_evaluation/combinative_matching_for_geometric_shape_assembly.md)

</div>

<!-- RELATED:END -->
