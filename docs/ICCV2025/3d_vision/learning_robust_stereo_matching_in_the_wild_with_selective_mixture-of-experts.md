---
title: >-
  [Paper Note] Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts
description: >-
  [ICCV 2025][3D Vision][Stereo Matching] This paper proposes SMoEStereo, which integrates variable-rank MoE-LoRA and variable-kernel MoE-Adapter modules into a frozen Visual Foundation Model (VFM)…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Stereo Matching"
  - "Visual Foundation Model"
  - "Mixture-of-Experts"
  - "LoRA"
  - "Cross-Domain Generalization"
date: 2026-05-08
content_hash: 3d91c5f390717836
---

# Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts

**Conference**: ICCV 2025
**arXiv**: [2507.04631](https://arxiv.org/abs/2507.04631)  
**Code**: [GitHub](https://github.com/cocowy1/SMoE-Stereo)  
**Area**: 3D Vision
**Keywords**: Stereo Matching, Visual Foundation Model, Mixture-of-Experts, LoRA, Cross-Domain Generalization

## TL;DR

This paper proposes SMoEStereo, which integrates variable-rank MoE-LoRA and variable-kernel MoE-Adapter modules into a frozen Visual Foundation Model (VFM), combined with a lightweight decision network for selective activation of MoE modules, achieving scene-adaptive robust stereo matching with state-of-the-art performance on cross-domain and joint generalization benchmarks.

## Background & Motivation

Stereo matching is a core task in computer vision with broad applications in autonomous driving, robotic navigation, and augmented reality. Although recent learning-based methods perform well on standard benchmarks, their **cross-domain generalization remains limited**:

**Core Challenges**:

**Domain Shift**: Significant scene discrepancies and imbalanced disparity distributions exist across datasets.

**Noisy Feature Maps**: Domain shift can lead to noisy and distorted feature maps, degrading model robustness.

**Why VFMs?** Visual Foundation Models (e.g., DINOv2, SAM, DepthAnything), pretrained on large-scale diverse data, are capable of extracting robust and generalizable features. However, direct application raises two critical issues:

**Limited Zero-Shot Performance**: VFMs excel at semantic information extraction (segmentation/classification) but lack the discriminative features required for precise similarity measurement.

**Inflexible Fixed Fine-Tuning**: Fixed-rank LoRA or fixed-kernel CNN decoders apply uniform processing to all inputs, failing to adapt to the heterogeneity of in-the-wild scenes.

**Proposed Solution**: Both the rank of LoRA and the kernel size of CNN Adapters are made dynamically selectable as MoE experts, enabling scene-specific feature adaptation.

## Method

### Overall Architecture

Using RAFT-Stereo as the backbone, the feature extractor is replaced with a VFM, and two types of MoE modules along with a decision network are embedded into the ViT blocks:

**VFM Feature Extraction** → **MoE-LoRA Layer** (attention branch) + **MoE-Adapter Layer** (MLP branch) → Shallow CNN compression → Correlation volume → GRU iterative disparity update

### MoE-LoRA Layer (Variable-Rank Experts)

Each MoE-LoRA layer contains $M$ LoRA experts, each corresponding to a different matrix rank $r_i$:

$$x_{\text{out}} = W_{q,k,v} x_{\text{in}} + \sum_{i=1}^{M} R_L(x_{\text{in}}) \cdot E_L^i(x_{\text{in}})$$

where each expert $E_L^i(x) = W_i^{\text{up}} W_i^{\text{down}} x$, and the router selects optimal experts via Top-k selection:

$$R(x) = \text{Topk}\left(\frac{\exp(W^{\text{router}} x / \tau)}{\sum_k \exp(W^{\text{router}} x / \tau)}, k\right)$$

**Design Motivation**: Different scenes require different low-rank subspaces — low-rank suffices for simple scenes, while high-rank is needed for complex textured regions.

### MoE-Adapter Layer (Variable-Kernel Experts)

Multiple CNN Adapter experts with different convolutional kernel sizes $k$ are embedded to capture local geometric structures at varying receptive fields:

$$E_A^j(x) = \text{Conv}_{1\times1}^{\text{up}}(\text{Conv}_{k\times k}^j(\text{Conv}_{1\times1}^{\text{down}}(x)))$$

**Complementary Design**: The CNN branch emphasizes fine-grained local geometric details, while the LoRA path models long-range interactions, reducing D1 error by up to 30%.

### Decision Network (Selective Activation)

A lightweight MLP predicts binary activation decisions for each MoE layer, with Gumbel Softmax enabling end-to-end training:

$$\mathcal{L}_{\text{usage}} = \left(\frac{1}{L}\sum_{l=1}^L M_L^l - \gamma\right)^2 + \left(\frac{1}{L}\sum_{l=1}^L M_A^l - \gamma\right)^2$$

The hyperparameter $\gamma \in (0,1]$ controls the computational budget.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{disp}} + \lambda_1 \mathcal{L}_{\text{blc}} + \lambda_2 \mathcal{L}_{\text{usage}}$$

- $\mathcal{L}_{\text{disp}}$: L1 disparity loss with exponentially weighted iterative predictions
- $\mathcal{L}_{\text{blc}}$: MoE expert load balancing loss to prevent uneven expert utilization

## Key Experimental Results

### Cross-Domain Generalization (Main Results)

| Method | KIT12 Bad3.0 | KIT15 Bad3.0 | Middle Bad2.0 | ETH3D Bad1.0 |
|:---|:---:|:---:|:---:|:---:|
| RAFT-Stereo | 5.1 | 5.7 | 12.6 | 3.3 |
| LoS | 4.4 | 5.5 | 19.6 | 3.1 |
| Former-RAFT‡ (DAM) | 3.9 | 5.1 | 8.1 | 3.3 |
| **SMoEStereo (DAMV2)** | **4.22** | **4.86** | **7.05** | **2.10** |

### Ablation Study: Component Effectiveness

| Ablation Setting | Key Findings |
|:---|:---|
| Fixed LoRA vs. MoE-LoRA | MoE-LoRA improves performance through scene-adaptive rank selection |
| w/o Adapter vs. MoE-Adapter | Injecting inductive bias significantly improves geometric feature extraction |
| Full MoE vs. Selective MoE | Decision network reduces redundant computation while maintaining accuracy |
| VFM capacity comparison | ViT-Base achieves satisfactory results with only 1/3 the parameters of ViT-Large |

### Efficiency Comparison

| Method | Capacity | Memory (GB) | Time (s) | Extra Params (M) |
|:---|:---:|:---:|:---:|:---:|
| Former-RAFT (DAM)† | ViT-Large | 4.1 | 0.47 | 6.9 |
| **SMoEStereo (DAM)** | ViT-Base | **1.9** | **0.18** | **2.86** |

### Key Findings

1. SMoEStereo achieves state-of-the-art cross-domain generalization across all four benchmarks without dataset-specific adaptation.
2. Compared to the VFM-LoRA baseline, D1 error is reduced by up to 30%.
3. The decision network flexibly controls the computational budget ($\gamma$ from 0.3 to 1.0), accommodating different resource constraints.
4. On DrivingStereo adverse weather evaluation, the average D1 is reduced from 5.0 to 4.3.

## Highlights & Insights

1. **Heterogeneous MoE Design**: Unlike conventional homogeneous MoE, LoRA experts employ variable ranks while Adapter experts use variable kernel sizes, leveraging the complementarity of both rank and receptive field dimensions.
2. **Selective Activation**: The binary decision mechanism of the decision network achieves Pareto-optimal accuracy-efficiency trade-offs.
3. **Plug-and-Play**: SMoE can be integrated as a plugin module into most stereo matching networks.

## Limitations & Future Work

1. Router training requires sufficiently diverse training data to learn meaningful expert assignment strategies.
2. More extreme domain shifts (e.g., underwater or medical stereo imaging) remain unvalidated.
3. The additional parameters introduced by MoE, though modest, increase overall model complexity.

## Related Work & Insights

- **Robust Stereo Matching**: CFNet, CREStereo++, LoS, Selective-IGEV
- **VFM Fine-Tuning**: LoRA, AdaptFormer, VPT
- **MoE**: Sparse MoE, LoRA-MoE fusion

## Rating

- Novelty: ⭐⭐⭐⭐ — First to apply heterogeneous MoE with selective activation to VFM-based stereo matching
- Technical Depth: ⭐⭐⭐⭐ — Complete design comprising variable-rank LoRA, variable-kernel Adapter, and decision network
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive multi-benchmark, multi-VFM, multi-ablation, and efficiency evaluations
- Writing Quality: ⭐⭐⭐⭐ — Open-source code, clear efficiency advantages, suitable for practical deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[ICCV 2025\] BANet: Bilateral Aggregation Network for Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matching.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] MaskHand: Generative Masked Modeling for Robust Hand Mesh Reconstruction in the Wild](maskhand_generative_masked_modeling_for_robust_hand_mesh_reconstruction_in_the_w.md)
- [\[ICCV 2025\] Stereo Any Video: Temporally Consistent Stereo Matching](stereo_any_video_temporally_consistent_stereo_matching.md)

</div>

<!-- RELATED:END -->
