---
title: >-
  [Paper Note] Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation
description: >-
  [AAAI 2026][Segmentation][Causal Analysis] This paper proposes Causal-Tune, a causality-driven VFM fine-tuning strategy that decomposes VFM features into causal (domain-invariant) and non-causal (domain-specific) components via DCT frequency-domain transformation and Gaussian band-pass filtering. Learnable tokens are applied exclusively to the causal components for refinement, effectively suppressing VFM artifacts and improving generalization in domain generalized semantic segmentation.
tags:
  - AAAI 2026
  - Segmentation
  - Causal Analysis
  - VFM Fine-tuning
  - Frequency Domain Decomposition
  - Band-pass Filtering
  - Domain Generalized Segmentation
date: 2026-05-08
content_hash: 1b046fe6682eb4e7
---

# Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation

**Conference**: AAAI 2026
**arXiv**: [2512.16567](https://arxiv.org/abs/2512.16567)
**Code**: [https://github.com/zhangyin1996/Causal-Tune](https://github.com/zhangyin1996/Causal-Tune)
**Area**: Semantic Segmentation / Domain Generalization
**Keywords**: Causal Analysis, VFM Fine-tuning, Frequency Domain Decomposition, Band-pass Filtering, Domain Generalized Segmentation

## TL;DR
This paper proposes Causal-Tune, a causality-driven VFM fine-tuning strategy that decomposes VFM features into causal (domain-invariant) and non-causal (domain-specific) components via DCT frequency-domain transformation and Gaussian band-pass filtering. Learnable tokens are applied exclusively to the causal components for refinement, effectively suppressing VFM artifacts and improving generalization in domain generalized semantic segmentation.

## Background & Motivation

**Background**: Vision foundation models (VFMs) such as DINOv2 and CLIP have demonstrated strong performance in domain generalized semantic segmentation (DGSS) through parameter-efficient fine-tuning (PEFT). Representative methods such as Rein refine feature maps by inserting trainable parameters between layers.

**Limitations of Prior Work**: VFMs trained over extended large-scale pretraining produce feature artifacts that persist even after adapter-based fine-tuning (as visualized in Figure 1). Existing PEFT methods fine-tune features across all layers indiscriminately and are unable to suppress these redundant representations. Moreover, DGSS data contains both explicit (rain, snow, fog, night) and implicit (brightness, blur, noise, reflection) non-causal factors, with the latter largely overlooked.

**Key Challenge**: A fundamental tension exists between the powerful representational capacity of VFMs and their feature redundancy (artifacts)—artifacts encode domain-specific, non-causal information that impedes the exploitation of valuable cross-domain invariant representations.

**Goal**: To identify and disentangle causal from non-causal components in VFM features from a causal perspective, preserving causal components while discarding non-causal ones during fine-tuning, thereby improving domain generalization.

**Key Insight**: An empirical observation that non-causal factors (both explicit and implicit) are predominantly concentrated in the high- and low-frequency components of the DCT spectrum, while the intermediate frequency band retains cross-domain invariant structural and textural patterns (causal factors).

**Core Idea**: Apply DCT to transform per-layer VFM features into the frequency domain; use a Gaussian band-pass filter to separate causal from non-causal components; discard the non-causal components; refine the causal components in the frequency domain using causal-aware learnable tokens; and finally recover the spatial domain representation via iDCT.

## Method

### Overall Architecture
Causal-Tune modules are inserted between the frozen layers of the VFM. For each layer output feature $f_i$: (1) DCT transforms it to the frequency domain $F_i^{DCT}$; (2) Gaussian band-pass filtering separates the causal component $F_i^{cau}$ from the non-causal component $F_i^{n-cau}$; (3) the non-causal component is discarded, and causal-aware learnable tokens refine the causal component via an attention mechanism; (4) iDCT maps the result back to the spatial domain to produce the feature increment $\Delta f_i$.

### Key Designs

1. **DCT Frequency-Domain Causal/Non-Causal Separation**:

    - **Function**: Decomposes VFM features into domain-invariant and domain-specific components.
    - **Mechanism**: A 2D DCT is applied to the feature map to obtain its frequency-domain representation, followed by a Gaussian band-pass filter $G(u,v) = \exp(-\frac{u^2+v^2}{2R_H^2}) - \exp(-\frac{u^2+v^2}{2R_L^2})$ for decomposition. Frequencies below $R_L$ and above $R_H$ constitute non-causal components, while the intermediate band constitutes the causal component: $F_i^{cau} = F_i^{DCT} \cdot G(u,v)$.
    - **Design Motivation**: Experimental validation (Figure 2 visualization) demonstrates that various non-causal factors (noise, blur, brightness variation, etc.) are predominantly manifested at the spectral extremes, whereas the intermediate frequency band exhibits the greatest robustness to domain shift.

2. **Causal-Aware Learnable Token Refinement**:

    - **Function**: Refines the causal component in the frequency domain.
    - **Mechanism**: A set of learnable tokens $T_i^{cau}$ interacts with the causal frequency-domain features $F_i^{cau}$ via an attention mechanism to produce refined causal features $\hat{F}_i^{cau}$, which are then transformed back to the spatial domain via iDCT as a feature increment.
    - **Design Motivation**: Separating causal components alone is insufficient; further refinement is needed to amplify domain-invariant signals. Attention-based interaction adaptively emphasizes important causal frequency components, offering greater flexibility than simple linear transformations.

3. **Explicit Discarding of Non-Causal Components**:

    - **Function**: Completely eliminates domain-specific noise.
    - **Mechanism**: The non-causal component $F_i^{n-cau} = F_i^{DCT} \cdot (1-G(u,v))$ is directly discarded and excluded from all subsequent computation. This constitutes an explicit elimination strategy, which is more effective than implicitly training the model to ignore such components.
    - **Design Motivation**: Adversarial approaches have limited effectiveness against implicit non-causal factors; hard filtering in the frequency domain provides a more direct and thorough solution.

### Loss & Training
Standard cross-entropy loss for semantic segmentation. The VFM (DINOv2) is frozen; only the learnable tokens in Causal-Tune and the segmentation head are trained. The band-pass thresholds $R_L$ and $R_H$ are treated as hyperparameters.

## Key Experimental Results

### Main Results

| Method | Night | Snow | Fog | Rain | Avg (ACDC) |
|---|---|---|---|---|---|
| ResNet-ISW | 24.3 | 49.8 | 64.3 | 56.0 | 48.6 |
| Rein (DINOv2) | baseline | baseline | baseline | baseline | high |
| SET | competitive | competitive | competitive | competitive | competitive |
| **Causal-Tune** | **best** | **+4.8%↑** | **best** | **best** | **best** |

### Ablation Study

| Configuration | Performance | Note |
|---|---|---|
| FFT instead of DCT | degraded | DCT provides better causal/non-causal separation |
| No band-pass filtering (full-spectrum fine-tuning) | degraded | Non-causal components interfere with fine-tuning |
| Low-pass only / high-pass only | degraded | Both spectral extremes contain non-causal information |
| Band-pass + token refinement (full model) | best | Causal separation and refinement are complementary |

### Key Findings
- The most significant gain is observed under Snow conditions (+4.8% mIoU), as snow-induced domain shift primarily manifests in high-frequency texture—precisely the region removed by band-pass filtering.
- DCT is more suitable than FFT for causal/non-causal separation, as its real-valued frequency representation affords more interpretable frequency decomposition.
- Visualization confirms that Causal-Tune effectively eliminates feature artifacts from DINOv2 (Figure 1(c)).
- The method introduces minimal additional parameters, consistent with the efficiency principle of PEFT.

## Highlights & Insights
- **Causal Interpretation of VFM Artifacts**: Attributing feature artifacts produced by long-term VFM pretraining to non-causal factors opens a new perspective for PEFT design.
- **Interpretable Frequency-Domain Separation**: Visualization confirms that non-causal factors are indeed concentrated at the spectral extremes, providing clear physical intuition for the band-pass filtering approach.
- **Simplicity and Effectiveness**: The entire method adds only DCT/iDCT transforms, a single band-pass filter, and a small number of learnable tokens, yielding significant performance gains with minimal complexity.

## Limitations & Future Work
- $R_L$ and $R_H$ require manual tuning; different datasets may necessitate different frequency thresholds.
- The assumption that non-causal factors concentrate at the spectral extremes may not hold for certain domain shifts (e.g., color shifting).
- Validation is limited to semantic segmentation; effectiveness on detection, instance segmentation, and other tasks remains unexplored.
- Adaptively learning the filter parameters rather than using fixed Gaussian band-pass thresholds is a promising direction.

## Related Work & Insights
- **vs. Rein**: The first PEFT-based DGSS method, but it does not distinguish between causal and non-causal features; Causal-Tune performs explicit separation.
- **vs. SET**: Also operates in the frequency domain but uses FFT without causal analysis; Causal-Tune's DCT with band-pass filtering is more effective.
- **vs. MAD**: Eliminates implicit non-causal factors via data augmentation; Causal-Tune operates directly on the feature spectrum, providing a more direct solution.
- The paradigm of frequency-domain causal analysis is transferable to other VFM fine-tuning scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of causal perspective, frequency-domain separation, and VFM fine-tuning is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple cross-domain tasks, including weather conditions and urban scene transfer.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated with sufficient visualization support.
- Value: ⭐⭐⭐⭐ Provides meaningful guidance for the VFM PEFT community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/generalizable_knowledge_distillation_from_vision_foundation_models_for_semantic_.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)
- [\[AAAI 2026\] Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation](do_we_need_perfect_data_leveraging_noise_for_domain_generalized_segmentation.md)
- [\[ICCV 2025\] VSSD: Vision Mamba with Non-Causal State Space Duality](../../ICCV2025/segmentation/vssd_vision_mamba_with_non-causal_state_space_duality.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)

<!-- RELATED:END -->
