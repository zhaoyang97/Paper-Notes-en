---
title: >-
  [Paper Note] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Visual Prompting] This paper proposes APEX (Adaptive Prompt EXtraction), which adaptively retrieves input-specific visual prompts from a learnable prompt memory bank—rather than assigning a f…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Visual Prompting"
  - "Domain Adaptation"
  - "Domain Generalization"
  - "Medical Image Segmentation"
  - "Low-Frequency Contrastive Learning"
date: 2026-05-08
content_hash: 9ba027ab3329c46b
---

# From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2604.17455](https://arxiv.org/abs/2604.17455)  
**Code**: [https://github.com/cetinkayaevren/apex/](https://github.com/cetinkayaevren/apex/)  
**Area**: Medical Imaging
**Keywords**: Visual Prompting, Domain Adaptation, Domain Generalization, Medical Image Segmentation, Low-Frequency Contrastive Learning

## TL;DR

This paper proposes APEX (Adaptive Prompt EXtraction), which adaptively retrieves input-specific visual prompts from a learnable prompt memory bank—rather than assigning a fixed prompt per domain—and incorporates low-frequency contrastive learning (LFC) to enhance inter-domain discriminability, achieving significant improvements in medical image segmentation on both seen and unseen domains.

## Background & Motivation

**Background**: Visual Prompting (VP) has attracted considerable attention as a domain adaptation approach for medical image segmentation. VP methods introduce learnable parameters in the input image space; once optimized, they map target-domain data into a space compatible with pretrained models. Since the original model parameters remain frozen, catastrophic forgetting is naturally avoided.

**Limitations of Prior Work**: Existing VP methods (e.g., VPT, FVP, A2XP) optimize a single prompt per target domain and apply it uniformly to all images. This leads to two fundamental limitations: (1) intra-domain variability is ignored—images acquired from the same device may differ substantially in acquisition settings and pathological characteristics, making a single prompt overly coarse; (2) inter-domain variability is ignored—a prompt optimized for a specific domain generalizes poorly to data from different devices or institutions, limiting performance on unseen domains.

**Key Challenge**: A trade-off exists between expressiveness and generalizability—coarse domain-level prompts are easy to optimize but lack expressiveness, while fine-grained input-level prompts are more expressive but require a sophisticated retrieval mechanism.

**Goal**: To design an adaptive prompt extraction framework that dynamically combines the most suitable prompts for each individual input image while ensuring generalization to both seen and unseen domains.

**Key Insight**: Domain shift in medical images primarily originates from global appearance changes (contrast, brightness, hue), which are encoded in the low-frequency components of the frequency domain. Extracting domain-discriminative features from low-frequency information enables precise retrieval of matched prompts.

**Core Idea**: Construct a memory bank of diverse prompt vectors; employ a low-frequency spectrum-based domain feature encoder to query the memory bank; derive input-specific prompts via weighted combination; and enhance inter-domain discriminability through low-frequency contrastive (LFC) learning.

## Method

### Overall Architecture

The input image is transformed via FFT to extract low-frequency amplitude components, which are fed into the APEX module (domain feature encoder → prompt memory query → prompt decoder) to generate input-specific prompts. The prompt is applied to the original low-frequency amplitude via element-wise multiplication; the result is reconstructed via IFFT and passed to the frozen segmentation model. During optimization, only APEX parameters are updated; the segmentation model remains entirely frozen.

### Key Designs

1. **Adaptive Prompt Memory Retrieval**:

    - Function: Dynamically compose optimal prompts based on the domain characteristics of each input image.
    - Mechanism: The domain feature encoder $E^D$ extracts a $K$-dimensional feature vector $z_m^n$ from the low-frequency amplitude. The prompt memory $B \in \mathbb{R}^{J \times K}$ contains $J$ learnable prompt vectors (orthogonally initialized). Cosine similarities between $z_m^n$ and each $b_j$ are computed to obtain an addressing vector $a_m^n$; a weighted sum $z_m^{\prime n} = \sum_j a_{m,j}^n \cdot b_j$ yields the final prompt feature, which is decoded by $D^P$ to produce the spatial prompt.
    - Design Motivation: Weighted combination allows the system to flexibly compose existing knowledge at the feature level, producing adaptive capacity beyond any single stored prompt. Orthogonal initialization promotes diversity among memory slots.

2. **Low-Frequency Contrastive Learning (LFC)**:

    - Function: Enhance the inter-domain discriminability and intra-domain compactness of the domain feature encoder.
    - Mechanism: An auxiliary projection head is applied to domain features $z_m^n$ to obtain $z_m^{n,aux}$. The contrastive loss $\mathcal{L}_{LFC}$ attracts features from the same domain and repels those from different domains, with temperature parameter $\tau$ controlling similarity scaling. The auxiliary projection head is used only during training and discarded at inference.
    - Design Motivation: The domain feature encoder must simultaneously capture inter-domain differences (cross-domain discrimination) and intra-domain variation (fine-grained differences within a domain). Contrastive learning clusters same-domain features to capture shared domain characteristics while separating different domains to encode their distinctions.

3. **Low-Frequency Spatial Prompt Application**:

    - Function: Enable effective domain adaptation without disrupting anatomical structures.
    - Mechanism: The prompt is applied exclusively to the low-frequency amplitude components in the frequency domain via element-wise multiplication; high-frequency components and phase information remain unchanged.
    - Design Motivation: Domain shift in medical images is primarily reflected in low-frequency components (contrast, brightness, and other global appearance properties), while high-frequency components and phase encode fine anatomical details and spatial layout. Applying prompts in the low-frequency space effectively addresses domain shift while preserving the structural information essential for segmentation.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{Seg} + \mathcal{L}_{LFC}$. The segmentation loss combines Dice and Cross-Entropy. The memory $B$ is updated via backpropagation. The domain feature encoder is jointly optimized by both loss terms.

## Key Experimental Results

### Main Results

| Task | Backbone | Domain Type | Source Only | VPAD (Best Baseline) | APEX |
|------|----------|-------------|-------------|----------------------|------|
| Polyp Segmentation | UNet | Seen Avg | 81.43 | 82.39 | **83.75** |
| Polyp Segmentation | UNet | Unseen Avg | 55.16 | 56.31 | **58.03** |
| Polyp Segmentation | PraNet | Seen Avg | 81.44 | 82.45 | **83.75** |
| OC/OD | UNet | Seen Avg | 85.08 | 85.88 | **88.43** |
| OC/OD | UNet | Unseen Avg | 73.46 | 76.76 | **82.57** |

### Ablation Study

| Configuration | Dice (Seen) | Dice (Unseen) | Note |
|---------------|-------------|---------------|------|
| APEX (Full) | Best | Best | Complete method |
| w/o LFC | Decreased | Significantly decreased | Insufficient inter-domain discriminability |
| w/o Memory (single prompt) | Decreased | Significantly decreased | Reverts to conventional VP |
| Fixed prompt (non-adaptive) | Decreased | Decreased | Cannot handle intra-domain variation |

### Key Findings

- Improvements on unseen domains are particularly pronounced; for example, APEX achieves a 41.44% Dice gain on the RIM-ONE-r3 domain in the OC/OD task with UNet.
- LFC contributes most to unseen-domain generalization, indicating that domain feature discriminability is the key to generalization.
- As a plug-and-play module, APEX consistently improves performance across five different backbones, demonstrating the universality of the approach.
- A memory slot count of $J=150$ yields optimal performance in most settings.

## Highlights & Insights

- The paradigm shift from "one prompt per domain" to "one prompt per image" is conceptually clear and empirically effective. The memory-and-retrieval mechanism allows prompt components learned on a limited set of training domains to be freely combined, thereby generalizing to unseen domains.
- Applying prompts in the low-frequency amplitude space is an insightful design choice—it leverages the physical interpretability of the frequency domain (low frequency = global appearance = primary cause of domain shift) while preserving the high-frequency structural information most critical for segmentation.
- The plug-and-play nature of the method makes it highly practical for clinical deployment; any existing segmentation model can be paired with APEX to improve cross-domain performance.

## Limitations & Future Work

- Training APEX requires data from multiple domains; when the number of available domains is small, memory diversity may be insufficient.
- The low-frequency prompt assumption holds when domain shift is predominantly low-frequency; it may be less effective for high-frequency domain shifts (e.g., differences in resolution).
- Effectiveness on 3D medical images (CT/MRI volumetric data) has not been validated.
- A promising direction for future work is the introduction of an online prompt memory update mechanism to adapt to new domains encountered at test time.

## Related Work & Insights

- **vs. VPT/FVP/A2XP**: Conventional VP methods use fixed domain-level prompts, whereas APEX employs input-level adaptive prompts, with the advantage being especially pronounced on unseen domains.
- **vs. VPTTA**: VPTTA also uses a memory structure but is specifically designed for test-time adaptation, representing a different problem setting.
- **vs. Domain Adaptation Fine-tuning**: Fine-tuning modifies model parameters and risks catastrophic forgetting; APEX leaves all model parameters entirely unchanged.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of adaptive prompt retrieval and low-frequency contrastive learning is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two tasks, five backbones, four comparison methods, and full coverage of seen/unseen domains.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived; method is described in detail.
- Value: ⭐⭐⭐⭐⭐ The plug-and-play domain generalization solution is highly practical for medical image segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](../../ICCV2025/medical_imaging/progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)

</div>

<!-- RELATED:END -->
