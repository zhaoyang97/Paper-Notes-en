---
title: >-
  [Paper Note] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection
description: >-
  [CVPR 2026][Medical Imaging][Few-shot anomaly detection] This paper proposes AnoPLe — a lightweight multimodal bidirectional prompt learning framework that requires neither manually crafted anomaly descriptions nor exter…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Few-shot anomaly detection"
  - "multi-class unified model"
  - "bidirectional prompt learning"
  - "scale-aware training"
  - "CLIP"
date: 2026-05-08
content_hash: 15ce552860ebc09c
---

# Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection

**Conference**: CVPR 2026
**arXiv**: [2408.13516](https://arxiv.org/abs/2408.13516)  
**Code**: N/A  
**Area**: Medical Imaging / Industrial Anomaly Detection
**Keywords**: Few-shot anomaly detection, multi-class unified model, bidirectional prompt learning, scale-aware training, CLIP

## TL;DR
This paper proposes AnoPLe — a lightweight multimodal bidirectional prompt learning framework that requires neither manually crafted anomaly descriptions nor external auxiliary modules. Through text–visual prompt bidirectional interaction and scale-aware prefixes, AnoPLe achieves few-shot multi-class anomaly detection, delivering strong competitive results on MVTec-AD/VisA/Real-IAD while maintaining efficient inference (~28 FPS).

## Background & Motivation
**Background**: Industrial anomaly detection is shifting from "one model per class" toward more practical **few-shot + multi-class** unified models. Few-shot MCAD requires: (a) only a few normal samples per class; (b) a single model covering multiple product categories.

**Limitations of Prior Work**:
   - WinCLIP relies on hand-crafted prompt templates, lacking flexibility.
   - PromptAD depends on class-specific anomaly descriptions (e.g., "broken fabric," "missing wire"); under multi-class settings, the description pool suffers semantic conflicts, and t-SNE visualizations reveal collapsed and aliased anomaly features.
   - IIPAD employs a large Q-Former to generate instance prompts, incurring substantial computational overhead.

**Core Observations**: (a) **Normality is shared across classes** (intact, uncontaminated, geometrically regular); (b) anomalies are highly **class-dependent**; (c) class names themselves serve as strong semantic priors (validated in the CCL paper).

**Core Idea**: Use only class names as textual anchors (without describing anomaly types), and leverage **bidirectional text–visual prompt interaction** to automatically learn class-aware, anomaly-type-agnostic representations.

## Method

### Overall Architecture
CLIP backbone → deep learnable text prompts + visual prompts → bidirectional matching interaction → scale-aware prefix (global + local training, global-only inference) → lightweight decoder producing pixel-level anomaly maps → anomaly scoring combined with a memory bank.

### Key Designs
1. **Text Prompt Learning**:

    - Normal: $\mathbf{e}_0^+ = [\texttt{class}]$; Anomalous: $\mathbf{e}_0^- = [\texttt{abnormal}][\texttt{class}]$
    - Shared learnable context vector $\mathbf{P}_0^t$ with layer-wise deep prompts.
    - **Design Motivation**: Using only the class name together with a unified "abnormal" token — the normal branch preserves a complete class prototype, while the anomalous branch is initialized from CLIP's implicitly encoded "non-normal" prior and subsequently refined through visual interaction.
    - **Key Distinction**: No hand-crafted anomaly descriptions such as "broken," "contaminated," or "missing" are required.

2. **Bidirectional Prompt Interaction**:
   At each layer $j$, learnable linear projections $f_{v\rightarrow t}$ and $f_{t\rightarrow v}$ fuse the two modalities:
    $$\tilde{\mathbf{P}}_j^t = [\mathbf{P}_j^t, f_{v\rightarrow t}(\mathbf{P}_j^v)], \quad \tilde{\mathbf{P}}_j^v = [\mathbf{P}_j^v, f_{t\rightarrow v}(\mathbf{P}_j^t)]$$
    - **Why Bidirectional**: Ablations show that unidirectional T→I (84.2% VisA I-AUC) and I→T (82.0%) are both inferior to bidirectional T↔I (**86.0%**). Text provides categorical structure; vision provides instance-level details; the two modalities are mutually complementary.

3. **Scale-Aware Prefix**:

    - Training: full-resolution image $I_0$ (240×240) + four non-overlapping sub-image crops $I_1,\ldots,I_4$ (480×480 crops).
    - Learnable prefix $c \in \mathbb{R}^{(N+1) \times d_v}$; inputs at different scales use the corresponding $c_i$.
    - **Inference uses the global prefix only**: zero additional inference cost.
    - Ablation: single-scale without prefix 89.9% → multi-scale without prefix 91.8% → multi-scale with prefix **94.5%**.

4. **Alignment Loss**:
    $$\mathbf{s} = \sum_{(i,j)} \hat{\mathbf{M}}_{ij} \circ D_{ij}(\mathbf{z}), \quad \mathcal{L}_{align} = 1 - \langle \mathbf{z}_0, \mathbf{s} \rangle$$
   Pixel-level anomaly evidence is aggregated with learned weighting and aligned with the global [CLS] representation, ensuring consistency between local and global decisions.

### Loss & Training
- $\mathcal{L} = \mathcal{L}_{pixel} + \mathcal{L}_{img} + \mathcal{L}_{align}$
- Pixel-level: Dice loss + Focal loss.
- Image-level: contrastive cross-entropy (with pseudo-anomalies generated via pixel-space and latent-space perturbations).
- At inference, memory-bank similarity scores and decoder outputs are fused via the harmonic mean.

## Key Experimental Results

### Main Results (1-shot Multi-Class AD)

| Method | MVTec I-AUC | MVTec P-PRO | VisA I-AUC | VisA P-PRO | Real-IAD I-AUC |
|--------|------------|------------|-----------|-----------|---------------|
| PatchCore | 66.5 | 66.9 | 69.8 | 70.0 | 59.3 |
| WinCLIP | 77.5 | 70.8 | 70.0 | 61.2 | 69.4 |
| PromptAD | 91.2 | 86.1 | 82.4 | 77.8 | 52.2 |
| INP-Former | 94.7 | 90.7 | 84.0 | 84.0 | **84.4** |
| **AnoPLe** | 94.5 | **90.8** | **86.0** | **87.5** | 81.2 |

### Ablation Study

| Configuration | MVTec I-AUC | VisA I-AUC | Note |
|---------------|-----------|-----------|------|
| Text prompt only | 93.0 | 81.7 | Lacks instance-level details |
| Visual prompt only | 90.3 | 82.0 | Lacks categorical structure |
| T→I unidirectional | 93.5 | 84.2 | Text guides vision |
| **T↔I bidirectional** | **94.5** | **86.0** | Bidirectional complementarity is optimal |
| w/o alignment loss | 93.7 | 86.0→73.3 (Real-IAD) | Large impact on Real-IAD |
| w/o multi-scale / prefix | 89.9 | 82.3 | Indispensable |

### Key Findings
- AnoPLe leads on VisA (the most challenging cross-class benchmark) with 86.0% I-AUC; its P-PRO of 87.5% substantially surpasses PromptAD's 77.8%.
- Unseen-class generalization (leave-one-class-out): AnoPLe drops only 6.3% on held-out MVTec classes, versus a 26.2% drop for INP-Former.
- Unseen anomaly type generalization: removing descriptions from PromptAD causes a large performance drop (−10.0 on screw), whereas AnoPLe is inherently unaffected.
- Inference speed ~28 FPS, significantly faster than IIPAD (which requires an additional Q-Former forward pass).

## Highlights & Insights
- **Elegance through simplicity**: Without anomaly descriptions or external large modules, AnoPLe reaches state-of-the-art performance using only class names and bidirectional interaction.
- The scale-aware prefix design — multi-scale during training, global-only at inference — elegantly resolves the efficiency–accuracy trade-off.
- The paper highlights the often-overlooked asymmetry: normality is shared across classes, whereas anomalies are class-dependent.
- The alignment loss contributes most significantly on large-scale multi-class data (Real-IAD, 30 categories).

## Limitations & Future Work
- AnoPLe trails INP-Former by ~3% on Real-IAD; purely visual methods may be better suited to certain scenarios.
- The CLIP ViT-B/16+ backbone sets a performance lower bound; larger backbones may yield further gains.
- The pseudo-anomaly generation strategy (pixel-space and latent-space perturbations) may not closely approximate real defects.
- The zero-shot setting (no normal samples whatsoever) remains unexplored.

## Related Work & Insights
- Aligned with the CCL perspective (class-aware contrastive learning): class semantics serve as a strong prior for organizing multi-class representations.
- The bidirectional prompt interaction idea is generalizable to other VLM adaptation tasks (e.g., medical image analysis).
- The scale-aware training strategy offers transferable value to any task requiring multi-scale reasoning under efficiency constraints.
- Successful generalization to medical domains (e.g., fundus images, X-rays) validates the framework's universality.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Bidirectional prompt interaction and scale-aware prefix design are novel, though the overall approach follows the VLM prompt tuning paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three major benchmarks, multiple shot settings, generalization experiments, t-SNE visualizations, and attention analysis are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; comparison with prior methods is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical — lightweight, efficient, free of expert knowledge, and well-suited for real-world industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[AAAI 2026\] MPA: Multimodal Prototype Augmentation for Few-Shot Learning](../../AAAI2026/medical_imaging/mpa_multimodal_prototype_augmentation_for_few-shot_learning.md)

</div>

<!-- RELATED:END -->
