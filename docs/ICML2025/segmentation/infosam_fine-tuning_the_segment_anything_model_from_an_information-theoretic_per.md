---
title: >-
  [Paper Note] InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective
description: >-
  [ICML2025 Spotlight][Segmentation][SAM Fine-Tuning] This paper proposes InfoSAM, which designs a relationship compression and distillation framework based on Rényi mutual information for the Parameter-Efficient Fine-Tuning (PEFT) of SAM from an information-theoretic perspective, enhancing fine-tuning performance by compressing pseudo-invariant information and preserving domain-invariant relationships.
tags:
  - "ICML2025 Spotlight"
  - "Segmentation"
  - "SAM Fine-Tuning"
  - "Information Bottleneck"
  - "Knowledge Distillation"
  - "Parameter-Efficient Fine-Tuning"
  - "Rényi Mutual Information"
  - "Domain-Invariant Relations"
date: 2026-05-08
content_hash: b30368da7bceb2de
---

# InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective

**Conference**: ICML2025 Spotlight  
**arXiv**: [2505.21920](https://arxiv.org/abs/2505.21920)  
**Code**: [InfoSAM project page](https://github.com/zhangyuanhong/InfoSAM)  
**Area**: Image Segmentation  
**Keywords**: SAM Fine-Tuning, Information Bottleneck, Knowledge Distillation, Parameter-Efficient Fine-Tuning, Rényi Mutual Information, Domain-Invariant Relations

## TL;DR
This paper proposes InfoSAM, which designs a relationship compression and distillation framework based on Rényi mutual information for the Parameter-Efficient Fine-Tuning (PEFT) of SAM from an information-theoretic perspective, enhancing fine-tuning performance by compressing pseudo-invariant information and preserving domain-invariant relationships.

## Background & Motivation
- **Problem**: SAM performs exceptionally well in general segmentation but poorly in specialized domains like medical imaging, remote sensing, and agriculture, necessitating PEFT adaptation.
- **Limitations of Prior Work**: Existing PEFT methods (LoRA, Adapter, etc.) independently adjust parameters of each module, neglecting the implicit relationships between the encoder and decoder in pretrained models; traditional distillation methods focus on layer-wise feature alignment and lack guidance on inter-module relationships.
- **Key Insight**: The massive pretraining of SAM learns domain-invariant structural relations (e.g., geometric contours), but fine-tuning easily destroys these relations; meanwhile, not all relations are beneficial—pseudo-invariant features like color can interfere with generalization.
- **Goal**: How to **extract** domain-invariant relations from pretrained SAM? How to **transfer** them to the fine-tuned model?

## Method
InfoSAM consists of two complementary information-theoretic objectives, forming a "compression-distillation" framework:

### 1. Relation Module
- Input: Image encoder embedding $z_i^T \in \mathbb{R}^{B \times H \times W \times D}$ and mask decoder output token $z_m^T \in \mathbb{R}^{B \times N \times D}$.
- Obtains Q and K through LayerNorm + linear projection, calculates attention scores, and adds a residual connection:
$$S_\alpha = \frac{QK^\top}{\sqrt{D}} + z_m^T \cdot {z_i^T}^\top$$
- The output is normalized via $\ell_2$ normalization to obtain the relation representation $r^T = f^T(z_i^T, z_m^T; \theta)$.

### 2. Relation Compression Loss $\mathcal{L}_r$ (Intra-SAM)
- **Goal**: Minimize $\mathbf{I}_\alpha(z_i^T, z_m^T; r^T)$, serving as an information bottleneck to compress pseudo-invariant information.
- Based on Rényi $\alpha$-entropy ($\alpha=2$), utilizing the Frobenius norm to avoid eigenvalue decomposition:
$$\mathcal{L}_r = -\log_2 \|G_r^T\|_F^2 + \log_2 \|G_{imr}^T\|_F^2$$
- Where $G_{imr}^T = G_i^T \circ G_m^T \circ G_r^T$ (Hadamard product), and $G$ is the polynomial kernel Gram matrix.

### 3. Cross-Model Distillation Loss $\mathcal{L}_d$ (Inter-SAM)
- **Goal**: Maximize the mutual information $\mathbf{I}_\alpha(r^T; r^S)$ between the teacher relations $r^T$ and student relations $r^S$.
$$\mathcal{L}_d = \log_2 \|G_r^T\|_F^2 + \log_2 \|G_r^S\|_F^2 - \log_2 \|G_r^{TS}\|_F^2$$
- The teacher and student share the same relation module parameters $\theta$.

### 4. Total Loss Function
$$\mathcal{L} = \mathcal{L}_{ce} + \lambda_1 \mathcal{L}_r + \lambda_2 \mathcal{L}_d$$
- $\mathcal{L}_{ce}$ is the standard segmentation loss (weighted IoU + BCE).

## Key Experimental Results

### Table 1: Comparison of PEFT Methods (SAM ViT-B, 5 Datasets × 4 Domains)

| Method | CAMO $S_\alpha$↑ | ISIC Jac↑ | Kvasir $S_\alpha$↑ | Leaf IoU↑ | Road IoU↑ |
|------|:-:|:-:|:-:|:-:|:-:|
| SAM (zero-shot) | 79.7 | 61.0 | 71.4 | 37.6 | 7.2 |
| LoRA | 87.7 | 87.8 | 93.0 | 71.4 | 59.0 |
| Adapter | 88.2 | 87.7 | 93.4 | 74.4 | 60.5 |
| SU-SAM | 88.3 | 87.8 | 93.8 | 74.7 | 60.2 |
| **Adapter+Ours** | **88.6** | **88.0** | **94.4** | **75.6** | **61.4** |

### Table 2: Comparison of Distillation Methods (Compared with Adapter Student)

| Method | Kvasir $S_\alpha$↑ | Leaf IoU↑ | Road IoU↑ |
|------|:-:|:-:|:-:|
| Student (No Distillation) | 93.4 | 74.4 | 60.5 |
| TinySAM | 88.5 | 48.6 | 25.7 |
| MobileSAM | 92.5 | 71.9 | 59.2 |
| VID | 93.7 | 75.1 | 60.7 |
| **InfoSAM (Ours)** | **94.4** | **75.6** | **61.4** |

### Ablation Study

| $\mathcal{L}_r$ | $\mathcal{L}_d$ | Kvasir $S_\alpha$ | Leaf IoU | Road IoU |
|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | 93.4 | 74.4 | 60.5 |
| ✓ | ✗ | 93.6 (+0.2) | 75.2 (+0.8) | 61.0 (+0.5) |
| ✓ | ✓ | **94.4 (+1.0)** | **75.6 (+1.2)** | **61.4 (+0.9)** |

- Both losses make positive contributions, with $\mathcal{L}_d$ playing a major role in cross-domain distillation.
- Equally effective on SAM2 (Hiera-B+): Kvasir 94.5, Leaf 77.3, Road 61.3.

## Highlights & Insights
- **First Information-Theoretic SAM Adaptation Framework**: Introduces Information Bottleneck theory to SAM PEFT, offering a novel perspective and solid theoretical derivation.
- **Aligning Relations Instead of Features**: Rather than performing layer-wise feature alignment, it extracts and transfers domain-invariant relations between the encoder and decoder, preventing performance degradation from distillation when the teacher performs poorly in the downstream domain.
- **Plug-and-Play**: Orthogonal to PEFT methods like LoRA and Adapter, and independent of the SAM/SAM2 architectures.
- **Filtering Pseudo-Invariant Information**: Compresses domain-specific information such as color via the information bottleneck, preserving only domain-invariant information such as geometric structure.
- **Simplified Calculation via Rényi $\alpha=2$**: Replaces eigenvalue decomposition with the Frobenius norm to reduce computational overhead.

## Limitations & Future Work
- Limited performance gain: The improvement on some datasets (e.g., LoRA+Ours vs. LoRA) is around 0.5-1%, suggesting a potential ceiling for the benefits of domain-invariant relations.
- When the teacher model performs extremely poorly in the target domain (e.g., Road IoU is only 7.2%), positive transfer still occurs via distillation, but the magnitude is limited.
- Only validated in box/point prompt scenarios, without exploring text prompts or fully automatic segmentation.
- The Rényi entropy order $\alpha$ is fixed at 2, and the impact of different $\alpha$ values on performance is not explored.
- The relation module introduces extra parameters and computation, and the paper does not analyze these overheads in detail.
- Only validated on medium-sized datasets, lacking experiments on large-scale datasets (e.g., SA-1B subsets).
- Insufficient sensitivity analysis of the hyperparameters $\lambda_1, \lambda_2$.

## Related Work & Insights
- The combination of **Information Bottleneck + Distillation** can be extended to the PEFT of other foundation models (e.g., CLIP, DINOv2).
- Unlike mutual information-based distillation methods such as VID and IBD, InfoSAM focuses on **inter-module relations** rather than single-layer features.
- The concept of domain-invariant features originates from Domain Adaptive Segmentation (DAS), but is quantified and transferred using information theory for the first time.

## Rating
- Novelty: ⭐⭐⭐⭐ — First time applying information bottleneck theory to SAM PEFT, with a complete theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 8 datasets across 4 domains, SAM+SAM2, with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear information-theoretic formulations and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Provides a new perspective on SAM fine-tuning; the plug-and-play distillation scheme holds practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](../../AAAI2026/segmentation/segment_and_matte_anything_in_a_unified_model.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[CVPR 2025\] EdgeTAM: On-Device Track Anything Model](../../CVPR2025/segmentation/edgetam_on-device_track_anything_model.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](../../CVPR2025/segmentation/sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[AAAI 2026\] InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](../../AAAI2026/segmentation/infoclip_bridging_vision-language_pretraining_and_open-vocab.md)

</div>

<!-- RELATED:END -->
