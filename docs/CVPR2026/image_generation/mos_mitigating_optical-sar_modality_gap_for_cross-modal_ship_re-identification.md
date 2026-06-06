---
title: >-
  [Paper Note] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification
description: >-
  [CVPR 2026][Image Generation][Cross-modal ReID] The paper proposes the MOS framework to address optical-SAR cross-modal ship re-identification. It comprises two core modules: (1) MCRL…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Cross-modal ReID"
  - "Optical-SAR"
  - "Ship recognition"
  - "Diffusion bridge model"
  - "Modality alignment"
date: 2026-05-08
content_hash: 54ca012989ae67e9
---

# MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification

**Conference**: CVPR 2026
**arXiv**: [2512.03404](https://arxiv.org/abs/2512.03404)  
**Code**: Coming soon  
**Area**: Image Generation / Cross-Modal Retrieval
**Keywords**: Cross-modal ReID, Optical-SAR, Ship recognition, Diffusion bridge model, Modality alignment

## TL;DR
The paper proposes the MOS framework to address optical-SAR cross-modal ship re-identification. It comprises two core modules: (1) MCRL, which reduces the modality gap during training via SAR image denoising and a category-level modality alignment loss; and (2) CDGF, which generates pseudo-SAR samples from optical images using a Brownian bridge diffusion model at inference time and fuses the resulting features. On the HOSS ReID dataset, MOS achieves a +16.4% R1 improvement in the SAR→Optical direction.

## Background & Motivation

**Background**: Ship ReID is critical for maritime surveillance and management. SAR sensors enable all-weather, all-day imaging but suffer from severe speckle noise. Optical-SAR cross-modal ReID is highly challenging due to the large modality gap, with only two pioneering works (TransOSS, SMART-Ship) existing in this area.

**Limitations of Prior Work**: (a) The fundamentally different imaging mechanisms of optical and SAR sensors lead to severe feature misalignment; (b) inherent SAR speckle noise significantly disrupts feature extraction; (c) models tend to favor intra-modal matching over correct cross-modal matching, as modality discrepancy dominates identity discrepancy.

**Key Challenge**: A conflict between closing the modality gap and preserving identity discriminability — both objectives must be satisfied simultaneously.

**Goal**: Mitigate the optical-SAR modality gap from two complementary perspectives: the training stage and the inference stage.

**Key Insight**: The observation that SAR noise concentrates in low-pixel-value regions, and that modality distribution alignment can be decomposed into two independent components — mean and variance.

**Core Idea**: Apply SAR denoising and category-level Wasserstein alignment during training, and perform diffusion-bridge-based cross-modal generation with feature fusion during inference.

## Method

### Overall Architecture
The dataset is defined as $\mathcal{D} = \{(I_i, y_i, m_i)\}$ with $m_i \in \{opt, sar\}$. The MCRL module learns modality-invariant representations during training, while the CDGF module generates cross-modal samples and fuses features at inference time.

### Key Designs

1. **SAR Image Denoising**:

    - Function: Remove inherent speckle noise from SAR images.
    - Mechanism: Sort all pixel values in ascending order, truncate the lowest $\alpha\%$ (corresponding to noise), and renormalize the remaining values to $[0, 255]$: $\hat{p}_k = \frac{255(p_k - p_{min})}{p_{max} - p_{min} + \epsilon}$
    - Design Motivation: Noise is empirically observed to concentrate in low-pixel-value regions, making simple truncation an effective remedy.

2. **Category-level Modality Alignment Loss (CMAL)**:

    - Function: Align the optical and SAR feature distributions within each identity class.
    - Mechanism: For each identity $c$, compute the class-wise optical and SAR centroids $\mu_{opt}^c, \mu_{sar}^c$ and variances $\text{var}_{opt}^c, \text{var}_{sar}^c$, then minimize $\mathcal{L}_{CMAL} = \frac{1}{|C|}\sum_{c\in C}(\|\mu_{opt}^c - \mu_{sar}^c\|_2^2 + \|\text{var}_{opt}^c - \text{var}_{sar}^c\|_2^2)$
    - Theoretical Derivation: Under a diagonal covariance approximation, this serves as a tractable surrogate for the Wasserstein-2 distance. The mean term pulls class centers together, while the variance term aligns intra-class dispersion.
    - Total Training Loss: $\mathcal{L} = \lambda_{id}\mathcal{L}_{ID} + \lambda_{tri}\mathcal{L}_{Triplet} + \lambda_{cmal}\mathcal{L}_{CMAL}$

3. **Cross-modal Data Generation and Feature Fusion (CDGF)**:

    - Function: Generate pseudo-SAR samples at inference time to assist retrieval.
    - Mechanism: A Brownian Bridge Diffusion Model (BBDM) is trained with forward process $q(x_t|x_0,y) = \mathcal{N}(x_t; (1-m_t)x_0 + m_t y, \delta_t I)$, where $x_0$ denotes the SAR latent feature and $y$ the optical feature. The reverse process learns to denoise, enabling SAR generation from optical inputs. At inference time, $K$ pseudo-SAR samples are generated and fused as: $f_{fused}^i = \frac{(1-\tau)f_{opt}^i + \tau(\frac{1}{K}\sum_{k=1}^K f_{pseudo}^{i,k})}{\|(1-\tau)f_{opt}^i + \tau(\frac{1}{K}\sum_{k=1}^K f_{pseudo}^{i,k})\|_2}$
    - Design Motivation: Aligning feature spaces alone is insufficient; generating a "view from the other modality" enriches cross-modal representations.

### Loss & Training
- Backbone: ViT (following the TransOSS baseline)
- $\lambda_{id} = \lambda_{tri} = 1$
- The BBDM is trained independently for use during inference-time generation.

## Key Experimental Results

### Main Results on HOSS ReID

| Method | Type | ALL2ALL mAP/R1 | O→SAR mAP/R1 | SAR→O mAP/R1 |
|--------|------|----------------|--------------|--------------|
| TransReID | Single-modal ReID | 48.1/60.8 | 27.3/18.5 | 20.9/11.9 |
| DEEN | Cross-modal ReID | 43.8/58.5 | 31.3/21.5 | 27.4/22.4 |
| VersReID | Cross-modal ReID | 49.3/59.7 | 25.7/13.8 | 27.7/17.9 |
| TransOSS | Optical-SAR | 57.4/65.9 | 48.9/33.8 | 38.7/29.9 |
| **MOS (Ours)** | Optical-SAR | **60.4/68.8** | **51.4/40.0** | **48.7/46.3** |

### Ablation Study

| Configuration | ALL R1 | O→SAR R1 | SAR→O R1 | Notes |
|---------------|--------|----------|----------|-------|
| Baseline TransOSS | 65.9 | 33.8 | 29.9 | No augmentation |
| + SAR Denoising | 66.5 | 35.4 | 32.8 | Denoising is effective |
| + CMAL | 67.6 | 38.5 | 40.3 | Core training-stage alignment |
| + CDGF | **68.8** | **40.0** | **46.3** | Generation fusion yields further gains |

### Key Findings
- The SAR→Optical direction yields the largest improvement (+16.4% R1), as CDGF generates pseudo-SAR matches for optical queries.
- CMAL is the central training-stage component: SAR→O R1 improves from 29.9 to 40.3.
- CDGF contributes an additional +6.0 points through inference-time augmentation.
- SAR denoising, though simple, consistently improves performance — low-pixel truncation is effective against speckle noise.
- The performance advantage over general cross-modal methods (CM-NAS, LbA, etc.) confirms that the optical-SAR domain requires task-specific designs.

## Highlights & Insights
- **Diagonal approximation of Wasserstein alignment**: The full $W_2$ matrix square-root computation is simplified to per-dimension mean and variance alignment, achieving high computational efficiency without sacrificing effectiveness. This simplification is transferable to general cross-domain alignment scenarios.
- **Training–inference two-stage synergy**: MCRL establishes a shared feature space during training, and CDGF further bridges the modality gap at inference time; the two modules are complementary.
- **Brownian bridge diffusion for cross-modal translation**: The endpoint-conditioned nature of BBDM naturally suits cross-modal mapping tasks.

## Limitations & Future Work
- The HOSS dataset is relatively small; generalizability to large-scale data remains to be validated.
- The denoising strategy is overly simplistic (pixel-value truncation); more advanced SAR denoising methods may yield larger gains.
- CDGF incurs inference overhead, as each query requires multiple diffusion sampling steps.
- Multi-scale feature fusion and hard sample mining are not discussed.

## Related Work & Insights
- **vs. TransOSS**: MOS augments TransOSS with dedicated modality alignment and cross-modal generation modules.
- **vs. face/pedestrian ReID methods**: The poor performance of general cross-modal methods in the optical-SAR domain highlights the need for domain-specific designs.
- **vs. GAN-based translation methods**: BBDM offers greater training stability and generates more diverse samples compared to CycleGAN and similar approaches.

## Rating
- Novelty: ⭐⭐⭐ The Wasserstein approximation and BBDM fusion are creative, though the individual components are relatively independent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-protocol evaluation with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations and experimental design are clearly presented.
- Value: ⭐⭐⭐ The application domain is narrow, but the contribution to optical-SAR ReID is concrete.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification](pose-dive_pose-diversified_augmentation_with_diffusion_model_for_person_re-ident.md)
- [\[CVPR 2026\] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)
- [\[CVPR 2026\] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization](diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)
- [\[AAAI 2026\] Multi-Aspect Cross-modal Quantization for Generative Recommendation](../../AAAI2026/image_generation/multi-aspect_cross-modal_quantization_for_generative_recommendation.md)
- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](../../ICLR2026/image_generation/uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)

</div>

<!-- RELATED:END -->
