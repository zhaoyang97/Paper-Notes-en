---
title: >-
  [Paper Note] Closed-Loop Transfer for Weakly-supervised Affordance Grounding
description: >-
  [ICCV 2025][Image Restoration][weakly-supervised affordance grounding] This paper proposes LoopTrans, a closed-loop knowledge transfer framework that unifies exocentric and egocentric image activation via a shared CAM mo…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "weakly-supervised affordance grounding"
  - "closed-loop knowledge transfer"
  - "shared CAM"
  - "denoising distillation"
  - "exocentric-egocentric transfer"
date: 2026-05-08
content_hash: 55bb794517c90a5c
---

# Closed-Loop Transfer for Weakly-supervised Affordance Grounding

**Conference**: ICCV 2025
**arXiv**: [2510.17384](https://arxiv.org/abs/2510.17384)
**Code**: [https://github.com/nagara214/LoopTrans](https://github.com/nagara214/LoopTrans)
**Area**: Visual Understanding / Affordance
**Keywords**: weakly-supervised affordance grounding, closed-loop knowledge transfer, shared CAM, denoising distillation, exocentric-egocentric transfer

## TL;DR

This paper proposes LoopTrans, a closed-loop knowledge transfer framework that unifies exocentric and egocentric image activation via a shared CAM module, refines coarse activations into precise localizations using pixel-level pseudo-masks, and feeds egocentric localization results back to enhance exocentric knowledge extraction through denoising distillation, achieving state-of-the-art performance across all metrics on AGD20K.

## Background & Motivation

Affordance grounding aims not only to predict the actions an object can support, but also to precisely localize the specific regions that enable those actions (e.g., bicycle handlebars → push; handlebars + seat → ride). Under the weakly-supervised setting, models learn affordance knowledge solely from image-level interaction labels (e.g., "lie on"), transferring knowledge from exocentric (third-person perspective) interaction images to egocentric (object-centric perspective) images for localization.

Existing methods face two core challenges:

**Imprecise exocentric knowledge extraction**: Exocentric interaction images contain complex backgrounds; CAM activations frequently include human body parts and background regions, and attention disperses rather than focusing on interaction regions in complex scenes.

**Limitations of unidirectional transfer**:
   - Existing methods (Cross-view-AG, LOCATE, WSMA) all adopt a unidirectional pipeline: exocentric CAM activation → feature alignment → egocentric localization.
   - Cross-domain feature alignment relies on appearance similarity in exocentric interaction regions and fails when those regions are fully occluded by the human body (e.g., "lie on", "ride").
   - The object-centric nature of egocentric images (clean, background-free) has not been exploited to improve exocentric knowledge extraction.

## Method

### Overall Architecture

LoopTrans constructs a closed-loop knowledge transfer pipeline:

**Interaction → Activation** (Shared CAM) → **Activation → Localization** (pixel-level decoding) → **Localization → Activation** (denoising distillation)

These three stages form a closed loop: precise egocentric localization feeds back to enhance exocentric knowledge activation, while exocentric interaction knowledge is in turn passed to egocentric images via the shared CAM.

### Key Designs

#### 1. Unified Exocentric-Egocentric Activation (Shared CAM / $\Theta_{\text{SCAM}}$)
- **Function**: A single CAM module with shared parameters $\theta$ processes both exocentric and egocentric images simultaneously.
- **Mechanism**: Rather than using two separate CAM modules for each viewpoint, parameters are shared:

$$\mathcal{G}^{\text{exo}}, \mathcal{G}^{\text{ego}} = \Theta_{\text{SCAM}}(\{\mathcal{F}^{\text{exo}}, \mathcal{F}^{\text{ego}}\}; \theta)$$

The classification loss jointly maximizes confidence across both viewpoints:

$$\mathcal{L}_{\text{cls}} = -\sum_{i=1}^{N} \mathbb{I}(c_i = \hat{c}) \log(\sigma(z_i^{\text{exo}}) \cdot \sigma(z_i^{\text{ego}}))$$

- **Design Motivation**:
    - Egocentric images are object-centric and background-free; their activations naturally concentrate on object regions, helping exocentric CAM suppress human-body and background interference.
    - Shared parameters enforce cross-view consistency and reduce domain discrepancy.
    - Even when interaction regions in exocentric images are fully occluded, the shared CAM can identify affordance regions through egocentric activations.

#### 2. Region Activation to Pixel Localization
- **Function**: Refines coarse CAM activation regions into precise object-part-level localizations.
- **Mechanism**: A two-step procedure is employed:
    - **Activation to object parts**: Self-supervised ViT DINO features are used for unsupervised clustering, partitioning egocentric images into $K$ semantic parts $\{o_1,...,o_K\}$. The part with the highest IoU against the egocentric activation map $\mathcal{G}^{\text{ego}}_{\hat{c}}$ is selected as the pseudo-mask:

$$\mathcal{M}^{\text{ego}} = \arg\max_{o_k} \text{IoU}(o_k, \mathbb{I}(\mathcal{R}(\mathcal{G}^{\text{ego}}_{\hat{c}}) \geq \mu))$$

  - **Object parts to localization**: A pixel-level affordance decoder $\Theta_{\text{pixel}}$ is trained with dice loss and MSE loss supervision:

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum_{i,j} \mathcal{P}_{i,j,\hat{c}} \cdot \mathcal{M}^{\text{ego}}_{i,j}}{\sum_{i,j} \mathcal{P}_{i,j,\hat{c}} + \sum_{i,j} \mathcal{M}^{\text{ego}}_{i,j}}$$

- **Design Motivation**: An inherent limitation of CAM is that it highlights only the most salient region, failing to cover the complete interaction part. Semantically complete pseudo-masks are generated via DINO feature clustering, and a pixel-level decoder is then trained to achieve precise localization.

#### 3. Egocentric-to-Exocentric Denoising Distillation
- **Function**: Feeds precise egocentric localization back to the shared CAM to suppress background and human-body noise in exocentric images.
- **Mechanism**: $M$ noise-absorbing heads $\mathcal{G}^{\text{noise}}$ are added to the shared CAM:

$$f^{\text{exo}} = \text{GAP}(\mathcal{R}(\mathcal{G}^{\text{exo}}_{\hat{c}}) \circ \mathcal{F}^{\text{exo}})$$
$$f^{\text{pixel}} = \text{GAP}(\mathcal{R}(\mathcal{P}_{\hat{c}}) \circ \mathcal{F}^{\text{ego}})$$
$$\{f^{\text{noise}}_m\} = \text{GAP}(\mathcal{R}(\{\mathcal{G}^{\text{noise}}_m\}) \circ \mathcal{F}^{\text{exo}})$$

Denoising distillation loss:
$$\mathcal{L}_{\text{dill}} = \log(1 + \sum_{m=1}^{M} \exp((s^{\text{noise}}_m - s^{\text{pixel}})/\tau))$$

where $s^{\text{noise}}_m = \text{sim}(f^{\text{noise}}_m, f^{\text{exo}})$ and $s^{\text{pixel}} = \text{sim}(f^{\text{pixel}}, f^{\text{exo}})$.

- **Design Motivation**: Noise-absorbing heads explicitly isolate non-affordance context, aligning affordance activation features with clean egocentric localization features while pushing noise features away. This forms a positive feedback loop: precise localization → denoised activation → more precise localization.

### Loss & Training

Total loss: $\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dill}} \mathcal{L}_{\text{dill}} + \lambda_{\text{pixel}} \mathcal{L}_{\text{pixel}} + \lambda_{\text{corr}} \mathcal{L}_{\text{corr}}$

where $\mathcal{L}_{\text{corr}}$ aligns affordance correlations between exocentric and egocentric images. The model is trained end-to-end at input resolution 224×224, with $K=4$ clusters, SGD optimizer, learning rate 1e-3, on a single NVIDIA TITAN GPU.

## Key Experimental Results

### Main Results

Comparison on the AGD20K image benchmark:

| Method | AGD20K-Seen KLD↓ | SIM↑ | NSS↑ | AGD20K-Unseen KLD↓ | SIM↑ | NSS↑ |
|--------|------------------|------|------|---------------------|------|------|
| LOCATE (CVPR23) | 1.226 | 0.401 | 1.177 | 1.405 | 0.372 | 1.157 |
| WSMA (AAAI24) | 1.176 | 0.416 | 1.247 | 1.335 | 0.382 | 1.220 |
| INTRA (ECCV24) | 1.199 | 0.407 | 1.239 | 1.365 | 0.375 | 1.209 |
| **LoopTrans** | **1.088** | **0.445** | **1.322** | **1.247** | **0.403** | **1.315** |

On HICO-IFF: LoopTrans achieves KLD=1.399, SIM=0.379, NSS=1.226, surpassing WSMA by approximately 10.5%.

On video benchmarks (EPIC/OPRA), LoopTrans also achieves comprehensive superiority under both the weakly-supervised setting and the image-to-video generalization setting.

### Ablation Study

Module ablation on AGD20K-Seen:

| Shared CAM | Pixel Alignment | Denoising Distillation | KLD↓ | SIM↑ | NSS↑ |
|------------|----------------|------------------------|------|------|------|
| ✗ | ✗ | ✗ | 1.318 | 0.384 | 1.135 |
| ✓ | ✗ | ✗ | 1.259 | 0.409 | 1.179 |
| ✗ | ✗ | ✓ | 1.251 | 0.392 | 1.196 |
| ✓ | ✓ | ✗ | 1.149 | 0.425 | 1.266 |
| ✓ | ✗ | ✓ | 1.222 | 0.405 | 1.183 |
| **✓** | **✓** | **✓** | **1.088** | **0.443** | **1.322** |

### Key Findings

1. The shared CAM alone yields a +4.5% KLD improvement (Seen), effectively promoting knowledge extraction through cross-view synergy.
2. Pixel alignment further provides +8.7% improvement on top of the shared CAM, refining coarse activations into regionally complete localizations.
3. The denoising distillation mechanism contributes a +5.1% baseline improvement, effectively filtering background noise by establishing a closed-loop knowledge cycle.
4. The combination of all three modules surpasses their individual contributions summed independently, demonstrating synergistic gains from the closed-loop design.
5. The method handles occlusion scenarios (e.g., "sit on", "catch") significantly better than appearance-alignment-based approaches.

## Highlights & Insights

- **Closed-loop paradigm**: This work is the first to introduce bidirectional knowledge transfer in affordance grounding, breaking the conventional unidirectional exo→ego assumption.
- **Problem essence**: The paper recognizes that egocentric images (clean, object-centric) are an underexploited "free lunch" that can inversely assist exocentric knowledge extraction.
- **Denoising distillation**: The noise-absorbing head design is elegant in its simplicity—purifying affordance activations by explicitly isolating noise patterns.
- **Occlusion robustness**: The shared CAM enables handling of scenes where interaction regions are entirely occluded by the human body, a fundamental weakness of prior methods.

## Limitations & Future Work

- The clustering number $K=4$ is fixed; different objects have different numbers of parts (e.g., chair vs. knife), and adaptive determination of $K$ may yield further improvements.
- Pseudo-mask quality depends on the accuracy of DINO feature clustering, which may degrade for objects with uniform texture.
- The number of noise-absorbing heads $M$ is a hyperparameter with no dedicated ablation provided; excessive heads may lead to over-fragmentation of noise concepts.
- Validation is limited to the affordance grounding setting; whether the closed-loop transfer idea generalizes to other cross-domain tasks remains unexplored.
- The video extension employs only a simple LSTM without leveraging stronger temporal modeling approaches such as temporal attention.

## Related Work & Insights

- **Cross-view-AG** (CVPR22) and **LOCATE** (CVPR23) serve as the primary baselines, representing the progression of the unidirectional transfer paradigm.
- The inherent limitation of CAM (activating only the most salient region) is a bottleneck across multiple weakly-supervised tasks; this paper addresses it elegantly via a two-step strategy combining DINO clustering and pixel-level decoding.
- The closed-loop / mutual feedback idea has broad implications for multimodal learning—bidirectional enhancement may be achievable between any two modalities or domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The closed-loop knowledge transfer framework represents a conceptual breakthrough in the field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation on image and video benchmarks with 12 groups of detailed ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, intuitive figures, and well-articulated closed-loop pipeline.
- **Value**: ⭐⭐⭐⭐ Achieves comprehensive superiority over prior SOTA across all metrics; the closed-loop transfer idea has strong generalization potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](../../CVPR2026/image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](../../CVPR2026/image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)

</div>

<!-- RELATED:END -->
