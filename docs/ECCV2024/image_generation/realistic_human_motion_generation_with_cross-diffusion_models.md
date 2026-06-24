---
title: >-
  [Paper Note] Realistic Human Motion Generation with Cross-Diffusion Models
description: >-
  [ECCV 2024][Image Generation] Proposes the CrossDiff framework, which integrates 3D and 2D motion information through a unified encoding and cross-decoding mechanism. It leverages cross-diffusion to capture finer full-body motion details and supports learning 3D motion generation from in-the-wild 2D data.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 2ade1775ea782441
---

# Realistic Human Motion Generation with Cross-Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2312.10993](https://arxiv.org/abs/2312.10993)  
**Area**: Image Generation

## TL;DR

Proposes the CrossDiff framework, which integrates 3D and 2D motion information through a unified encoding and cross-decoding mechanism. It leverages cross-diffusion to capture finer full-body motion details and supports learning 3D motion generation from in-the-wild 2D data.

## Background & Motivation

- The demand for text-driven human motion generation has been increasing in fields such as gaming, VR, and robotics.
- Existing methods (MDM, MLD, T2M-GPT) rely solely on 3D motion information for training, ignoring subtle motion details.
- **Key Insight**: When using only 3D representations, models tend to focus on primary movements while neglecting local details (e.g., fingers, facial expressions). In contrast, 2D projections can amplify these subtle movements from different perspectives.
- Collecting high-quality 3D motion data is highly expensive, whereas 2D motion data can be easily extracted from videos.
- **Core Problem**: How to leverage the complementary information of 2D motion to enhance full-body details in 3D motion generation?

## Method

### Overall Architecture

CrossDiff consists of three core modules:

1. **Mixed Representations**: Orthogonally projects 3D motion data into four directions (front, left, right, back) to obtain corresponding 2D motion data.
2. **Unified Encoding**: Two independent encoders ($\mathcal{E}_{3D}$, $\mathcal{E}_{2D}$) handle 3D/2D motion noise respectively, coupled with a shared-weight encoder $\mathcal{E}_{share}$ to map them into a unified feature space.
3. **Cross-Decoding**: Independent 3D/2D decoders can output corresponding dimensions of motion from unified features of any dimension.

### Key Designs

**1. Cross-Diffusion Mechanism**

The framework generates four output paths: 3D→3D, 2D→3D, 3D→2D, and 2D→2D, achieving cross-dimensional noise reversal:

$$\hat{x}_{iD \to jD,0} = G_{iD \to jD}(x_{iD,t}, t, c) = \mathcal{D}_{jD}(\mathcal{E}_{share}(\mathcal{E}_{iD}(x_{iD,t}, t, c)))$$

**2. Two-Stage Training Strategy**

- **Stage I**: Concurrently learns the reverse diffusion processes for all four directions to establish mapping relationships between 2D and 3D motions.
  $$\mathcal{L}_{stage I} = \mathcal{L}_{3D \to 3D} + w_{23}\mathcal{L}_{2D \to 3D} + w_{32}\mathcal{L}_{3D \to 2D} + w_{22}\mathcal{L}_{2D \to 2D}$$
- **Stage II**: Finetunes solely using 3D generation loss to focus on 3D denoising while retaining the rich motion features learned from 2D.
  $$\mathcal{L}_{stage II} = \mathcal{L}_{3D \to 3D}$$

**3. Mixture Sampling**

During inference, denoising can first be performed in the 2D domain up to timestep $\alpha$, and then projected to the 3D domain via $G_{2D \to 3D}$ to continue denoising. This strategy utilizes richer motion details in the 2D domain to guide 3D generation.

**4. Learning 3D Motion from In-the-Wild 2D Data**

Utilizes the pretrained $G_{2D \to 3D}$ to generate pseudo 3D labels from 2D poses estimated from videos, enabling the finetuning of out-of-domain motion without 3D ground truth data.

### Loss & Training

Each path uses a simple reconstruction target:

$$\mathcal{L}_{iD \to jD} = \mathbb{E}_{t \sim [1,T]} \|x_{jD,0} - G_{iD \to jD}(x_{iD,t}, t, c)\|_2^2$$

## Key Experimental Results

### Main Results

Comparison with SOTA methods on HumanML3D and KIT-ML datasets:

| Method | R-Prec(top3)↑ | FID↓ | MM Dist↓ | DIV→ | FID-U↓ | FID-L↓ |
|------|-------------|------|----------|------|--------|--------|
| MDM | 0.611 | 0.544 | 5.566 | 9.559 | 0.825 | 0.840 |
| T2M-GPT | 0.775 | 0.141 | 3.121 | 9.722 | 0.145 | 0.607 |
| MLD | 0.772 | 0.473 | 3.196 | 9.724 | 0.541 | 0.553 |
| ReMoDiffuse | 0.795 | **0.103** | **2.974** | 9.018 | 0.125 | 0.565 |
| **CrossDiff** | 0.730 | 0.162 | 3.358 | 9.577 | **0.118** | **0.281** |

KIT-ML dataset:

| Method | R-Prec(top3)↑ | FID↓ | MM Dist↓ | FID-U↓ | FID-L↓ |
|------|-------------|------|----------|--------|--------|
| MDM | 0.396 | 0.497 | 9.191 | 0.925 | 0.973 |
| T2M-GPT | 0.745 | 0.514 | 3.007 | 0.602 | 0.715 |
| ReMoDiffuse | 0.765 | **0.155** | **2.814** | 0.205 | 0.644 |
| **CrossDiff** | 0.704 | 0.474 | 3.308 | **0.434** | **0.625** |

### Ablation Study

Impact of each component of CrossDiff on HumanML3D:

| Setting | R-Prec↑ | FID↓ | MM Dist↓ | DIV→ |
|------|---------|------|----------|------|
| MDM baseline | 0.611 | 0.544 | 5.566 | 9.559 |
| 50% 3D | 0.666 | 0.586 | 3.894 | 9.513 |
| 100% 3D | 0.685 | 0.224 | 3.690 | 9.445 |
| 50% 3D + 100% 2D | 0.672 | 0.422 | 3.708 | 9.345 |
| 100% 3D + 100% 2D | **0.730** | **0.162** | **3.358** | 9.577 |
| w/o Shared Encoder | 0.714 | 0.187 | 3.496 | 9.488 |
| w/ Shared Encoder | **0.730** | **0.162** | **3.358** | 9.577 |
| 1 View (Front) | 0.722 | 0.186 | 3.467 | 9.798 |

### Key Findings

1. **Comprehensive Lead in Upper and Lower Body FID Metrics**: CrossDiff achieves FID-U=0.118 (best) and FID-L=0.281 (best), demonstrating more balanced quality in full-body motion generation.
2. ReMoDiffuse and T2M-GPT obtain low upper-body FID but relatively high lower-body FID, indicating imbalanced motion generation.
3. Incorporating 2D data significantly reduces the FID (from 0.224 to 0.162), validating the effectiveness of cross-dimensional complementary information.
4. The shared encoder is necessary; removing it increases FID from 0.162 to 0.187.
5. In the user study, CrossDiff receives the highest preference in terms of motion vitality and diversity.
6. Out-of-domain 3D motions (e.g., pull-ups, cycling) can be successfully generated from UCF101 in-the-wild 2D data.

## Highlights & Insights

- **The core innovation lies in cross-dimensional complementarity**: 2D projections amplify subtle movements from different perspectives, supplementing local details often ignored by 3D representations.
- **Proposes separate FID evaluation metrics for the upper and lower body** (FID-U/FID-L) to analyze full-body motion generation quality at a finer granularity.
- **Mixture sampling strategy** allows flexible switching between 2D and 3D domains, theoretically supporting 3D motion generation from pure 2D noise.
- **Practical value**: A vast amount of 2D motion data can be extracted from videos, and this method substantially reduces the cost of collecting 3D motion data.

## Limitations & Future Work

- Underperforms compared to methods like ReMoDiffuse on traditional metrics (R-Precision, FID); its advantages primarily manifest in fine-grained motion quality.
- The KIT-ML dataset is dominantly "walking," which is unsuitable for demonstrating the method's superiority in capturing detailed motions.
- Depth ambiguity exists in 2D-to-3D mapping, leading to less precise root node estimation when trained purely on 2D data.
- The two-stage training pipeline increases training complexity.

## Rating

- Innovation: ⭐⭐⭐⭐ — Novel cross-dimensional diffusion mechanism
- Practicality: ⭐⭐⭐⭐ — Supports training with in-the-wild 2D data
- Performance: ⭐⭐⭐ — Traditional metrics are not dominant, but leads in fine-grained metrics
- Overall Rating: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)

</div>

<!-- RELATED:END -->
