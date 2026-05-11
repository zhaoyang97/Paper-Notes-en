---
title: >-
  [Paper Note] Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling
description: >-
  [ICCV 2025][Image Generation][dance generation] This paper introduces the SoulDance dataset (the first high-quality 3D dance dataset encompassing body, hand…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "dance generation"
  - "music alignment"
  - "whole-body motion"
  - "hierarchical residual vector quantization"
  - "cross-modal retrieval"
date: 2026-05-08
content_hash: a1a7eb68d82c3462
---

# Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling

**Conference**: ICCV 2025
**arXiv**: [2507.14915](https://arxiv.org/abs/2507.14915)
**Code**: [Project Page](https://xjli360.github.io/SoulDance)
**Area**: image_generation (3D Dance Generation)
**Keywords**: dance generation, music alignment, whole-body motion, hierarchical residual vector quantization, cross-modal retrieval

## TL;DR

This paper introduces the SoulDance dataset (the first high-quality 3D dance dataset encompassing body, hand, and facial motion) and the SoulNet framework (hierarchical residual vector quantization + music-aligned generative model + cross-modal retrieval), achieving the first whole-body 3D dance generation with coordinated facial expressions, body, and hand movements aligned to musical rhythm and emotion.

## Background & Motivation

**Core Problem**: Existing methods generate only body or body+hand dance motions and lack facial expression generation, resulting in incomplete emotional expressiveness. Additionally, insufficient cross-modal alignment between music and dance remains a persistent issue.

**Three Major Challenges**:

**Dataset Gap**: Existing dance datasets either contain only body motion (AIST++) or lack facial expressions (FineDance), and most are derived from video-based pose estimation with insufficient accuracy.

**Modeling Difficulty**: Complex hierarchical dependencies exist among body, hands, and face, and existing methods cannot effectively capture coordinated cross-part motion.

**Poor Cross-Modal Alignment**: Existing methods directly feed low-level audio features (e.g., Librosa, Jukebox features) into motion generators without explicit music-dance alignment mechanisms.

**Motivation**: To construct the first whole-body motion dataset using professional motion capture systems, while designing a hierarchical quantization and cross-modal alignment framework to address coordinated generation and music synchronization.

## Method

### Overall Architecture

SoulNet consists of three core components:

1. **HRVQ** (Hierarchical Residual Vector Quantization): Models cross-part dependencies among body, hands, and face via hierarchical residual quantization.
2. **MAGM** (Music-Aligned Generative Model): A Transformer-based model that generates dance token sequences aligned to music.
3. **MMR** (Music-Motion Retrieval Module): A pretrained cross-modal alignment prior for music-motion retrieval.

### Key Designs 1: HRVQ

Conventional VQ-VAE quantizes whole-body motion uniformly, leading to significant information loss and neglect of inter-part relationships. The core innovation of HRVQ is a **multi-level body–hand–face chained design**:

- Whole-body motion is decomposed into body $m^b$, hand $m^h$, and face $m^f$ components.
- At each RVQ level $v$, the body residual is quantized first to obtain $b^v$; $b^v$ is then passed as a hint to guide hand quantization yielding $h^v$; finally $h^v$ is passed as a hint to guide facial quantization yielding $f^v$.
- A transformation function $\mathcal{T}$ fuses upstream part information with the current part residual, enabling hierarchical dependency modeling.

$$b^v = VQ_b(r_b^v), \quad h^v = VQ_h(\mathcal{T}(r_h^v, b^v)), \quad f^v = VQ_f(\mathcal{T}(r_f^v, h^v))$$

$V=5$ residual quantization levels are employed, balancing reconstruction accuracy and generation efficiency.

### Key Designs 2: MAGM Two-Stage Generation

- **Stage 1**: A Transformer layer performs mask-and-predict on the base layer $T_0$ (concatenated body+hand+face tokens), incorporating music features to generate primary motion tokens.
- **Stage 2**: $V-1$ residual layers progressively predict residuals $T_{1:V}$ to refine motion details.
- At each stage, MMR alignment constraints are introduced via $\mathcal{L}_{\text{Align-body}}$ and $\mathcal{L}_{\text{Align-whole}}$, respectively.

### Key Designs 3: MMR Cross-Modal Retrieval

Inspired by CLIP, a pretrained music-motion alignment model is employed:

- **Motion Encoder**: Compresses motion sequences into latent codes $\mathbf{z}$.
- **Music Encoder**: Extracts audio features using the Jukebox encoder and maps them to the aligned space $\mathbf{c}$.
- Contrastive loss $\mathcal{L}_{\text{Align}}$ is used to pull matched music-dance pairs closer.
- Two MMR modules are trained: $\mathcal{L}_{\text{Align-body}}$ aligns body motion with beat, and $\mathcal{L}_{\text{Align-whole}}$ aligns whole-body dynamics with musical emotion.

### Loss & Training

**HRVQ Training**:

$$\mathcal{L}_{hrvq} = \|m - \hat{m}\|_1 + \alpha \sum_v \|r_b^v - sg[b^v]\|_2 + \beta \sum_v \|r_h^v - sg[h^v]\|_2 + \gamma \sum_v \|r_f^v - sg[f^v]\|_2$$

**MAGM Training** (two stages):

$$\mathcal{L}_{mask} = \sum_{t_i \in \text{mask}} -\log_\theta(T_0 | T_0^m, C) + \lambda_b \mathcal{L}_{\text{Align-body}}$$

$$\mathcal{L}_{res} = \sum_{i=1}^{V} -\log_\phi(T_{i:V} | T_0, C) + \lambda_w \mathcal{L}_{\text{Align-whole}}$$

where $\lambda_b = \lambda_w = 0.5$.

## Key Experimental Results

### SoulDance Dataset

| Property | Value |
|----------|-------|
| Total Duration | 12.5 hours |
| Frame Rate | 60 FPS |
| Dance Styles | 15 |
| Music Clips | 284 |
| Dancers | 5 professional dancers |
| Body/Hand/Face Included | ✓/✓/✓ (only holistic dataset) |
| Motion Capture Cameras | 15 optical cameras |
| Facial Parameters | ARKit 52-dim blendshapes |

### Main Results: Motion Reconstruction (MPJPE, mm↓)

| Method | Whole Body | Body | Hand | Face |
|--------|-----------|------|------|------|
| Vanilla VQ (512) | 137.130 | 97.660 | 129.518 | 4.013 |
| RVQ-5 | 108.983 | 71.831 | 106.836 | 2.379 |
| **HRVQ-5 (Ours)** | **83.679** | **47.895** | **85.085** | **1.153** |

HRVQ reduces whole-body reconstruction error by **23.2%** compared to RVQ.

### Main Results: Dance Generation (SoulDance)

| Method | FID↓ | Div↑ | MM↑ | MMR-MS↓ | BAS↑ | EAS↑ |
|--------|------|------|-----|---------|------|------|
| FACT | 1.008 | 0.646 | 0.656 | 0.685 | 0.221 | 0.358 |
| Bailando | 1.379 | 1.307 | 1.117 | 0.585 | 0.236 | 0.401 |
| EDGE | 2.619 | 0.723 | 0.745 | 0.716 | 0.241 | 0.246 |
| FineNet | 1.463 | 1.262 | 0.832 | 0.694 | 0.213 | 0.263 |
| **SoulNet** | **0.029** | **1.312** | **1.310** | **0.369** | **0.244** | **0.594** |

SoulNet achieves an FID of only **0.029** (vs. second-best 1.008), reduces MMR-MS by **46.3%** (vs. Bailando), and improves EAS by **48.1%** (vs. Bailando).

### Ablation Study: HRVQ + MMR

| Method | FID↓ | MMR-MS↓ |
|--------|------|---------|
| VQ-512 | 1.610 | 0.703 |
| RVQ-512 + MMR | 0.067 | 0.540 |
| HRVQ-512 | 0.048 | 0.418 |
| **HRVQ-512 + MMR** | **0.029** | **0.369** |

### Ablation Study: Alignment Losses

| $\mathcal{L}_{\text{Align-body}}$ | $\mathcal{L}_{\text{Align-whole}}$ | FID↓ | BAS↑ | MMR-MS↓ |
|---|---|------|------|---------|
| ✓ | ✓ | **0.029** | **0.244** | **0.369** |
| ✓ | ✗ | 0.031 | 0.242 | 0.387 |
| ✗ | ✓ | 0.042 | 0.237 | 0.372 |

$\mathcal{L}_{\text{Align-body}}$ primarily improves local feature alignment (FID, BAS), while $\mathcal{L}_{\text{Align-whole}}$ primarily enhances global structural alignment (MMR-MS).

### User Study (1–10 scale)

| Method | Overall | Body | Hand | Emotion | Alignment |
|--------|---------|------|------|---------|-----------|
| FineNet | 6.44 | 5.89 | 6.11 | 5.33 | 6.56 |
| **SoulNet** | **7.33** | **8.45** | **7.56** | **7.67** | **7.78** |

### Key Findings

- Increasing residual levels $V$ from 1 to 5 consistently improves performance; generation quality degrades beyond 5–6 levels.
- The MMR module has a limited effect on FID improvement but significantly enhances MMR-MS (music alignment).
- Inference speed is 0.086s (vs. FACT 1.782s, EDGE 1.521s), approximately **17–20× faster**.

## Highlights & Insights

1. **First Holistic Dance Dataset**: SoulDance is the first high-quality motion-capture dance dataset simultaneously containing body, hand, and facial motion, filling a critical gap in the field.
2. **Chained Hierarchical Quantization**: The body→hand→face chained information flow in HRVQ is an elegant design—using body motion as a hint to guide hand quantization, which in turn guides face quantization, reflecting the physical coordination principles inherent in dance.
3. **Dual-Granularity Alignment**: $\mathcal{L}_{\text{Align-body}}$ handles local beat alignment while $\mathcal{L}_{\text{Align-whole}}$ handles global emotional alignment, forming complementary supervision signals.
4. **Novel Evaluation Metrics**: EmotionAlign Score and MMR-Matching Score address the inadequacy of existing metrics in evaluating facial expression generation and fine-grained rhythmic alignment.
5. **High Inference Efficiency**: The mask-and-predict generation paradigm achieves an order-of-magnitude speedup over diffusion-based models.

## Limitations & Future Work

1. **Dataset Scale**: SoulDance contains only 12.5 hours of data across 15 dance styles, providing less coverage than FineDance's 22 styles.
2. **Limited Diversity on AIST++**: The paper acknowledges that SoulNet's Diversity metric is lower than some competing methods on the smaller AIST++ benchmark.
3. **Limited Facial Representation**: The use of ARKit 52-dim blendshapes rather than finer-grained facial models may result in information loss for subtle expressions.
4. **Unidirectional Chained Quantization**: The unidirectional body→hand→face chain cannot model the reverse influence of facial expressions on body motion.
5. **Evaluation Metric Dependency**: EAS relies on the accuracy of expression recognition algorithms, and MMR-MS depends on the quality of its own pretraining, introducing a risk of circular evaluation.

## Related Work & Insights

- **Bailando** (VQ-VAE + RL): The first work to introduce VQ-VAE into dance generation, but training is complex and whole-body modeling is absent.
- **EDGE** (Diffusion-based): Generates high-quality body dance via diffusion models, but ignores hands and face.
- **FineNet** (FineDance): The first method to jointly generate body and hand motion, but without facial expressions.
- **MoMask** (RVQ for motion): Introduced RVQ into human motion generation; this work extends it to hierarchical RVQ.
- **CLIP** (Contrastive Learning): The MMR module directly draws on CLIP's cross-modal alignment paradigm.
- **Insights**: The chained hierarchical quantization design is generalizable to other multi-part coordinated generation tasks, such as whole-body sign language generation and character animation.

## Rating

| Dimension | Score (1–10) | Remarks |
|-----------|-------------|---------|
| Novelty | 8 | First holistic dance dataset + novel HRVQ chained quantization design |
| Technical Depth | 8 | Complete three-module co-design with clear theoretical derivation of HRVQ |
| Experimental Thoroughness | 9 | Three datasets, multiple ablations, user study, and two new metrics—very comprehensive |
| Writing Quality | 8 | Clear paper structure with highly informative figures and tables |
| Value | 7 | Dataset and framework have direct application value for whole-body animation |
| **Overall** | **8.0** | Solid ICCV-level contribution combining dataset, methodology, and thorough evaluation |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Diffusion-based 3D Hand Motion Recovery with Intuitive Physics](diffusion-based_3d_hand_motion_recovery_with_intuitive_physics.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](../../AAAI2026/image_generation/diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [\[ICCV 2025\] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models](smgdiff_soccer_motion_generation_using_diffusion_probabilistic_models.md)

</div>

<!-- RELATED:END -->
