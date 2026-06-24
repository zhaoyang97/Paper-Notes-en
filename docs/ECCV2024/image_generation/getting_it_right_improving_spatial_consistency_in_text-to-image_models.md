---
title: >-
  [Paper Note] Getting it Right: Improving Spatial Consistency in Text-to-Image Models
description: >-
  [ECCV 2024][Image Generation][Text-to-Image] A systematic investigation of spatial relationship generation deficiencies in text-to-image models. Finding that existing vision-language datasets severely lack spatial descriptions, the authors construct the SPRIGHT dataset (~6 million images re-captioned with spatial relations). Fine-tuning with <500 multi-object images achieves SOTA on the T2I-CompBench spatial score (0.2133), representing a 41% improvement over the baseline.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Text-to-Image"
  - "Spatial Consistency"
  - "Dataset Construction"
  - "Synthetic Captioning"
  - "Efficient Fine-Tuning"
date: 2026-05-08
content_hash: 420de31df3e611f1
---

# Getting it Right: Improving Spatial Consistency in Text-to-Image Models

**Conference**: ECCV 2024  
**arXiv**: [2404.01197](https://arxiv.org/abs/2404.01197)  
**Code**: [Yes](https://spright-t2i.github.io/)  
**Area**: Image Generation / Text-to-Image  
**Keywords**: Text-to-Image, Spatial Consistency, Dataset Construction, Synthetic Captioning, Efficient Fine-Tuning

## TL;DR

A systematic investigation of spatial relationship generation deficiencies in text-to-image models. Finding that existing vision-language datasets severely lack spatial descriptions, the authors construct the SPRIGHT dataset (~6 million images re-captioned with spatial relations). Fine-tuning with <500 multi-object images achieves SOTA on the T2I-CompBench spatial score (0.2133), representing a 41% improvement over the baseline.

## Background & Motivation

### Problem Introduction

Current text-to-image diffusion models (e.g., Stable Diffusion, DALL-E 3) excel at generating high-quality images, but perform poorly in adhering to spatial relationships (e.g., "left/right", "above/below") described in the text. This is a common bottleneck across all T2I model variants (different text encoders, prior models, and inference strategies).

### Root Cause Discovery

**Data level**: Spatial relationship vocabularies are severely underrepresented in existing vision-language datasets. Although spatial prepositions are frequently used in everyday English, they are extremely scarce in annotations like COCO and LAION:
- In COCO captions, "left" appears in only 0.16% of annotations, and "right" in 0.47%.
- In LAION captions, "left" appears in 0.27%, and "above" in only 0.16%.

**Model level**: CLIP text encoders barely distinguish between spatially opposite prompts (e.g., "A above B" vs "B above A", with a cosine similarity of >0.92).

### Key Insight

Two key findings drive the approach of this paper:
1. The lack of spatial relationship data is one of the root causes of poor spatial capability in T2I models.
2. Training images with a **large number of objects** play a decisive role in improving spatial consistency—more objects imply more spatial relationships.

## Method

### Overall Architecture

The work consists of three parts:
1. **Dataset Construction**: Using LLaVA-1.5-13B to generate spatially focused synthetic captions (SPRIGHT dataset) for ~6 million images.
2. **Standard Fine-Tuning**: Fine-tuning Stable Diffusion v2.1 using a subset of SPRIGHT (~15k images).
3. **Efficient Fine-Tuning**: Discovering that SOTA can be achieved using only <500 multi-object images.

### Key Designs

#### 1. SPRIGHT Dataset Construction

LLaVA-1.5-13B is utilized to re-generate spatially focused captions for images from four datasets:
- CC-12M: 2.3 million images (filtering out images with resolution <768×768)
- Segment Anything (SA): 3.5 million images (naturally containing a large number of objects)
- COCO validation set: ~40,000 images
- LAION-Aesthetics: 50,000 images

**Results**: SPRIGHT increases the occurrence of "left" in COCO from 0.16% to 26.80%, "right" from 0.47% to 23.48%, and "front" from 3.39% to 41.68%.

**Quality Verification (Three-fold Evaluation)**:
- FAITHScore (40k pairs, GPT-3.5 decomposes into atomic claims + LLaVA verification): 88.9% overall, 83.6% for spatial relations.
- GPT-4(V) (444 images, 1-10 rating): LAION mean score 7.49, SA mean score 7.36.
- Human Annotation (3,000 images, 149 annotators): Accuracy of 66.57%.

#### 2. Standard Fine-Tuning Strategy

- Base Model: Stable Diffusion v2.1
- Training Set: 13,500 images (50% from LAION-Aesthetics and 50% from SA), Validation Set: 1,500 images.
- Each image is paired with both its original caption and the SPRIGHT spatial caption, randomly selected with a 50:50 ratio during training.
- Fine-tune both the U-Net and CLIP text encoder (freezing CLIP for the first 10k steps), with learning rate $5 \times 10^{-6}$, AdamW, batch size = 128, for 15k steps.

#### 3. Efficient Fine-Tuning Strategy (Core Discovery)

**Key Hypothesis**: Images with more of objects naturally contain more spatial relationships. The Recognize Anything Model is used to automatically detect the number of objects per image, and the training is grouped into bins by object count:

| Number of Objects | <6 | <11 | 11 | >11 | >18 |
|---------|-----|------|-----|------|------|
| Number of Training Images | 444 | 1346 | 1346 | 1346 | 444 |
| Spatial Score | 0.1309 | 0.1468 | 0.1667 | 0.1613 | **0.2133** |

**Conclusion**: Fine-tuning with only 444 images containing >18 objects is sufficient to reach SOTA.

### Loss & Training

Standard diffusion training loss (noise prediction), while fine-tuning both the U-Net and the CLIP text encoder.

## Key Experimental Results

### Main Results

**Standard Fine-Tuning Results (~15k images)**:

| Method | OA↑ | VISOR(uncond)↑ | VISOR(cond)↑ | VISOR1↑ | VISOR4↑ | Spatial Score↑ | FID↓ | CMMD↓ |
|------|-----|---------------|-------------|---------|---------|----------|------|-------|
| SD 2.1 | 47.83 | 30.25 | 63.24 | 64.42 | 4.70 | 0.1507 | 21.646 | 0.703 |
| **+SPRIGHT** | **53.59** | **36.00** | **67.16** | 66.09 | **9.13** | **0.1840** | **14.925** | **0.494** |

**Efficient Fine-Tuning SOTA (<500 images)**:

| Method | OA↑ | VISOR(uncond)↑ | VISOR(cond)↑ | VISOR1↑ | VISOR4↑ | Spatial Score↑ | FID↓ | CMMD↓ |
|------|-----|---------------|-------------|---------|---------|----------|------|-------|
| SD 2.1 | 47.83 | 30.25 | 63.24 | 64.42 | 4.70 | 0.1507 | 21.646 | 0.703 |
| **+SPRIGHT(<500)** | **60.68** | **43.23** | **71.24** | **71.78** | **16.15** | **0.2133** | **16.149** | **0.512** |

**Full Comparison on VISOR Benchmark**:

| Method | OA↑ | VISOR1↑ | VISOR4↑ |
|------|-----|---------|---------|
| GLIDE | 3.36 | 6.72 | 0.03 |
| DALLE-2 | 63.93 | 73.59 | 7.49 |
| Attend-and-Excite | 42.07 | 49.29 | 0.08 |
| **Ours (<500)** | **60.68** | **71.78** | **16.15** |

**GenEval Benchmark**:

| Method | Overall | Single Object | Two Objects | Counting | Position |
|------|---------|--------------|-------------|----------|----------|
| SD 2.1 | 0.50 | 0.98 | 0.51 | 0.44 | 0.07 |
| SDXL | 0.55 | 0.98 | 0.74 | 0.39 | 0.15 |
| **Ours (<500)** | **0.51** | **0.99** | **0.59** | **0.49** | **0.11** |

### Ablation Study

**Impact of Spatial Caption Ratio**:

| Spatial Caption Ratio | 25% | 50% | 75% | 100% |
|------------|------|------|------|-------|
| T2I-CompBench Spatial Score↑ | 0.154 | **0.178** | 0.161 | 0.140 |

50% is the optimal ratio. A 100% spatial caption ratio actually degrades performance because the model loses the capability to generate from general descriptions.

**Impact of Long/Short Captions**:

| Model/Setting | Long Captions | Short Captions |
|----------|--------|--------|
| SD 1.5, w/o CLIP FT | 0.0910 | 0.0708 |
| SD 2.1, w/o CLIP FT | 0.1605 | 0.1420 |
| SD 2.1, w/ CLIP FT | **0.1777** | 0.1230 |

Long captions consistently outperform short captions. Fine-tuning CLIP has a positive impact on long captions but is actually detrimental to short captions.

**Improvement in CLIP Semantic Understanding**:

| Spatial Relationship | "above" | "below" | "left of" | "right of" | "in front of" | "behind" |
|---------|---------|---------|-----------|------------|--------------|---------|
| Baseline CLIP | 0.9225 | 0.9259 | 0.9229 | 0.9223 | 0.9231 | 0.9289 |
| CLIP+SPRIGHT | **0.8674** | **0.8673** | **0.8658** | **0.8528** | **0.8417** | **0.8713** |

Fine-tuned CLIP distinguishes spatial semantic differences better (lower cosine similarity = higher discriminative power).

### Key Findings

1. **Lack of spatial words is the root cause**: Spatial description words are extremely scarce in existing datasets; SPRIGHT increases the frequency of spatial phrases by 10-100 times.
2. **Object count is the key driver**: Training on 444 images with >18 objects/image outperforms training on 1346 images with <11 objects/image (0.2133 vs 0.1468).
3. **50% spatial caption ratio is optimal**: Too many spatial captions degrade the general generation capability of the model.
4. **CLIP fine-tuning is effective for long captions**: Long captions (~68 tokens) are out-of-distribution (OOD) data for CLIP; fine-tuning helps CLIP adapt.
5. **Improved attention maps**: After fine-tuning, the model correctly directs attention from spatial words ("below", "right") to the corresponding regions of the image.
6. **Preliminary exploration of negation training**: Replacing descriptions with negations (e.g., using "not left" to represent "right") yields slight but limited improvements.

## Highlights & Insights

- The **systematic diagnosis** is highly valuable: it quantitatively proves from a data perspective that the lack of spatial vocabulary is a root cause of capability deficiency.
- The discovery that **fine-tuning with <500 images achieves SOTA** is highly practical, revealing that data quality is far more important than quantity.
- The **object count hypothesis** is novel and empirically strong: more objects → more spatial relationships → better spatial learning.
- Canonical Correlation Analysis (CKA) across CLIP layers reveals the key role of MLP and output attention projection layers in spatial understanding.
- Comprehensive ablations (caption ratio, long/short captions, CLIP fine-tuning, negation training) provide a rich reference for future research.

## Limitations & Future Work

- SPRIGHT relies on LLaVA-1.5-13B generation, which suffers from LLM hallucinations (human annotation accuracy is only 66.57%).
- Experiments are only conducted on Stable Diffusion v2.1; the transferability to newer models like SDXL/SD3 remains unknown.
- The generalizability of the efficient fine-tuning strategy (>18 objects) is yet to be verified—performance may vary across different domains/styles.
- Negation understanding remains a major challenge, showing limited improvement after fine-tuning.
- Inherent limitations of CLIP text encoders persist, as the upper bound of spatial reasoning is constrained by the encoder architecture.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Approaches the spatial consistency problem from a data perspective, presenting a novel object count hypothesis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive with multiple benchmarks, extensive ablations, and analytical dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ — Discovery-driven with a clear narrative structure and well-designed experiments.
- **Value**: ⭐⭐⭐⭐⭐ — The SPRIGHT dataset and the <500 image SOTA methodology make a massive contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpatialReward: Verifiable Spatial Reward Modeling for Fine-Grained Spatial Consistency in Text-to-Image Generation](../../CVPR2026/image_generation/spatialreward_verifiable_spatial_reward_modeling_for_fine-grained_spatial_consis.md)
- [\[ECCV 2024\] Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)

</div>

<!-- RELATED:END -->
