---
title: >-
  [Paper Note] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs
description: >-
  [ECCV 2024][Image Generation] ZipLoRA proposes a cheap and efficient LoRA merging method. By learning column-wise merging coefficients and minimizing the cosine similarity between columns, it achieves hyperparameter-free merging of independently trained subject LoRAs and style LoRAs, generating personalized "any subject × any style" images in diffusion models.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 59f7c645d84f2dff
---

# ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs

**Conference**: ECCV 2024  
**arXiv**: [2311.13600](https://arxiv.org/abs/2311.13600)  
**Area**: Model Compression

## TL;DR

ZipLoRA proposes a cheap and efficient LoRA merging method. By learning column-wise merging coefficients and minimizing the cosine similarity between columns, it achieves hyperparameter-free merging of independently trained subject LoRAs and style LoRAs, generating personalized "any subject × any style" images in diffusion models.

## Background & Motivation

Personalized generation in diffusion models has achieved success in both subject-driven and style-driven directions, but combining a specific subject with a specific style remains an unsolved challenge:

**Existing merging methods are unreliable**: The most common approach is a weighted linear combination of subject and style LoRAs ($\Delta W_m = w_c \cdot \Delta W_c + w_s \cdot \Delta W_s$), which requires laborious grid search to adjust the coefficients and lacks robustness across different subject-style combinations.

**Limitations of joint training**: Multi-concept joint training (such as Custom Diffusion) requires training from scratch, is computationally expensive, and struggles to disentangle style from subject.

**Signal interference between LoRAs**: There can be high cosine similarity (alignment) between the columns of the weight matrices of two independently trained LoRAs. Directly adding them leads to overlapping information and signal interference, causing the merged model to lose its capability to accurately synthesize individual concepts.

Two key observations of ZipLoRA: (1) LoRA weight matrices are sparse, where 90% of the elements can be discarded without affecting generation quality; (2) highly aligned columns lead to performance degradation during direct merging.

## Method

### Overall Architecture

ZipLoRA operates on a base model $D$ (SDXL v1.0). Given independently trained subject LoRA $L_c = \{\Delta W_c^{(i)}\}$ and style LoRA $L_s = \{\Delta W_s^{(i)}\}$, it learns merging coefficient vectors $m_c$ and $m_s$ to maintain both subject and style fidelity in the merged LoRA:

$$\Delta W_m = m_c \otimes \Delta W_c + m_s \otimes \Delta W_s$$

where $\otimes$ denotes column-wise broadcasting multiplication, and the $j$-th element of $m_c$ scales the contribution of the $j$-th column of $\Delta W_c$.

### Key Designs

**Sparsity of LoRA Weights**:
- Most elements in the LoRA update matrix $\Delta W$ have magnitudes close to zero.
- Empirical validation: Discarding the 90% lowest-magnitude elements has almost no impact on model performance.
- This sparsity allows certain columns to be ignored during merging, providing additional degrees of freedom to minimize interference.

**Column Alignment and Signal Interference**:
- Measuring the cosine similarity of corresponding columns between two LoRAs reveals that the similarity is significantly non-zero in many layers.
- Directly adding highly aligned columns leads to overlapping concept information, causing distortion.
- When columns are orthogonal (zero cosine similarity), merging can fully retain their respective information.

**ZipLoRA Optimization**:
- Like a zipper combining two pieces of fabric, ZipLoRA learns a set of non-overlapping merging coefficients to "zip" the subject and style LoRAs together.
- The base model and LoRA weights are frozen, and only the merging coefficient vectors $m_c$ and $m_s$ are optimized.
- It requires only 100 gradient updates, which is 1/10 of the joint training.

**SDXL's Strong Style Learning Ability**:
- It is discovered that SDXL can achieve high-quality style learning through DreamBooth LoRA fine-tuning using only a single reference style image.
- It requires no human-feedback iterative training like StyleDrop.
- This characteristic makes ZipLoRA particularly effective on SDXL.

### Loss & Training

$$\mathcal{L}_{merge} = \|(D \oplus L_m)(x_c, p_c) - (D \oplus L_c)(x_c, p_c)\|_2$$
$$+ \|(D \oplus L_m)(x_s, p_s) - (D \oplus L_s)(x_s, p_s)\|_2$$
$$+ \lambda \sum_i |m_c^{(i)} \cdot m_s^{(i)}|$$

The three objectives ensure:
1. The merged model retains the subject generation capability.
2. The merged model retains the style generation capability.
3. The cosine similarity between subject and style columns is minimized ($\lambda = 0.01$).

## Key Experimental Results

### Main Results

**Table 1: User preference study (ZipLoRA win rate)**

| Baselines | ZipLoRA Preference Rate |
|----------|---------------|
| Direct Merge (Linear Combination) | 82.7% |
| Joint Training | 71.1% |
| StyleDrop + DreamBooth | 68.0% |
| Mix of Show | 87.3% |
| Custom Diffusion | 88.1% |

**Table 2: Comparison of CLIP/DINO alignment scores**

| Method | Style-align↑ | Subject-align↑ | Text-align↑ |
|------|-------------|---------------|-------------|
| **ZipLoRA** | 0.699 | **0.420** | **0.303** |
| Joint Training | 0.680 | 0.378 | 0.296 |
| Direct Merge | **0.702** | 0.357 | 0.275 |
| StyleDrop + DreamBooth | 0.646 | 0.394 | 0.263 |
| Mix of Show | 0.635 | 0.374 | 0.251 |
| Custom Diffusion | 0.616 | 0.346 | 0.262 |

### Ablation Study

**Running time and resource comparison**

| Method | Training Steps | Running Time (s) | Trainable Parameters | GPU Memory |
|------|---------|-------------|-----------|---------|
| **ZipLoRA** | **100** | **560** | **1.6M** | **21 GB** |
| Joint Training | 1000 | 3540 | 180M | 38 GB |
| Custom Diffusion | 1000 | 3890 | 180M | 38 GB |
| Mix of Show | 1000+1000+1780 | 4980 | 180M | 38 GB |

**Storage Overhead**

| Method | Storage Requirement |
|------|---------|
| **ZipLoRA** | **6.5 MB** (merging coefficients only) |
| Other methods | 360 MB (complete LoRA weights) |

### Key Findings

1. ZipLoRA achieves overwhelming user preference in all comparisons, especially against Mix of Show (87.3%) and Custom Diffusion (88.1%).
2. Regarding subject alignment score, ZipLoRA reaches 0.420, substantially leading the second-best, StyleDrop+DreamBooth (0.394), while maintaining competitive style alignment.
3. ZipLoRA requires only 100 gradient updates (560 seconds), which is 1/6 of the training time of joint training, with only 1.6M trainable parameters (compared to 180M).
4. The merged model retains the capacity to generate the subject and style independently, whereas a direct merge tends to lose this capability.

## Highlights & Insights

- **Elegant design of the zipper metaphor**: LoRA merging is analogized to a zipper. By learning disjoint multipliers to achieve orthogonal merging, it offers clear physical intuition.
- **Fundamental observation of LoRA weights**: It reveals the critical impact of LoRA sparsity and column alignment on merger quality, providing theoretical guidance for the LoRA ecosystem.
- **Discovery experiments on SDXL style learning**: It is pointed out for the first time that SDXL can learn high-quality styles from just a single image, a finding that holds independent value.
- **Extreme efficiency**: With only 100 optimization steps, 1.6M parameters, and 6.5MB of storage, it makes large-scale LoRA compositions feasible.

## Limitations & Future Work

1. If the style LoRA fails to disentangle style and content (e.g., binding a cliff to a watercolor style), ZipLoRA cannot decompose them further, leading to content leakage from the style reference image into the output.
2. The current focus is on merging two LoRAs (one subject + one style); multi-LoRA composition is left for future work.
3. The performance depends on the base model's style-learning capability. On SDv1.5, the effectiveness of ZipLoRA is limited due to the lower quality of the style LoRAs.
4. CLIP and DINO alignment metrics are imperfect, particularly exhibiting limitations in measuring fine-grained style differences.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [\[ECCV 2024\] ZigMa: A DiT-style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)
- [\[ECCV 2024\] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)

</div>

<!-- RELATED:END -->
