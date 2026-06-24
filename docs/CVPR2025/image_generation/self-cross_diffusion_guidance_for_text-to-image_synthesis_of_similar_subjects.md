---
title: >-
  [Paper Note] Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects
description: >-
  [CVPR 2025][Image Generation][Diffusion guidance] This paper proposes Self-Cross Diffusion Guidance, which effectively addresses the subject mixing problem when generating similar subjects with diffusion models by penalizing the overlap between the aggregated self-attention map of one subject and the cross-attention map of another. It represents the first training-free method to simultaneously leverage the interactions between self-attention and cross-attention.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion guidance"
  - "subject blending"
  - "self-cross attention"
  - "training-free inference"
  - "similar subject generation"
date: 2026-05-08
content_hash: 781606756c7902e1
---

# Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects

**Conference**: CVPR 2025  
**arXiv**: [2411.18936](https://arxiv.org/abs/2411.18936)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Diffusion guidance, subject blending, self-cross attention, training-free inference, similar subject generation

## TL;DR

This paper proposes Self-Cross Diffusion Guidance, which effectively addresses the subject mixing problem when generating similar subjects with diffusion models by penalizing the overlap between the aggregated self-attention map of one subject and the cross-attention map of another. It represents the first training-free method to simultaneously leverage the interactions between self-attention and cross-attention.

## Background & Motivation

- While diffusion models have made remarkable progress in text-to-image generation, subject mixing remains a critical unsolved issue.
- This problem is particularly severe when generating multiple visually similar subjects (e.g., a leopard and a tiger), where the features of different subjects tend to bleed into each other.
- Existing methods (Attend&Excite, INITNO, CONFORM) perform guidance based on either cross-attention or self-attention individually, neglecting the interaction between the two.
- Focusing only on the most discriminative patches (e.g., a bird's beak) is insufficient—other foreground patches can also lead to subject mixing.
- Existing evaluation benchmarks lack challenging prompts for similar subject scenarios, and CLIP scores correlate poorly with human judgment.

## Method

### Overall Architecture

Self-Cross Guidance is a training-free inference-time optimization method. During the first half of the denoising steps in the diffusion reverse process, corresponding patches are selected from the cross-attention map of each subject using Otsu thresholding. The self-attention maps of these patches are then aggregated to penalize the overlap between the aggregated self-attention and the cross-attention of other subjects. This is implemented through a combination of initial noise optimization and iterative latent refinement.

### Key Designs

**Design 1: Self-Attention Map Aggregation**

- **Function**: To obtain a self-attention representation covering the entire subject region.
- **Mechanism**: Apply Otsu thresholding to the cross-attention map $A_i^c$ of subject $i$ to select high-response patches. The self-attention maps of the selected patches are summed as a weighted average based on their cross-attention values: $A_i^s = \frac{\sum_{x_m,y_n}(A_i^c[x_m,y_n] \times A_{x_m,y_n}^s)}{\sum_{x_m,y_n} A_i^c[x_m,y_n]}$
- **Design Motivation**: Self-attention maps of different patches vary significantly; relying solely on the single most discriminative patch cannot cover the entire area of the subject. Aggregating the self-attention maps of multiple patches provides a more comprehensive representation of the subject's attended region.

**Design 2: Self-Cross Guidance Loss**

- **Function**: To penalize the overlap between the self-attention region of one subject and the cross-attention region of another, thereby eliminating subject mixing.
- **Mechanism**: For a pair of subjects $(i, j)$, the overlap is computed as $g(i,j) = \sum_{x,y} \min(A_i^s[x,y], A_j^c[x,y]) + \sum_{x,y} \min(A_i^c[x,y], A_j^s[x,y])$. For $N$ similar subjects, the average of all $C_N^2$ pairs is calculated. The total loss is defined as $\mathcal{L}_{total} = S_{self-cross} + \lambda \cdot S_{cross-attn}$.
- **Design Motivation**: The essence of subject mixing is that the self-attention of one subject invades the region of another subject. The overlap between the aggregated self-attention map and the cross-attention map captures this intrusion more precisely than using either attention type in isolation.

**Design 3: SSD Benchmark and GPT-4o Evaluation**

- **Function**: To provide a challenging evaluation benchmark for similar subject generation.
- **Mechanism**: Release the Similar-Subject Dataset (SSD), which contains text prompts featuring two or three similar subjects. Utilize GPT-4o to automatically evaluate subject presence, identifiability, and attribute binding in the generated images via visual question answering.
- **Design Motivation**: CLIP scores cannot effectively differentiate subject mixing issues. GPT-4o evaluation exhibits higher consistency with human judgment.

### Loss & Training

$$\mathcal{L}_{total} = S_{self-cross} + \lambda \cdot S_{cross-attn}$$

where $S_{cross-attn}$ follows the cross-attention response score of Attend&Excite, and $\lambda$ is a balancing coefficient. This loss is only applied during the first half of the denoising steps and to the intermediate layers.

## Key Experimental Results

### Quantitative Results on SSD Benchmark

| Method | Presence ↑ | Identifiability ↑ | Attribute Binding ↑ | FID ↓ |
|------|---------|-----------|-----------|------|
| Stable Diffusion | Baseline | Baseline | Baseline | Baseline |
| Attend&Excite | Improved | Limited Improvement | Limited Improvement | — |
| INITNO | Improved | Partial Improvement | Partial Improvement | — |
| CONFORM | Improved | Partial Improvement | Partial Improvement | — |
| **Self-Cross (Ours)** | **Best** | **Best** | **Best** | **Maintained** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Cross-attn guidance only | Fails to eliminate subject mixing |
| Self-attn guidance only | Partial improvement |
| Single-patch self-attn + cross-attn | Limited improvement |
| **Aggregated self-attn + cross-attn** | **Significant elimination of subject mixing** |

### Key Findings

- Self-Cross guidance substantially outperforms methods that rely on a single attention map, such as INITNO, in eliminating subject mixing.
- Aggregating multi-patch self-attention yields significantly better results than single-patch approaches.
- The method is compatible with both UNet-based (SD 1.x/2.x) and Transformer-based (SD3) diffusion models.
- As a side benefit, subject omission issues are also alleviated.
- The overall image quality (FID) is not visibly affected.

## Highlights & Insights

1. **First exploration of the interaction between self-attention and cross-attention**: Provides a new understanding of the causes of subject mixing—self-attention intrusion into other regions leading to feature copying.
2. **Necessity of multi-patch aggregation**: Demonstrates that focusing solely on the most discriminative patch is insufficient to eliminate subject mixing.
3. **GPT-4o evaluation scheme**: Provides a more reliable automated evaluation instrument for the diffusion model community.

## Limitations & Future Work

- Initial noise optimization and iterative refinement increase inference time.
- The method primarily targets mixing between similar subjects, offering limited improvement for attribute binding of non-similar subjects.
- It requires the user to specify which subjects are "similar," lacking an automatic detection mechanism.
- Future work can explore incorporating Self-Cross guidance into the training phase to scale to more scenarios.

## Related Work & Insights

- **Attend&Excite** [Chefer et al.] prevents subject omission via maximizing cross-attention.
- **INITNO** [Guo et al.] optimizes initial noise by combining self-attention conflict scores.
- **CONFORM** [Meral et al.] uses contrastive loss for subject separation.
- This work is the first to reveal the crucial role of the interaction between self-attention and cross-attention in subject mixing.

## Rating

⭐⭐⭐⭐ — Deep analysis on the causes of subject mixing (self-attention intrusion), and the design of the Self-Cross guidance loss is intuitive yet effective. The significant advantage of the multi-patch aggregation strategy over the single-patch scheme is convincing. The SSD benchmark and GPT-4o evaluation provide valuable tools to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](../../CVPR2026/image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)

</div>

<!-- RELATED:END -->
