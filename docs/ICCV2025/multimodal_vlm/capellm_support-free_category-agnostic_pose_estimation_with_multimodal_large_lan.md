---
title: >-
  [Paper Note] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Category-agnostic pose estimation] This work is the first to introduce multimodal large language models (MLLMs) into category-agnostic pose estimation (CAPE), enabling keypoint localization for arbitrary categories using only a query image and textual descriptions—without requiring traditional support images or annotations—surpassing the 5-shot state-of-the-art on the MP-100 benchmark.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Category-agnostic pose estimation
  - MLLM
  - support-free
  - keypoint detection
  - distribution modeling
date: 2026-05-08
content_hash: e32c3a658bf73de7
---

# CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models

**Conference**: ICCV 2025
**arXiv**: [2411.06869](https://arxiv.org/abs/2411.06869)
**Code**: Available (link provided in paper)
**Area**: Multimodal VLM
**Keywords**: Category-agnostic pose estimation, MLLM, support-free, keypoint detection, distribution modeling

## TL;DR

This work is the first to introduce multimodal large language models (MLLMs) into category-agnostic pose estimation (CAPE), enabling keypoint localization for arbitrary categories using only a query image and textual descriptions—without requiring traditional support images or annotations—surpassing the 5-shot state-of-the-art on the MP-100 benchmark.

## Background & Motivation

Category-agnostic pose estimation (CAPE) aims to generalize to object categories unseen during training. Existing methods (e.g., GraphCAPE, CapeX) rely on a "support set"—annotated reference images from the same category—to establish keypoint correspondences. This introduces two core problems:

**Performance instability**: The same query image paired with support images of varying quality produces inconsistent predictions.

**Heavy annotation burden**: New categories require re-annotating a support set.

CapeX attempts to replace support images with keypoint name text, but still depends on skeleton structure information and uses only simple keypoint names, failing to leverage the rich semantic priors embedded in large language models.

The central question of this paper is: **Can support sets be entirely eliminated, relying solely on an image and detailed textual descriptions for CAPE?** The answer is affirmative—by harnessing the powerful vision-language understanding of MLLMs, no auxiliary queries are needed.

## Method

### Overall Architecture

CapeLLM consists of three core components: (1) a DINOv2 visual encoder for image feature extraction, (2) a linear projection layer mapping visual tokens to the language space, and (3) LLaMA3.1 as the backbone language model for inference. The input is a query image paired with a textual instruction containing keypoint names and detailed descriptions; the output is keypoint coordinates in floating-point format.

### Key Designs

1. **Floating-Point Decoding**:
   Coordinates are generated directly in the format `[0.abc, 0.def]`, with each decimal digit predicted as an independent token. For a scalar value $y$, the prediction is approximated via autoregressive factorization over $K=3$ digit tokens:
   $$p(y|\mathbf{x}) \approx \prod_{k=1}^{K} p_{\phi,\theta}(\mathbf{Y}_k | \mathbf{x}, \mathbf{Y}_1, ..., \mathbf{Y}_{k-1})$$
   This is equivalent to modeling a truncated conditional density at precision $10^{-K}$, offering greater flexibility than Gaussian parameterization. It implicitly models arbitrary probability distributions, avoiding the limitations of fixed-variance Gaussians (e.g., keypoints at object boundaries should not spread probability mass into the background).

2. **Instruction Design**:
   A core innovation is replacing simple keypoint names with **detailed spatial relation descriptions**. These descriptions encode precise spatial positions and inter-keypoint relative relationships (e.g., "next to the fifth wheel, in front of the second wheel"), avoiding ambiguous expressions. This design yields approximately 6% improvement in mPCK.

3. **Dynamic-Round Training**:

   - **Fixed-round**: Keypoints for each category are divided into fixed-size groups of $k$, each paired with the image to form multi-turn dialogues.
   - **Dynamic-round**: The number of keypoints per turn varies randomly, encouraging the model to leverage other keypoints for spatial reasoning.

   Dynamic-round training is particularly suited to the **cumulative reasoning** inference strategy, enabling the model to utilize previously predicted keypoints as context to refine subsequent predictions.

4. **Cumulative Reasoning**:
   During inference, the prompt from the previous keypoint prediction is concatenated into the input for the next, forming an accumulating context. Combined with dynamic-round training, this strategy yields a 1.2% improvement on PCK@0.2.

### Loss & Training

- LoRA is applied to the visual encoder and the query/value projection layers of the LLM, with rank=8
- Optimizer: AdamW, learning rate $5 \times 10^{-4}$
- Trained for 12 epochs with 3% of total steps as warm-up
- Input images resized to 224×224
- Default of 4 dialogue turns per instruction
- 4× RTX-A6000 GPUs with gradient accumulation of 32 steps

## Key Experimental Results

### Main Results

| Method | Setting | Split1 | Split2 | Split3 | Split4 | Split5 | Avg. PCK@0.2 |
|--------|---------|--------|--------|--------|--------|--------|--------------|
| GraphCAPE (1-shot) | Requires support set | 94.63 | 89.79 | 90.30 | 87.81 | 90.07 | 90.52 |
| GraphCAPE (5-shot) | Requires support set | 95.81 | 90.78 | 90.94 | 90.42 | 92.27 | 92.04 |
| CapeX | Text + skeleton | 95.29 | 91.08 | 88.94 | 89.83 | 92.96 | 91.62 |
| **CapeLLM** | **Text-only (support-free)** | **97.01** | **92.40** | **90.58** | **90.90** | **92.11** | **92.60** |

### Ablation Study

| Configuration | PCK@0.05 | PCK@0.2 | mPCK |
|--------------|---------|---------|------|
| LocLLM-style training | 55.15 | 94.85 | 84.00 |
| Fixed-round training | 78.43 | 96.98 | 91.98 |
| Dynamic-round training | 76.55 | 96.05 | 90.92 |
| Dynamic + cumulative reasoning | — | 97.28 | — |
| Ambiguous descriptions | 69.82 | 91.97 | 85.98 |
| Spatial relation descriptions | 78.43 | 96.98 | 91.98 |

### Key Findings

- **Support-free surpasses 5-shot**: CapeLLM with textual descriptions alone outperforms GraphCAPE's 5-shot result (+0.56%), demonstrating that MLLM semantic understanding can fully substitute for support sets.
- **Instruction quality is critical**: Spatial relation descriptions vs. ambiguous descriptions yield a 6% gap in mPCK.
- **Advantages of implicit distribution modeling**: Floating-point decoding naturally models non-Gaussian distributions, producing more reasonable predictions for keypoints near object boundaries.
- **Robustness to input variation**: Performance remains largely stable when keypoint names and descriptions are paraphrased using GPT-4o.
- **Strong occlusion robustness**: The method handles self-occlusion and object occlusion without requiring skeleton information.

## Highlights & Insights

- **Paradigm shift**: Moving from support-dependent to support-free CAPE, entirely replacing traditional visual support sets with the prior knowledge embedded in MLLMs.
- **Floating-point decoding as implicit density estimation**: A particularly compelling finding—by predicting coordinates digit by digit, the MLLM implicitly models a truncated conditional density function, far more flexible than a fixed Gaussian.
- **Detailed textual descriptions > support images**: This suggests that in CAPE, precise semantic information is more valuable than visual similarity.

## Limitations & Future Work

- The MP-100 dataset is limited in scale (~200 images per category); performance on larger datasets remains to be verified.
- Floating-point precision is fixed at 3 decimal places ($10^{-3}$), which may be insufficient for tasks requiring higher localization accuracy.
- Textual descriptions currently require manual design; automatic generation of high-quality descriptions is a promising direction.
- Inference speed is slower than specialized models due to autoregressive LLM decoding overhead.
- Multi-instance scenarios (multiple objects of the same category in a single image) are not explored.

## Related Work & Insights

- The key distinction from LocLLM lies in category-agnostic capability—direct application of LocLLM, which is category-specific, performs poorly.
- Cumulative reasoning is analogous to chain-of-thought prompting, indicating that spatial relational reasoning among keypoints is learnable by MLLMs.
- Future work may consider integrating active learning mechanisms, where the model actively solicits additional contextual information for low-confidence keypoints.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of MLLMs to CAPE; significant paradigm innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ablations are highly detailed, covering training strategies, description types, resolutions, and decoding strategies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with systematic experimental design.
- **Value**: ⭐⭐⭐⭐ Demonstrates the feasibility of MLLMs for fine-grained visual localization tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs](autocompose_automatic_generation_of_pose_transition_descriptions_for_composed_po.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)

<!-- RELATED:END -->
