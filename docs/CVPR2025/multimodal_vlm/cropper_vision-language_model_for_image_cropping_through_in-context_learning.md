---
title: >-
  [Paper Note] Cropper: Vision-Language Model for Image Cropping through In-Context Learning
description: >-
  [CVPR 2025][Multimodal VLM][Image cropping] This paper proposes the Cropper framework, which is the first to leverage the in-context learning (ICL) capability of large vision-language models (VLMs) for image cropping. Through efficient prompt retrieval and feedback-based iterative crop refinement strategies, it significantly outperforms supervised state-of-the-art (SOTA) methods across three tasks—free cropping, subject-aware cropping, and aspect-ratio cropping—without requir…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Image cropping"
  - "visual in-context learning"
  - "VLM"
  - "prompt retrieval"
  - "iterative refinement"
date: 2026-05-08
content_hash: 26623685dd7220ce
---

# Cropper: Vision-Language Model for Image Cropping through In-Context Learning

**Conference**: CVPR 2025  
**arXiv**: [2408.07790](https://arxiv.org/abs/2408.07790)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Image cropping, visual in-context learning, VLM, prompt retrieval, iterative refinement

## TL;DR
This paper proposes the Cropper framework, which is the first to leverage the in-context learning (ICL) capability of large vision-language models (VLMs) for image cropping. Through efficient prompt retrieval and feedback-based iterative crop refinement strategies, it significantly outperforms supervised state-of-the-art (SOTA) methods across three tasks—free cropping, subject-aware cropping, and aspect-ratio cropping—without requiring any training.

## Background & Motivation
Image cropping is a critical operation in photography applications, aiming to identify the most visually appealing cropped regions. Existing cropping methods rely on training neural networks on specific datasets, which suffer from two major issues: (1) **Poor generalization**: Trained models struggle to adapt to new requirements or data distributions, necessitating retraining. (2) **Task fragmentation**: Free cropping, subject-aware cropping, and aspect-ratio cropping each require distinct network architectures and training pipelines.

With breakthroughs in large VLMs like GPT-4o and Gemini, ICL offers a training-free paradigm for adapting to downstream tasks. However, VLMs face two challenges in image cropping: (1) ICL performance heavily depends on the quality of in-context examples, and manual selection is not scalable. (2) Injecting aesthetic concepts into VLMs is non-obvious.

**Core Idea**: **Utilize the ICL capability of VLMs to unify three cropping tasks**, selecting semantically similar in-context examples via automated prompt retrieval, and iteratively refining cropping quality using feedback from an aesthetic scorer.

## Method

### Overall Architecture
Cropper consists of two stages: (1) **Visual Prompt Retrieval**: Given an input image, it automatically retrieves top-$S$ semantically similar images and their ground truth (GT) crop coordinates from the training set to serve as ICL examples. (2) **Iterative Crop Refinement**: The VLM generates $R$ candidate crops based on the ICL examples, which are subsequently scored by an evaluator and fed back to the VLM for iterative refinement over $L$ rounds. All information is passed through structured prompts without modifying VLM parameters.

### Key Designs
1. **Visual Prompt Retrieval**:

    - Image similarity metric $Q$: Uses CLIP ViT-B/32 to extract embeddings of the query image and dataset images, retrieving the top-$S$ similar images using cosine similarity.
    - GT crop selection metric $G$: Varies by task—for free cropping, the best crop is selected using MOS scores; for subject-aware cropping, the closest mask is chosen based on the L2 distance of the mask center; for aspect-ratio cropping, aspect ratios are matched.
    - Formalization: $\mathcal{Z} = \arg\max_{z_i \in \mathcal{D}} Q(z_q, z_i)$, $\mathcal{H} = \arg\max_{c_j \in C_j} G(z_q, c_j)$
    - Optimal $S=30$ (too few ICL examples provide insufficient information, while too many introduce noise and increase token cost).
    - **Design Motivation**: Semantically similar images are more likely to share analogous optimal cropping strategies.

2. **Iterative Crop Refinement**:

    - The VLM first generates $R=6$ candidate crop coordinates $(s, x_1, y_1, x_2, y_2)$ based on the ICL examples.
    - Each candidate crop is evaluated across three dimensions: VILA-R aesthetic score + CLIP content preservation + region area.
    - The candidate cropped images and their corresponding scores are fed back to the VLM, with a prompt to "generate similar crops with high scores."
    - Refinement is executed for $L=2$ rounds (performance saturates after two rounds).
    - The VLM temperature is set to $0.05$ (lower randomness yields superior IoU).
    - **Design Motivation**: VLMs lack a deep, intrinsic understanding of absolute coordinate systems and aesthetic standards, necessitating guidance via explicit feedback loops.

3. **Unified Multi-Task Prompt Design**:

    - Crop coordinates are normalized to a range of 1-1000.
    - Free cropping: Uses a 5-tuple $(s, x_1, y_1, x_2, y_2)$ containing the MOS score.
    - Subject cropping: The input contains subject mask annotations, and the output is a 4-tuple $(x_1, y_1, x_2, y_2)$.
    - Aspect-ratio cropping: Specifies the target aspect ratio in the prompt and outputs a 4-tuple.
    - **Design Motivation**: A unified coordinate representation and prompt template enable a single VLM to support three distinct cropping tasks.

### Loss & Training
Cropper involves no training and is based entirely on ICL inference. The scoring function combines three metrics normalized to $[0,1]$:
- **VILA-R Aesthetic Score**: Evaluates composition, color contrast, perspective, etc.
- **CLIP Content Preservation Score**: Cosine similarity between CLIP embeddings of the original and cropped images.
- **Area Score**: $A = \frac{H_{crop} W_{crop}}{HW}$, preventing excessively small crop areas.

## Key Experimental Results

### Main Results (GAICD Free Cropping)

| Method | Training-Free? | $Acc_{1/5}$ | $\overline{Acc}_5$ | $\overline{Acc}_{10}$ | $\overline{SRCC}$ |
|------|-----------|------------|---------------------|----------------------|-------------------|
| GAIC | ✗ | 68.2 | 63.1 | 81.6 | 0.849 |
| TransView | ✗ | 69.0 | 63.9 | 82.4 | 0.857 |
| Chao et al. | ✗ | 70.0 | 64.8 | 83.3 | 0.872 |
| **Cropper** | **✓** | **88.9** | **84.3** | **96.5** | **0.904** |

### Subject-Aware Cropping (SACD)

| Method | Training-Free? | IoU↑ | Disp↓ |
|------|-----------|------|-------|
| SAC-Net | ✗ | 0.767 | 0.0491 |
| **Cropper** | **✓** | **0.769** | **0.0372** |

### Aspect-Ratio Cropping (FCDB)

| Method | Training-Free? | IoU↑ | Disp↓ |
|------|-----------|------|-------|
| Mars | ✗ | 0.735 | 0.062 |
| **Cropper** | **✓** | **0.756** | **0.053** |

### Ablation Study

| Configuration | IoU | $\overline{Acc}_5$ | Description |
|------|-----|---------------------|------|
| Full Cropper (VILA+Area) | 0.748 | 84.3 | Best overall configuration |
| VILA only | 0.748 | 83.6 | Aesthetic score is the primary contributor |
| Area only | 0.752 | 83.9 | Area score provides independent contribution |
| CLIP only | 0.751 | 81.2 | Content preservation has weak contribution to Acc |
| VILA+Area+CLIP | 0.754 | 84.3 | Comparable to VILA+Area |
| Random prompt instead of retrieval | - | Significantly worse | Validates necessity of prompt retrieval |
| S=1 (1 ICL example) | ~0.72 | - | Insufficient examples yield poor performance |
| S=30 (Optimal) | ~0.75 | - | Optimal number of ICL examples |
| L=0 (No iterative refinement) | ~0.72 | - | Validates necessity of iterative refinement |
| L=2 (Optimal) | ~0.75 | - | Performance saturates after 2 rounds |

### Key Findings
- **Training-free method significantly outperforms supervised methods for the first time**: Cropper improves $Acc_{1/5}$ on GAICD from the SOTA of 70.0% to 88.9% (+18.9 percentage points).
- **The quantity of ICL examples and the retrieval strategy are crucial**: Random selection of ICL examples is unstable and underperforms, whereas CLIP-based retrieval of 30 examples is optimal.
- **Iterative refinement dramatically boosts efficiency**: Going from no refinement to 2-round refinement improves IoU by approximately 3 percentage points.
- **GPT-4o zero-shot cropping performs poorly**: Directly utilizing GPT-4o for subject-aware cropping often crops out critical parts of the subject.
- **Unified framework accommodates three tasks**: A single VLM with modified prompts can address three distinct cropping requirements.
- **VLM temperature has a high impact**: A temperature of 0.05 outperforms higher temperatures, as lower randomness leads to better crops.

## Highlights & Insights
- **Successful application of ICL to coordinate regression**: Image cropping essentially involves predicting bounding box coordinates, transferring the ICL capability of VLMs from NLP to precise numerical prediction.
- **Generalizable paradigm of feedback-driven iterative refinement**: Scoring VLM outputs via an external evaluator and feeding them back to the model is highly applicable to other downstream VLM tasks.
- **The integration of CLIP retrieval and VILA scoring** serves as a highly practical utility pipeline for image aesthetics-related tasks.
- **Valuable case of training-free methods surpassing training-based counterparts**: Demonstrates that the in-context learning capabilities of large VLMs can replace task-specific supervised training.

## Limitations & Future Work
- **High Computational/Financial Cost**: Generating crops for each image requires 30 ICL examples + 6 candidates $\times$ 2 rounds of iteration, incurring significant API call fees on Gemini 1.5 Pro.
- **Reliance on Closed-Source VLMs**: Experiments primarily depend on Gemini 1.5 Pro and GPT-4o, preventing local deployment.
- **High Latency**: Multiple rounds of VLM calls make inference speed considerably slower than conventional lightweight cropping networks.
- The SRCC on GAICD (0.904) is in some extreme cases worse than the PCC performance (0.860, which is lower than some baseline scores like 0.893).
- The performance of open-source VLMs (such as LLaVA or Qwen-VL) within this framework has not yet been investigated.
- The choice of coordinate normalization scheme (1-1000) in the prompt lacks ablation analysis.

## Related Work & Insights
- **vs. Supervised Methods like GAIC/TransView**: Cropper significantly outperforms them without any training, illustrating that the general visual understanding capability of VLMs combined with appropriate ICL guidance can replace domain-specific networks.
- **vs. Directly Using GPT-4o/Gemini for Cropping**: Native VLMs (even with CoT prompts) perform poorly on cropping tasks, indicating that both in-context example guidance and iterative feedback are indispensable.
- **vs. NLP ICL Methods**: Successfully migrates the semantic similarity retrieval strategy of Liu et al. from NLP to vision-based coordinate regression tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply VLM ICL to image cropping; the unified framework for three cropping tasks is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three datasets, three cropping tasks, detailed ablation studies, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, and the prompt design is fully detailed.
- Value: ⭐⭐⭐⭐ High concept-verification value, demonstrating the potential of VLM ICL in coordinate prediction tasks, though practical deployment is constrained by cost and latency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2025\] Mimic In-Context Learning for Multimodal Tasks](mimic_in-context_learning_for_multimodal_tasks.md)
- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](../../ACL2025/multimodal_vlm/transferring_textual_preferences_to_vision-language_understanding_through_model_.md)
- [\[CVPR 2025\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)

</div>

<!-- RELATED:END -->
