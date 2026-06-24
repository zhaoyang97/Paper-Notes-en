---
title: >-
  [Paper Note] FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance
description: >-
  [ECCV 2024][3D Vision][CLIP] FALIP (Foveal-Attention CLIP) is proposed as a training-free method that enhances the region-awareness capability of CLIP without modifying the original image by inserting a foveal-like attention mask into the multi-head self-attention module of CLIP. It achieves improvements across zero-shot tasks including referring expression comprehension, image classification, and 3D point cloud recognition.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "CLIP"
  - "Zero-Shot Learning"
  - "Visual Prompt"
  - "Attention Mechanism"
  - "Point Cloud Recognition"
date: 2026-05-08
content_hash: ec446ff0b91f0651
---

# FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance

**Conference**: ECCV 2024  
**arXiv**: [2407.05578](https://arxiv.org/abs/2407.05578)  
**Code**: [https://pumpkin805.github.io/FALIP/](https://pumpkin805.github.io/FALIP/)  
**Area**: 3D Vision / Vision-Language Models  
**Keywords**: CLIP, Zero-Shot Learning, Visual Prompt, Attention Mechanism, Point Cloud Recognition

## TL;DR

FALIP (Foveal-Attention CLIP) is proposed as a training-free method that enhances the region-awareness capability of CLIP without modifying the original image by inserting a foveal-like attention mask into the multi-head self-attention module of CLIP. It achieves improvements across zero-shot tasks including referring expression comprehension, image classification, and 3D point cloud recognition.

## Background & Motivation

**Background**: CLIP possesses strong zero-shot capabilities through large-scale image-text contrastive learning. Researchers have explored guiding CLIP to focus on specific regions through visual prompts (e.g., red circles, blur masks), which achieves improvements in tasks like referring expression comprehension.

**Limitations of Prior Work**: Existing visual prompting methods (such as RedCircle, Blur, etc.) directly modify the input image, inevitably destroying original image information. For example, RedCircle introduces extra red elements that may interfere with fine-grained classification, and Blur obscures most image details. This leads to performance degradation or even negative impacts in scenarios requiring high image fidelity.

**Key Challenge**: The goal of visual prompting is to guide the model to focus on specific regions, but there is a fundamental conflict between the implementation method (editing the image) and the goal (preserving the integrity of image information)—guiding attention inevitably removes key information.

**Goal**: Can the attention guiding effect of visual prompting be achieved without modifying the original image content?

**Key Insight**: Instead of modifying the image input end, an attention bias is directly injected at the self-attention level of the model to simulate human foveal vision characteristics.

**Core Idea**: Transforming the visual prompt from "editing images" to "editing attention"—adding a Gaussian-weighted foveal attention mask to the self-attention score matrix of CLIP ViT.

## Method

### Overall Architecture

The pipeline of FALIP: (1) Inputs the original image and the Region of Attention (ROA), generating an attention mask $M$ through the foveal attention generation module; (2) Inputs the original image into the CLIP image encoder, while injecting the mask $M$ into the multi-head self-attention (MSA) module; (3) Outputs are obtained via image-text similarity calculations tailored to different downstream tasks (REC, classification, point cloud recognition). The entire process preserves the original image and requires no additional training.

### Key Designs

1. **Foveal Attention Mask Generation**:

    - **Function**: Generates an attention bias matrix $M \in \mathbb{R}^{(N+1) \times (N+1)}$ based on the Region of Attention (ROA), which is injected into the self-attention calculation.
    - **Mechanism**: First, a 2D Gaussian distribution is generated in the token space corresponding to the ROA:
    $R_{i,j} = e^{-\frac{[i-(H'-1)/2]^2 + [j-(W'-1)/2]^2}{2\sigma^2}}$
      Normalized as: $R^{norm} = \alpha \times \frac{R - \text{Min}(R) + \epsilon}{\text{Max}(R) - \text{Min}(R) + \epsilon}$. The mask $M$ only assigns non-zero values to the ROA positions in the first row (corresponding to the `[CLS]` token), while other positions are 0. The final attention calculation becomes:
    $\text{Foveal-Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right)V$
    - **Design Motivation**: Inspired by the selective focusing capability of human foveal vision. Gaussian weighting enables a smooth transition from the focal region to the surrounding background, avoiding information loss caused by hard truncation. Modifying only the `[CLS]` token row is because experiments show that the `[CLS]` token plays a key role in the final prediction, and modifying other rows instead disrupts the original information between tokens.

2. **Referring Expression Comprehension (REC) Application**:

    - **Function**: Locating target objects in the image based on text descriptions.
    - **Mechanism**: Converts candidate boxes into attention masks $M^* \in \mathbb{R}^{B \times (N+1)^2}$, calculates the similarity with the text for each candidate box as $S_i = \mathbf{T}(t) \cdot \mathbf{V}^T(\mathbf{x}, M_i)$, and uses a "subtract" post-processing to subtract negative sample scores to filter out the best match.
    - **Design Motivation**: As a plug-and-play module, FALIP can directly enhance existing REC methods and is complementary to methods like RedCircle.

3. **3D Point Cloud Recognition Application**:

    - **Function**: Extends FALIP to 3D point cloud recognition to improve performance by enhancing CLIP's attention to foreground regions.
    - **Mechanism**: Following the PointCLIP framework, 3D point clouds are projected into 2D depth maps from six views $\bar{\mathbf{x}} \in \mathbb{R}^{6 \times C \times H \times W}$. The foreground positions in each depth map are located to generate corresponding foveal attention masks $M^* \in \mathbb{R}^{6 \times (N+1)^2}$. The scores from the six views are weighted and fused: $Score_i = \sum_{j=1}^6 \beta_j \mathbf{V}(\bar{\mathbf{x}}_j, M_j) \cdot \mathbf{T}^T(\bar{t}_i)$.
    - **Design Motivation**: Backgrounds occupy a large area in depth maps, and CLIP's original attention may be distracted by the background. FALIP can guide attention to concentrate on the foreground objects.

4. **Unleash Visual Prompts**:

    - **Function**: Discovers that different attention heads have varying sensitivity to visual prompts, and further enhances performance by adjusting sensitive heads.
    - **Mechanism**: Decomposes the model output into the sum of contributions from each attention head, computes the change of each head before and after using visual prompts as $\Delta = \sum_{h=1}^H (G'_h - G_h)$, and finds that the last 4 layers exhibit the largest changes. The potential of visual prompting is "unleashed" by amplifying these changes: $[\text{MSA}]_{cls} = \sum_{h=1}^H [G'_h + (G'_h - G_h)]$, $l \in [L-3, L]$.
    - **Design Motivation**: The potential of visual prompting is currently not fully exploited. Analyzing responses at the attention-head level can further boost performance.

### Loss & Training

FALIP is a completely training-free (train-free) method that does not require any fine-tuning or additional training. All operations are completed at inference time, involving only the calculation and injection of the attention mask. Hyperparameters $\alpha = 0.2$ and $\sigma = 100$ are empirically optimal values.

## Key Experimental Results

### Main Results

Referring Expression Comprehension (REC, gold setting, Without E and P):

| Method | RefCOCO TestA | RefCOCO TestB | RefCOCO+ TestA | RefCOCOg Test | Avg |
|------|-------------|-------------|---------------|-------------|-----|
| RedCircle | 41.6 | 36.2 | 44.7 | 45.4 | 41.3 |
| PASTA | 41.7 | 37.6 | 43.2 | 49.2 | - |
| **FALIP** | **44.2** | **39.4** | **46.8** | **51.5** | **45.2** |

Image Classification:

| Method | StanfordDogs Top1 | CUB-200 Top1 | ImageNet-S Top1 | Waterbirds Top1 |
|------|------------------|-------------|----------------|----------------|
| Original CLIP | 56.5 | 54.2 | 64.9 | 78.2 |
| RedCircle | 52.4 | 44.2 | 62.8 | 77.5 |
| **FALIP** | **58.3** | **54.3** | **67.3** | **79.7** |

3D Point Cloud Recognition:

| Method | ModelNet40 | ScanObjectNN | Avg |
|------|-----------|-------------|-----|
| Original CLIP | 16.5 | 14.6 | 15.6 |
| **FALIP** | **18.6** | **15.3** | **17.0** |

### Ablation Study

| Mask Formulation | RefCOCO TestA | RefCOCOg Test | Avg |
|---------|------------|-------------|------|
| No mask (Original CLIP) | 14.8 | 25.5 | 21.5 |
| Method a ([CLS] row only) | **44.2** | **51.5** | **45.2** |
| Method b (All rows) | 36.6 | 43.7 | 39.3 |
| Method c (Diagonal) | 13.3 | 16.1 | 15.8 |

Comparison of Attention Manipulation Methods:

| Method | Avg |
|------|-----|
| Replace v (Replace RedCircle v with original image v) | 39.0 |
| Replace q,k (Replace with original image q, k) | 38.6 |
| Feature mask (Direct feature masking) | 27.0 |
| **FALIP** | **45.2** |

### Key Findings

- **The essence of visual prompting is modifying attention**: Experiments confirm that the fundamental reason why visual prompts (e.g., RedCircle) are effective is that they change the attention weights of the model on specific regions, rather than introducing new visual information.
- **RedCircle degrades performance on classification tasks**: On StanfordDogs, it drops from 56.5 to 52.4, and on CUB-200, from 54.2 to 44.2, indicating that directly editing images destroys fine-grained features.
- **The `[CLS]` token row plays a decisive role in predictions**: Modifying only the `[CLS]` row (Method a) achieves the best performance. Modifying all rows (Method b) or the diagonal (Method c) instead disrupts the original informational relationships among tokens.
- **The Unleash mechanism can further boost performance by 4%+**: By amplifying the change of sensitive attention heads in the last 4 layers, the average accuracy of RedCircle is improved from 41.3% to 45.2%.
- **FALIP is complementary to existing methods**: The combination of RedCircle + FA outperforms using either method alone in multiple settings.

## Highlights & Insights

- **Conceptual Elegance**: Redefining the visual prompt from "editing inputs" to "editing attention" is both elegant and profound. This shift in perspective resolves the fundamental conflict of visual prompting methods.
- **Completely Training-Free**: FALIP requires no additional training or fine-tuning, providing a true plug-and-play solution. The additional computational overhead at inference time is negligible.
- **Cross-Task Generality**: From REC to classification and 3D point cloud recognition, the unified foveal attention mechanism is effective across various tasks, indicating that attention guidance is a general pathway for enhancing CLIP's capabilities.
- **Brilliant Attention-Head Decoupling Analysis**: Revealing the differences in sensitivity of various attention heads to visual prompts and proposing to "unleash" prompt potential by amplifying sensitive heads provides inspiring insights.

## Limitations & Future Work

- The absolute performance on 3D point cloud recognition remains low (ModelNet40 is only 18.6%), which is more a limitation of the PointCLIP framework itself rather than a problem with FALIP.
- The hyperparameters $\alpha$ and $\sigma$ need to be adjusted for different tasks, and the paper does not provide a general selection strategy.
- The choice of "amplifying the last 4 layers" in the Unleash mechanism (Eq. 6) is based on empirical observation and lacks a theoretical explanation.
- Experiments were only conducted on the ViT-B/16 model, and the effects on larger models (ViT-L, ViT-G) remain unknown.
- Whether it is still effective in tasks requiring precise regression (e.g., object detection/segmentation) rather than selection is worth exploring.

## Related Work & Insights

- **vs RedCircle**: RedCircle guides attention by drawing a red circle on the image, which introduces extra red elements that interfere with fine-grained classification. FALIP operates directly at the attention level, completely preserving the original image information.
- **vs Alpha-CLIP / RegionCLIP**: These methods require additional training or fine-tuning to enhance region-awareness capability, whereas FALIP is training-free, making it more lightweight and flexible.
- **vs PASTA**: PASTA is also a visual prompting method; FALIP outperforms PASTA on most metrics and provides a deeper analysis of the attention mechanism.
- **Insights for Visual Prompting Research**: The paper reveals the equivalent relationship "visual prompt = attention modification", providing a theoretical foundation for designing better visual prompting methods—attention distributions should be directly optimized rather than indirectly through image editing.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective shift of "editing attention rather than editing images" is ingenious, though the core execution (adding bias in attention scores) is technically simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three major tasks (REC, classification, and 3D recognition). The ablation studies are highly comprehensive (mask formulation, QKV replacement, hyperparameter sensitivity, unleash) with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ The motivation analysis is convincing and progresses step-by-step, featuring a complete logical chain from observation to hypothesis to verification.
- Value: ⭐⭐⭐⭐ Training-free, plug-and-play, and complementary to existing methods, showing high practical value. The decoupling analysis of attention heads carries academic value for understanding the internal mechanisms of VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[AAAI 2026\] VPN: Visual Prompt Navigation](../../AAAI2026/3d_vision/vpn_visual_prompt_navigation.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Language-Driven 6-DoF Grasp Detection Using Negative Prompt Guidance](language-driven_6-dof_grasp_detection_using_negative_prompt_guidance.md)

</div>

<!-- RELATED:END -->
