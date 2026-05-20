---
title: >-
  [Paper Note] Attention to the Burstiness in Visual Prompt Tuning!
description: >-
  [ICCV 2025][Multimodal VLM][Visual Prompt Tuning] This paper reveals the "burstiness" and non-Gaussian distribution of self-attention module data in Visual Prompt Tuning…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Visual Prompt Tuning"
  - "Burstiness"
  - "Data Whitening"
  - "Bilinear Model"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: b61d9387c0dea18d
---

# Attention to the Burstiness in Visual Prompt Tuning!

**Conference**: ICCV 2025
**arXiv**: [2506.22908](https://arxiv.org/abs/2506.22908)  
**Code**: [GitHub](https://github.com/WangYZ1608/BPT)  
**Area**: Parameter-Efficient Fine-Tuning / Visual Prompt Learning
**Keywords**: Visual Prompt Tuning, Burstiness, Data Whitening, Bilinear Model, parameter-efficient fine-tuning

## TL;DR
This paper reveals the "burstiness" and non-Gaussian distribution of self-attention module data in Visual Prompt Tuning, and proposes learning "bursty prompts" via data whitening and a bilinear model. The approach substantially outperforms VPT and its variants across multiple benchmarks, e.g., improving accuracy on CUB-200 from 42.15% to 77.86%.

## Background & Motivation
Visual Prompt Tuning (VPT) is a parameter-efficient fine-tuning technique that adapts pretrained Vision Transformers by learning a small number of parameters in the input space, referred to as prompts. However, VPT performs poorly on many datasets, achieving only 42.15% accuracy on CUB-200. Subsequent works such as SPT improve performance through carefully designed prompt initialization, which raises a fundamental question: why is learning prompts directly so difficult?

The authors conduct an in-depth analysis of the distributional characteristics of prompt–data interactions within the self-attention module and identify two key issues:

**Burstiness**: A small number of elements in $\mathbf{W}_q\mathbf{W}_k^T\mathbf{X}^T$ exhibit extremely large absolute values.

**Non-Gaussian distribution**: $\mathbf{W}_q\mathbf{W}_k^T$ follows a super-Laplacian distribution, while patch embeddings $\mathbf{X}$ follow a Laplacian distribution.

These non-Gaussian distributions intuitively pose challenges for prompt learning. Motivated by data preprocessing (whitening) and the bilinear model, the authors propose learning "bursty prompts" to accommodate such data characteristics.

## Method

### Overall Architecture
VPT concatenates prompts $\mathbf{P} \in \mathbb{R}^{m \times d}$ with image patch embeddings $\mathbf{X} \in \mathbb{R}^{n \times d}$ and feeds them into the Transformer's self-attention module. During attention computation, prompts undergo multiple matrix-multiplication interactions with the query projection $\mathbf{W}_q$, key projection $\mathbf{W}_k$, and image tokens. The core idea of BPT is to represent prompts as the product of two matrices (a bilinear form), thereby facilitating the learning of "bursty" final prompts.

### Key Designs

1. **BPT-fWhiten (Fixed Whitening Matrix)**:

    - **Function**: Applies ZCA whitening to $\tilde{\mathbf{X}} = \mathbf{W}_q\mathbf{W}_k^T\mathbf{X}^T$ to decorrelate the data and normalize variance.
    - **Mechanism**: Computes the covariance matrix $\boldsymbol{\Sigma} = \frac{1}{N}\tilde{\mathbf{X}}\tilde{\mathbf{X}}^T$, obtains the whitening matrix $\mathbf{W} = \boldsymbol{\Sigma}^{-1/2} = \mathbf{U}\mathbf{S}^{-1/2}\mathbf{U}^T$ via SVD, and fixes the whitening matrix during prompt learning: $\tilde{\mathbf{P}} = \mathbf{P}\mathbf{W}^T$.
    - **Design Motivation**: Whitening transforms the non-Gaussian distribution into one closer to Gaussian, reducing the difficulty of prompt learning. An effective whitening matrix can be computed from as few as 100 images.

2. **BPT-tWhiten (Tunable Whitening Matrix)**:

    - **Function**: Jointly optimizes the whitening matrix alongside the prompt during training.
    - **Mechanism**: Since the model is differentiable with respect to the whitening matrix $\mathbf{W}$, both the prompt and the whitening matrix are optimized simultaneously: $\hat{\mathbf{P}}, \hat{\mathbf{W}} = \min_{\mathbf{P},\mathbf{W}} \ell(\mathbf{Y}, \text{MODEL}(\mathbf{X}; \mathbf{P}\mathbf{W}^T, \mathbf{H}, \boldsymbol{\Theta}))$.
    - **Design Motivation**: Tuning the whitening matrix allows further adaptation to downstream tasks, at the cost of introducing additional parameters.

3. **BPT-bilinear (Low-Rank Bilinear Prompt)**:

    - **Function**: Learns two compact matrices $\mathbf{A} \in \mathbb{R}^{m \times p}$ and $\mathbf{B} \in \mathbb{R}^{d \times p}$, whose product serves as the final prompt.
    - **Mechanism**: $\tilde{\mathbf{P}} = \mathbf{A}\mathbf{B}^T$, yielding a low-rank prompt when $p < d$. For example, with $m=100, d=768, p=25$, the parameter count is reduced from $15.1 \times 10^6$ (VPT) to $4.3 \times 10^6$ (a $3.5\times$ reduction).
    - **Design Motivation**: The bilinear operation naturally produces bursty features, and the rank $p$ provides flexible control over parameter count and computational cost.

### Implementation Details
The two-matrix multiplication structure in BPT is implemented via $1 \times 1$ convolutions (without bias or nonlinear activations). The deep variant inserts prompts into multiple Transformer blocks; by default, prompt tuning is applied to the top 4 blocks to balance performance and parameter efficiency.

## Key Experimental Results

### Main Results

| Dataset / Method | Mean Acc | CUB-200 | NABirds | Flowers | Dogs | Cars |
|------------------|----------|---------|---------|---------|------|------|
| VPT-S (MAE) | 57.84 | 42.15 | 57.43 | 69.15 | 77.07 | 43.38 |
| SPT-S (MAE) | 73.95 | 71.15 | 61.87 | 89.47 | 80.01 | 67.23 |
| **BPT-S (MAE)** | **80.39** | **77.86** | **72.03** | **90.37** | **81.91** | **79.77** |
| SPT-D (MAE) | 83.26 | 80.13 | 76.28 | 93.07 | 82.23 | 84.61 |
| **BPT-D (MAE)** | **84.60** | **82.00** | **78.49** | **93.72** | **82.67** | **86.11** |
| Full fine-tuning | 82.80 | 80.55 | 77.87 | 91.71 | 80.38 | 83.51 |

### Ablation Study

| Configuration | #params (×10⁻²M) | IN-1K | CUB-200 | Notes |
|---------------|------------------|-------|---------|-------|
| VPT | 7.68 | 63.71 | 42.15 | Baseline |
| SPT | 7.68 | 69.98 | 71.15 | Prev. SOTA |
| BPT-fWhiten (fixed) | 7.68 | 72.09 | 77.48 | Whitening yields substantial gain |
| BPT-tWhiten (tunable) | 66.66 | 72.37 | 78.54 | Best accuracy but higher param count |
| BPT-bilinear (random init) | 6.51 | 72.15 | 77.86 | Fewest params, near-optimal performance |

### Key Findings
- BPT-Shallow outperforms VPT on CUB-200 by **35.71 percentage points** (77.86 vs. 42.15).
- BPT-bilinear uses the fewest parameters (6.51 vs. 7.68 ×10⁻²M) and is robust to random initialization.
- BPT performance increases monotonically with prompt length, unlike VPT/GateVPT which are sensitive to this hyperparameter.
- BPT also surpasses VPT and SPT on COCO object detection.
- BPT trained for 100 epochs outperforms SPT trained for 400 epochs.

## Highlights & Insights
- This work is the first to reveal the burstiness phenomenon in Transformer self-attention modules and to connect it with the difficulty of prompt learning.
- A counterintuitive finding: when faced with bursty data, learning equally bursty prompts yields the best performance.
- The method is remarkably simple—a single bias-free $1 \times 1$ convolution—yet delivers substantial performance gains.
- BPT-bilinear achieves optimal performance with the fewest parameters and is insensitive to initialization, making it highly practical.

## Limitations & Future Work
- The analysis focuses solely on $\mathbf{P}\mathbf{W}_q\mathbf{W}_k^T\mathbf{X}^T$ and does not cover all interaction terms within the attention matrix.
- The mechanism by which burstiness facilitates prompt learning lacks theoretical explanation and is supported only by empirical observations.
- The behavior of BPT under class-imbalanced settings remains unexplored, and the influence of biases inherent to pretrained models is not analyzed.
- When training data is abundant (>30% of ImageNet), full fine-tuning still outperforms prompt tuning methods.

## Related Work & Insights
- **vs. VPT**: BPT substantially surpasses VPT through its bilinear structure, revealing that VPT's poor performance stems from data distribution characteristics rather than architectural limitations.
- **vs. SPT**: SPT relies on carefully designed initialization (patch token clustering), whereas BPT is robust to random initialization and achieves superior performance.
- **vs. LoRA/SSF and other fine-tuning methods**: Under the supervised pretraining setting, BPT-Deep achieves a mean accuracy of 91.72% with 18.36×10⁻²M parameters, surpassing all compared methods.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The discovery of burstiness and its connection to whitening and the bilinear model offers a unique and convincing perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluations span FGVC, ImageNet, COCO detection and segmentation, with multiple pretraining strategies and backbone scales.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-organized, with a coherent logical chain from observed phenomena to proposed method.
- **Value**: ⭐⭐⭐⭐⭐ The method is simple, efficient, and plug-and-play, making a significant contribution to the VPT literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[NeurIPS 2025\] VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion](../../NeurIPS2025/multimodal_vlm/vipamin_visual_prompt_initialization_via_embedding_selection_and_subspace_expans.md)

</div>

<!-- RELATED:END -->
