---
title: >-
  [Paper Note] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][Few-shot learning] This paper proposes VT-FSL, a framework that leverages Cross-modal Iterative Prompting (CIP) to jointly exploit class names and support images for driving LLMs to generat…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Few-shot learning"
  - "large language models"
  - "cross-modal alignment"
  - "text prompting"
  - "visual synthesis"
  - "contrastive learning"
date: 2026-05-08
content_hash: 50a62319d73a0edd
---

# VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.25033](https://arxiv.org/abs/2509.25033)  
**Authors**: Wenhao Li, Qiangchang Wang, Xianjing Meng, Zhibin Wu, Yilong Yin (Shandong University, Shenzhen Loop Area Research Institute, Shandong University of Finance and Economics)
**Code**: [GitHub](https://github.com/peacelwh/VT-FSL)  
**Area**: Multimodal VLM
**Keywords**: Few-shot learning, large language models, cross-modal alignment, text prompting, visual synthesis, contrastive learning

## TL;DR

This paper proposes VT-FSL, a framework that leverages Cross-modal Iterative Prompting (CIP) to jointly exploit class names and support images for driving LLMs to generate accurate, visually grounded textual descriptions and zero-shot synthesize semantically consistent images. Combined with Kernelized Volume Contrastive Learning (CGA) for global nonlinear cross-modal alignment, VT-FSL achieves an average classification accuracy improvement of 4.2% across 10 few-shot learning benchmarks.

## Background & Motivation

### Problem Background
Few-shot learning (FSL) aims to recognize novel categories from only a handful of labeled samples. Metric-learning approaches construct class prototypes for nearest-neighbor classification, but limited samples cause prototypes to deviate from the true class centers. Incorporating semantic information from the text modality is a promising direction for improving prototype representations.

### Limitations of Prior Work
- **Insufficient class-name information**: Methods such as AM3 and SP rely solely on class names, providing extremely limited contextual information.
- **Semantic hallucination**: Methods such as SemFew and ECER leverage LLMs to generate richer descriptions but condition only on class names, ignoring visual patterns in support images. This leads to inconsistencies between generated text and actual visual evidence, requiring additional manual or algorithmic correction.
- **Naive prompting strategies**: Existing methods apply simple input prompts to LLMs without fully exploiting their reasoning and generation capabilities.
- **Limited alignment paradigm**: CLIP-style pairwise contrastive learning aligns each representation to a single anchor, neglecting the global structural relationships across modalities.

### Core Motivation
The paper simultaneously exploits class names and support images to generate visually grounded textual descriptions, eliminating semantic hallucinations. It further generates complementary textual and visual prompts that provide high-level class semantics and low-level intra-class diversity, respectively, and achieves consistent fusion of all feature modalities via geometry-aware alignment.

## Method

### Overall Architecture
VT-FSL consists of two core modules: Cross-modal Iterative Prompting (CIP) and Cross-modal Geometric Alignment (CGA). CIP is responsible for generating textual and visual prompts, while CGA performs global alignment across all modal representations.

### Cross-modal Iterative Prompting (CIP)
CIP jointly leverages class labels and $K$-shot support images to generate accurate, visually grounded class descriptions through structured reasoning. Inspired by Chain-of-Thought, the generation process is decomposed into four stages:

1. **Strategy**: Outlines the problem and specifies the key visual attributes to be extracted from support images.
2. **Perception**: Interprets visual patterns in the support images and extracts shared category-level features.
3. **Refinement**: Iteratively reasons to eliminate inconsistencies and improve the quality of textual descriptions.
4. **Conclusion**: Produces the final, precise class description.

Each stage is marked with structured labels, and the entire process is completed within a **single inference pass**, eliminating the need for multi-turn interactions and reducing latency and manual effort.

The generated descriptions are subsequently fed into a text-to-image model (Janus-Pro) to zero-shot synthesize semantically consistent images. An LLM-based pairwise comparison strategy selects the top-$K$ images most aligned with the textual descriptions, constructing an augmented $N$-way $(K+K)$-shot support set.

### Cross-modal Fusion
CLIP-encoded text features $Z_t$ are fused with support features $z_s$ along two dimensions:
- **Channel dimension**: A two-layer MLP generates a modulation vector $\beta$ to enhance support features along the channel dimension.
- **Spatial dimension**: $Z_t$ and $z_s$ are concatenated along the spatial dimension, and multi-head self-attention in a Transformer captures inter-token semantic correlations to produce enhanced support embeddings $Z_s$.

### Cross-modal Geometric Alignment (CGA) — Kernelized Volume Contrastive Learning
Unlike conventional pairwise contrastive learning, CGA measures alignment by computing the volume of the $k$-dimensional parallelepiped spanned by multiple vectors:

$$\mathrm{Vol}_{\mathcal{H}}(\mathbf{v}_1,\dots,\mathbf{v}_k)=\sqrt{\det(\mathbf{K})},\quad \mathbf{K}_{ij}=\kappa(\mathbf{v}_i,\mathbf{v}_j)$$

where $\kappa$ is an RBF kernel function that extends alignment to a Reproducing Kernel Hilbert Space (RKHS) to capture nonlinear relationships. Using the text modality as an anchor, bidirectional contrastive losses $\mathcal{L}_{\text{D2A}}$ and $\mathcal{L}_{\text{A2D}}$ are defined, and the total loss is:

$$\mathcal{L}_{\text{total}}=\sum_{i=1}^{M}\text{CE}(p_i,y_i)+\frac{1}{2}(\mathcal{L}_{\text{D2A}}+\mathcal{L}_{\text{A2D}})$$

### Inference
Text-augmented prototypes $c_t$ and visually augmented prototypes $c_v$ are computed separately and combined via a convex combination to form the final classification prototype: $C=uc_t+(1-u)c_v$, where the fusion factor $u$ is determined by grid search on the validation set.

## Key Experimental Results

### Experiment 1: Standard Few-Shot Classification (miniImageNet & tieredImageNet)

| Method | Backbone | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
|--------|----------|-------------------|-------------------|---------------------|---------------------|
| ProtoNet | ResNet-12 | 62.39 | 80.53 | 68.23 | 84.03 |
| SemFew (CVPR'24) | Swin-T (29M) | 78.94 | 86.49 | 82.37 | 89.89 |
| UAP (NeurIPS'24) | ResNet-12 | 81.63 | 79.05 | 79.68 | 76.78 |
| ECER (AAAI'25) | Visformer-T | 81.14 | - | 81.81 | - |
| **VT-FSL** | **Visformer-T (10M)** | **83.66** | **88.38** | **88.02** | **91.71** |

VT-FSL employs the lightweight Visformer-T backbone (10M parameters) and outperforms all methods using larger backbones (ViT-S 22M, Swin-T 29M, WRN28-10 36.5M). Compared to SemFew, VT-FSL achieves a 1-shot improvement of 3.7%–5.7%.

### Experiment 2: Fine-Grained and Cross-Domain Few-Shot Classification

| Setting | Dataset | 2nd Best 1-shot | VT-FSL 1-shot | Gain |
|---------|---------|----------------|--------------|------|
| Fine-grained | CUB-200 | 86.02 (SUITED) | **91.08** | +5.06 |
| Fine-grained | Stanford-Dogs | 76.55 (SUITED) | **86.58** | +10.03 |
| Fine-grained | Stanford-Cars | 89.97 (SUITED) | **92.95** | +2.98 |
| Cross-domain | mini→CUB | 51.55 (MEFP) | **66.86** | +15.31 |
| Cross-domain | mini→Places | 59.07 (SVasP) | **73.68** | +14.61 |
| Cross-domain | mini→Plantae | 41.55 (MEFP) | **45.90** | +4.35 |

Gains are particularly pronounced in the cross-domain 1-shot setting (up to +15.31%), demonstrating the transferability of cross-modal semantics.

### Ablation Study

| Text Prompt | Visual Prompt | Alignment Loss | miniImageNet 1-shot | CIFAR-FS 1-shot |
|------------|--------------|---------------|-------------------|----------------|
| ✗ | ✗ | ✗ | 68.47 | 76.43 |
| ✓ | ✗ | ✗ | 78.82 | 84.76 |
| ✓ | ✓ | ✗ | 82.08 | 87.72 |
| ✓ | ✓ | ✓ | **83.66** | **88.67** |

All three components consistently contribute: text prompting yields the largest gain (+10.35), visual prompting provides further improvement (+3.26), and the alignment loss adds a complementary boost (+1.58).

### Efficiency Comparison

| Method | Prompt Gen. (h) | Training (min) | Inference (ms) | Accuracy (%) |
|--------|----------------|---------------|---------------|-------------|
| SP | - | 1.7 | 78 | 72.31 |
| SemFew | 1.5 | 2.3 | 105 | 78.94 |
| ECER | 0.7 | 3.0 | 119 | 81.14 |
| **VT-FSL** | **0.7** | **1.1** | **76** | **83.66** |

VT-FSL achieves the fastest training, fastest inference, and highest accuracy, as cross-modal prompts are generated offline in a one-time pass without introducing additional overhead to the downstream model.

## Highlights & Insights

- **Visually grounded text generation**: This work is the first to jointly condition LLMs on class names and support images, eliminating the semantic hallucination problem inherent in purely text-driven methods. The four-stage structured reasoning is completed within a single inference pass.
- **Complementary cross-modal prompt design**: Text prompts supply high-level class semantics while synthesized images provide low-level intra-class diversity, forming a natural complement. The plug-in design allows these prompts to be directly used by any downstream FSL model.
- **Kernelized volume contrastive learning**: This is the first work to introduce volume-based contrastive learning into FSL. The kernel Gram matrix determinant in RKHS captures global nonlinear cross-modal relationships, outperforming both InfoNCE and linear volume-based approaches.
- **Comprehensive state-of-the-art**: VT-FSL achieves new state-of-the-art results across 10 benchmarks (standard, fine-grained, and cross-domain) with an average improvement of 4.2%, using the lightest backbone among competing methods.

## Limitations & Future Work

- **Dependence on LLM and generative model quality**: CIP relies on the visual understanding and reasoning capabilities of Qwen2.5-VL-32B, and image synthesis depends on the generation quality of Janus-Pro; performance with weaker models remains unknown.
- **Offline generation overhead**: Although prompt generation is a one-time process, it still requires 0.7 hours for large-scale datasets and demands GPU resources to run large models.
- **Manual search for the fusion factor**: The inference-time fusion factor $u$ requires grid search on the validation set, lacking full automation.
- **Sensitivity to the number of synthesized images**: Performance degrades when the number of synthesized images exceeds $K$, indicating a ceiling on generation quality.
- **Validation limited to image classification**: Generalization to more complex few-shot tasks such as detection and segmentation has not been verified.
- **High computational requirements**: The method requires GPUs at the level of NVIDIA RTX 6000 Ada, which is unfriendly to resource-constrained settings.

## Related Work & Insights

- **AM3 (NeurIPS'19)**: Adaptively fuses semantic features derived from class names with visual prototypes; information content is limited. VT-FSL generates rich, visually grounded descriptions instead.
- **CaFo (CVPR'22)**: Synthesizes images from class names to augment data, but textual descriptions lack visual grounding. VT-FSL extracts visual cues from support images to eliminate hallucinations.
- **SemFew (CVPR'24)**: Uses LLMs conditioned on class names to generate coherent descriptions for prototype enhancement, but naive prompting limits semantic quality and requires a 4.3M-parameter fusion module. VT-FSL requires only a two-layer MLP with 0.7M parameters.
- **ECER (AAAI'25)**: Extracts attribute-level textual information conditioned solely on class names. VT-FSL jointly conditions on class names and images with structured reasoning for more precise descriptions.
- **InfoNCE contrastive learning**: Pairwise alignment ignores global structural relationships across modalities. VT-FSL's kernelized volume loss simultaneously accounts for global geometric relationships among all three modalities.
- **SUITED (AAAI'25)**: The previous best method for fine-grained FSL. VT-FSL surpasses it by 3.0%–10.3% in the 1-shot setting.

## Rating

- Novelty: ⭐⭐⭐⭐ — Kernelized volume contrastive learning and the visually grounded CIP design are novel, though the overall framework is a sophisticated combination of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Ten benchmarks, three settings, detailed ablations, efficiency comparisons, and visualization analyses constitute an exceptionally comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ — The paper is well-structured with clear motivation, rich figures and tables, and complete mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ — Achieves comprehensive state-of-the-art results with a practical, plug-in design suitable for deployment; source code is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[NeurIPS 2025\] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)
- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[NeurIPS 2025\] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning](cyin_cyclic_informative_latent_space_for_bridging_complete_and_incomplete_multim.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)

</div>

<!-- RELATED:END -->
