---
title: >-
  [Paper Note] Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Weight Pruning] MoT probe experiments reveal asymmetric pruning sensitivity between the text and visual pathways in LVLMs — the text pathway is highly sensitive and must be calibrated with tex…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Weight Pruning"
  - "LVLM"
  - "Modality Asymmetry"
  - "Calibration Strategy"
  - "Sparsification"
date: 2026-05-08
content_hash: 144e494f539b48ec
---

# Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.16001](https://arxiv.org/abs/2603.16001)  
**Code**: [https://github.com/LezJ/ATV-Pruning](https://github.com/LezJ/ATV-Pruning)  
**Area**: Multimodal VLM
**Keywords**: Weight Pruning, LVLM, Modality Asymmetry, Calibration Strategy, Sparsification

## TL;DR
MoT probe experiments reveal asymmetric pruning sensitivity between the text and visual pathways in LVLMs — the text pathway is highly sensitive and must be calibrated with text tokens, while the visual pathway is highly redundant and can tolerate 60% sparsity. Based on these findings, ATV-Pruning constructs a calibration pool using all text tokens plus a small, layer-adaptively selected subset of visual tokens.

## Background & Motivation

**Background**: LVLMs have enormous parameter counts, making weight pruning an effective approach for reducing deployment costs. SparseGPT and Wanda perform well on text-only LLMs, with the latter evaluating importance via weight magnitude × activation norm. However, direct application to LVLMs yields suboptimal results.

**Limitations of Prior Work**: Existing LVLM pruning methods (e.g., TAMP), while multimodal-aware, still process text and visual tokens within a unified framework, ignoring fundamental behavioral differences between the two modalities under pruning: (1) text and visual activations occupy distinct cluster regions in representation space (t-SNE visualization); (2) pruning masks derived from text-only vs. visual-only calibration exhibit a wide IoU distribution.

**Key Challenge**: Modality-agnostic calibration strategies dilute the linguistic signals necessary to protect text-related weights.

**Goal**: How to design calibration strategies that account for the distinct sensitivities of different modality pathways?

**Key Insight**: Explicitly decouple text and visual pathways via a Mixture-of-Transformer (MoT) analysis probe to independently investigate their respective pruning sensitivities.

**Core Idea**: The text pathway is calibrated using all text tokens (preserving sensitivity), while the visual pathway requires only a small number of high-saliency visual tokens as a supplement (exploiting redundancy).

## Method

### Overall Architecture
ATV-Pruning builds upon Wanda's activation-aware pruning framework. The key improvement lies in calibration pool construction: $\mathcal{S}_{cal} = \mathcal{T} \cup \mathcal{V}_{sub}$, where $\mathcal{T}$ contains all text tokens and $\mathcal{V}_{sub}$ is a layer-adaptively selected subset of visual tokens.

### Key Designs

1. **MoT Sensitivity Analysis Probe (Motivating Experiment)**:

    - Function: Decouple text/visual pathways and independently evaluate pruning sensitivity.
    - Mechanism: Duplicate the QKV and FFN components of each Transformer block into separate text and visual pathways; prune each using text/visual/mixed calibration pools, then compare performance.
    - Key Finding A: The text pathway is extremely sensitive — at 60% sparsity, text calibration retains 84.65% performance, visual calibration collapses to 50.92%, and mixed calibration achieves only 64.97%.
    - Key Finding B: The visual pathway is highly redundant — at 60% sparsity, any calibration strategy retains 99.25%+ performance.

2. **Modality-Aware Calibration Pool**:

    - Function: Adaptively construct a calibration pool containing all text tokens and a small number of visual tokens.
    - Mechanism: Per Finding A, all text tokens are retained to protect language capability; per Finding B, only a small number of visual tokens are needed to capture vision-specific weights.

3. **Layer-Adaptive Visual Token Selection**:

    - Function: Select the most important visual tokens at each Transformer block.
    - Saliency Metric: Token representation drift (visual drift) $s_v = 1 - \cos(\mathbf{X}_{in,v}, \mathbf{X}_{out,v})$.
    - Intuition: If a block substantially updates the representation of a visual token, that token is actively engaged in computation within the block and should be included in calibration.
    - The top-$k$ visual tokens with the largest drift are added to the calibration pool.

### Loss & Training
- Uses Wanda's importance score $\mathbf{I}_{ij} = |\mathbf{W}_{ij}| \cdot \|\mathbf{X}_j\|_2$.
- Row-wise pruning of the lowest $\rho\%$ scores to obtain an unstructured sparse model.
- No retraining required; this is a post-hoc pruning approach.

## Key Experimental Results

### MoT Probe Experiments (LLaVA-NeXT)

| Pathway | Calibration Source | Mean @ 50% Sparsity | Mean @ 60% Sparsity |
|---|---|---|---|
| Text Pathway | Text | 98.26% | 84.65% |
| Text Pathway | Visual | 94.33% | 50.92% |
| Text Pathway | Mixed | 95.86% | 64.97% |
| Visual Pathway | Text | 100.27% | 100.05% |
| Visual Pathway | Visual | 99.37% | 99.25% |
| Visual Pathway | Mixed | 100.14% | 99.57% |

### Main Results (9 Multimodal Benchmarks)

| Method | Sparsity | Multi-Benchmark Avg. | vs. Wanda | vs. TAMP |
|---|---|---|---|---|
| ATV-Pruning | 50% | **Best** | Significantly better | Exceeds |
| ATV-Pruning | 60% | **Best** | Substantially better | Exceeds |

## Highlights & Insights
- The MoT probe experimental design is elegant, providing the first quantitative characterization of asymmetric pruning sensitivity between text and visual pathways in LVLMs.
- The method is remarkably simple — it modifies only the calibration token selection strategy on top of Wanda, yet achieves significant performance gains.
- The finding that the visual pathway suffers almost no performance degradation at 60% sparsity is a highly valuable empirical result.
- Visual drift serves as a token saliency metric that is intuitive, effective, and computationally lightweight.
- ATV-Pruning comprehensively outperforms baselines including Wanda, SparseGPT, and TAMP across 9 standard multimodal benchmarks.
- Finding B indicates substantial redundancy in the visual processing parameters of LVLMs, offering a new perspective for model compression.

### Experimental Thoroughness
- Results are validated consistently across multiple models including LLaVA-NeXT and Qwen2-VL.
- At 50% sparsity, ATV-Pruning retains 90%+ performance on MMBench, substantially outperforming vanilla Wanda.
- The advantage is most pronounced on SQA-img, which places the highest demands on textual reasoning ability.
- Visual token ratios ranging from 5% to 30% all yield competitive results; the default setting of 10% achieves the best trade-off.

## Limitations & Future Work
- Visual drift computation requires an additional forward pass (though this is a one-time cost during calibration).
- The top-$k$ ratio for visual token selection requires hyperparameter tuning, and the optimal ratio may vary across models and tasks.
- The current work only validates unstructured sparsity; structured pruning scenarios (e.g., channel pruning) warrant further exploration.
- Extending the asymmetric paradigm to other compression techniques such as quantization and knowledge distillation is a promising direction.
- The MoT probe decoupling is used for analysis only; actual pruning still operates on shared weights, and a gap may exist between probe findings and practical implementation.
- For video-input LVLMs where visual token counts grow substantially, the scalability of the selection strategy requires verification.
- The phenomenon of post-pruning performance improvement on VizWiz merits deeper investigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](tipsv2_patch_text_alignment.md)
- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](comp_collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)

</div>

<!-- RELATED:END -->
