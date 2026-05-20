---
title: >-
  [Paper Note] Nautilus: A Large Multimodal Model for Underwater Scene Understanding
description: >-
  [NeurIPS 2025][Multimodal VLM][underwater scene understanding] This paper presents Nautilus, the first large multimodal model supporting eight underwater scene understanding tasks. It introduces a physics-prior-driven Vi…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "underwater scene understanding"
  - "large multimodal model"
  - "visual feature enhancement"
  - "underwater imaging model"
  - "instruction tuning"
date: 2026-05-08
content_hash: c634c53085af9f6e
---

# Nautilus: A Large Multimodal Model for Underwater Scene Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2510.27481](https://arxiv.org/abs/2510.27481)  
**Code**: [GitHub](https://github.com/H-EmbodVis/NAUTILUS)  
**Area**: Multimodal Vision-Language Models (Multimodal VLM)
**Keywords**: underwater scene understanding, large multimodal model, visual feature enhancement, underwater imaging model, instruction tuning

## TL;DR

This paper presents Nautilus, the first large multimodal model supporting eight underwater scene understanding tasks. It introduces a physics-prior-driven Visual Feature Enhancement (VFE) module that explicitly rectifies underwater image degradation in feature space, improving the robustness of LMMs in underwater environments.

## Background & Motivation

**Background**: Underwater scene understanding is critical for ocean exploration, encompassing multi-granularity tasks such as object detection, counting, and image captioning. Existing underwater methods are predominantly designed for single tasks, while general-purpose LMMs perform poorly when directly applied to underwater scenes.

**Limitations of Prior Work**: (1) General LMMs suffer from an aerial-to-underwater domain shift; (2) underwater light scattering and absorption cause severe image degradation; (3) large-scale multi-task instruction tuning datasets for underwater scenarios are absent.

**Key Challenge**: Underwater scenes require comprehensive understanding at multiple granularities (image-level, region-level, and object-level), yet both the data and effective mechanisms for handling underwater degradation are lacking.

**Goal**: Construct NautData, an underwater instruction tuning dataset covering eight tasks, and design an LMM capable of explicitly handling underwater image degradation.

**Key Insight**: Leverage prior knowledge from the physical underwater imaging model to perform image enhancement in feature space rather than pixel space.

**Core Idea**: Quantify backscatter influence via the dark pixel prior and recover light absorption attenuation using depth information, yielding a plug-and-play visual feature enhancement module.

## Method

### Overall Architecture

Nautilus comprises five core components: an image encoder $\mathcal{I}_v$, a depth encoder $\mathcal{I}_d$, a vision-language projector $\mathcal{P}_{v-l}$, a Visual Feature Enhancement (VFE) module, and an LLM. Given an underwater image, visual and depth features are extracted separately; the VFE module enhances visual features using physical priors. Both the original and enhanced features are passed in parallel through a shared projector to be aligned into the language space, after which the LLM performs multimodal reasoning.

### Key Designs

1. **NautData Dataset Construction**: Contains 1.45 million image-text pairs covering eight underwater tasks (coarse-grained/fine-grained classification, counting, VQA, detection, grounding, region captioning, and image captioning). Data generation adopts three strategies: rule-based template generation, integrated generation (templates combined with LMM outputs), and free-form generation (open-ended LMM question answering). A multi-stage quality control pipeline is employed: Gemini 2.0 Flash for initial generation, Qwen2.5-VL-72B for quality assessment, and GPT-4o for test set validation.

2. **Underwater Imaging Physical Prior**: An underwater image is modeled as the superposition of direct signal $\bm{D}_c$ and backscatter $\bm{B}_c$:
$$\bm{I}_c = \bm{D}_c + \bm{B}_c, \quad \bm{D}_c = \bm{J}_c e^{-\beta_c(\bm{z}) \cdot \bm{z}}$$
   The enhancement objective is to recover the unattenuated original color $\bm{J}_c$:
$$\bm{J}_c = \frac{\bm{I}_c - \bm{B}_c}{e^{-\beta_c(\bm{z}) \cdot \bm{z}}}$$
   Backscatter intensity is quantified via the dark pixel prior, and attenuation coefficients are fitted using depth information.

3. **Visual Feature Enhancement (VFE) Module**: Operates in two steps—

    - **Backscatter Removal**: The image patch with the lowest mean RGB value is identified as the dark token $\bm{f}_{v,k}$. Global semantics $\bm{q}$ are extracted via a cross-attention layer, and the backscatter estimate is computed as $\bm{s} = \bm{f}_{v,k} - \bm{q}$, which is then subtracted pixel-wise from the global features.
    - **Light Absorption Recovery**: An MLP predicts absorption weights $\bm{W} = \text{MLP}(\bm{d})$ from depth features. The final enhanced features are:
$$\bm{v}_e = (\bm{v} - \bm{s}) \oslash \exp(-\bm{W})$$

4. **Dual-Path Feature Fusion**: The original visual features preserve authentic underwater environmental information, while the enhanced features reduce imaging interference. Both are fed in parallel to the LLM through a shared projector, enabling complementary understanding.

### Loss & Training

- Parameter-efficient fine-tuning (PEFT) is adopted; trainable components include the vision-language projector, LoRA (rank=128), and the VFE module.
- The framework is adapted on two baselines: LLaVA-1.5 and Qwen2.5-VL.
- Training runs for 1 epoch on 4×A800-80GB GPUs, taking approximately 3 days.

## Key Experimental Results

### Main Results

Comparison with prominent LMMs on the NautData test set:

| Method | Coarse Cls. Acc | Fine Cls. Acc | Captioning METEOR | Grounding PR@0.5 | Detection mAP@0.5 | VQA METEOR |
|--------|----------------|--------------|------------------|-----------------|------------------|-----------|
| GPT-4o (zero-shot) | 55.2 | 54.4 | 0.179 | 4.3 | 1.4 | 0.242 |
| Qwen2.5-VL-72B (zero-shot) | 55.2 | 54.2 | 0.171 | 46.4 | 14.7 | 0.222 |
| LLaVA-1.5 | 90.0 | 89.8 | 0.208 | 48.2 | 19.0 | 0.359 |
| Qwen2.5-VL | 85.3 | 88.2 | 0.222 | 57.6 | 41.7 | 0.380 |
| **Nautilus (Qwen2.5-VL)** | **90.3** | **93.8** | **0.223** | **58.8** | **45.3** | **0.381** |

### Ablation Study

Ablation results with incremental module addition (Qwen2.5-VL baseline):

| Baseline | Depth Encoder | Absorption Recovery | Backscatter Removal | Coarse Cls. Acc | Fine Cls. Acc | Grounding PR@0.5 | Detection AP@0.5 |
|----------|--------------|--------------------|--------------------|----------------|--------------|-----------------|-----------------|
| ✔ | - | - | - | 87.9 | 89.1 | 55.4 | 35.9 |
| ✔ | ✔ | - | - | 89.5 | 89.1 | 55.0 | 36.4 |
| ✔ | ✔ | ✔ | - | 85.7 | 91.2 | 53.9 | 34.2 |
| ✔ | ✔ | ✔ | ✔ | **90.0** | **91.4** | **55.9** | **36.2** |

### Key Findings

- Zero-shot commercial LMMs (GPT-4o, Gemini 2.0 Flash) perform substantially worse than fine-tuned open-source models on underwater scenes.
- The VFE module consistently improves performance across most tasks on both baselines (fine-grained classification +5.6% and detection mAP@0.5 +3.6% on Qwen2.5-VL).
- Zero-shot evaluation on MarineInst20M validates generalization capability.
- Compared to pixel-space enhancement methods (Reti-Diff, SMDR-IS, etc.), feature-space enhancement avoids information loss.

## Highlights & Insights

- This work is the first to inject physical imaging model priors into LMM feature-space enhancement, offering a novel approach with clear physical interpretability.
- NautData establishes a large-scale underwater instruction tuning dataset covering eight tasks, filling a critical data gap in the field.
- The VFE module is designed as plug-and-play and can be flexibly integrated into different LMM frameworks.
- The derivation chain—dark pixel prior → backscatter quantification → feature subtraction—is physically grounded and logically coherent.

## Limitations & Future Work

- Multi-task joint optimization introduces inter-task conflicts (e.g., a slight drop in counting accuracy metrics).
- Validation is limited to 7B/8B scale models; the effectiveness at larger scales remains unknown.
- Depth estimation relies on a frozen Depth Anything V2, whose cross-domain generalization is yet to be verified.
- The dark pixel prior assumption may fail under extreme underwater conditions (e.g., complete darkness or intense illumination).

## Related Work & Insights

- MarineGPT: the first publicly available underwater LMM, but supports image-level understanding only.
- MarineInst20M: a large-scale underwater vision-language dataset supporting object-level description.
- The paradigm of using physical models to guide deep learning is generalizable to other degradation scenarios (e.g., haze, low-light).
- The finding that feature-space enhancement outperforms pixel-space enhancement provides a useful reference for other domain adaptation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Physics-prior-driven feature-space enhancement is innovative, though the overall framework builds upon existing LMMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across eight tasks with thorough ablations and zero-shot generalization verification.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed physical derivations, and rich figures and tables.
- Value: ⭐⭐⭐⭐ A pioneering contribution to underwater scene understanding; both the dataset and the method offer practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs](towards_comprehensive_scene_understanding_integrating_first_and_third-person_vie.md)
- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](act_as_human_multimodal_large_language_model_data_annotation.md)
- [\[NeurIPS 2025\] Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models](situat3dchange_situated_3d_change_understanding_dataset_for_multimodal_large_lan.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)

</div>

<!-- RELATED:END -->
