---
title: >-
  [Paper Note] Diffusion Self-Distillation for Zero-Shot Customized Image Generation
description: >-
  [CVPR 2025][Image Generation][Self-Distillation] This paper proposes Diffusion Self-Distillation (DSD), which leverages the emergent capability of pre-trained T2I models to generate consistent grid images in order to automatically construct identity-preserving paired datasets (using LLMs for prompt generation and VLMs for filtering). The same model is then fine-tuned to achieve zero-shot identity-preserving image generation, delivering results close to DreamBooth without requ…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Self-Distillation"
  - "Zero-Shot Customization"
  - "Identity-Preserving Generation"
  - "Diffusion Models"
  - "Synthetic Paired Data"
date: 2026-05-08
content_hash: 8890404c18a8fed3
---

# Diffusion Self-Distillation for Zero-Shot Customized Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.18616](https://arxiv.org/abs/2411.18616)  
**Code**: [https://primecai.github.io/dsd](https://primecai.github.io/dsd)  
**Area**: Image Generation / Personalization  
**Keywords**: Self-Distillation, Zero-Shot Customization, Identity-Preserving Generation, Diffusion Models, Synthetic Paired Data

## TL;DR
This paper proposes Diffusion Self-Distillation (DSD), which leverages the emergent capability of pre-trained T2I models to generate consistent grid images in order to automatically construct identity-preserving paired datasets (using LLMs for prompt generation and VLMs for filtering). The same model is then fine-tuned to achieve zero-shot identity-preserving image generation, delivering results close to DreamBooth without requiring test-time optimization.

## Background & Motivation

**Background**: Creators frequently need to maintain a consistent identity for characters or assets across different contexts—the core requirement of "identity-preserving generation." While DreamBooth and LoRA achieve this, they require per-instance fine-tuning, which incurs high computational costs as model sizes scale up (e.g., FLUX 12B). Zero-shot methods (such as IP-Adapter and InstantID) eliminate the need for fine-tuning but often fail to robustly preserve identities or are restricted strictly to human faces.

**Limitations of Prior Work**: The core obstacle is the lack of large-scale identity-preserving paired datasets. Manually collecting paired data showcasing the "same character in different scenarios" is both expensive and difficult to scale.

**Key Challenge**: Supervised training requires paired data $\rightarrow$ but paired data does not exist in scale $\rightarrow$ existing zero-shot methods must rely on unsupervised or weakly supervised training $\rightarrow$ resulting in performance inferior to supervised per-instance fine-tuning.

**Goal**: To break this deadlock by enabling the model to generate its own paired data for self-training.

**Key Insight**: Recent T2I models (such as SD3 and FLUX) exhibit an emergent capability of generating consistent grid images (likely originating from exposure to comic strips, photo albums, etc., in their training data), which can produce "four different scenes of the same character."

**Core Idea**: (1) Use T2I models to generate consistent grid images $\rightarrow$ prompt variety via LLMs $\rightarrow$ screen for genuinely identity-consistent pairs using VLMs $\rightarrow$ construct a large-scale synthetic paired dataset; (2) Extend the T2I diffusion transformer into a "two-frame video" architecture (inputting a reference frame and generating a target frame) and perform supervised training on this synthetic dataset.

## Method

### Overall Architecture
Three phases: (1) LLM generates diverse grid prompts based on LAION captions; (2) T2I model generates grid images $\rightarrow$ crop them into pairs $\rightarrow$ VLM uses Chain-of-Thought reasoning to judge identity consistency $\rightarrow$ filter and retain; (3) Extend the diffusion transformer to process two frames in parallel, where the first frame reconstructs the reference image (identity mapping) and the second frame generates the conditionally edited result.

### Key Designs

1. **Self-Distillation Data Generation Pipeline**:

    - Function: Fully automated construction of identity-preserving paired datasets
    - Mechanism: Utilizing LAION image captions as references, an LLM generates grid prompts (e.g., "four-panel grid showing the same [character] in different scenes"). The T2I model generates these grids, which are then cropped into pairs. A VLM employs Chain-of-Thought to determine if the subjects in the two images match, keeping only consistent pairs. The entire process requires no human intervention.
    - Design Motivation: To leverage the "emergent capabilities" stemming from comics or photo albums in the T2I model's training data. The model already possesses the capability to paint "the same character in different frames"; it only needs guidance and screening.

2. **Parallel Processing Architecture**:

    - Function: A general image-to-image conditional generation framework
    - Mechanism: The input reference image is treated as the first frame of a "two-frame video," and the output consists of two frames. The first frame is the reference image reconstruction (identity mapping), while the second frame is the conditionally edited result. Both frames are processed within the same DiT, natively exchanging information through the attention mechanism.
    - Design Motivation: Unlike ControlNet (structure-preserving editing) or IP-Adapter (concept extraction), the parallel two-frame architecture permits fine-grained identity information transfer without requiring spatial alignment.

3. **Automated VLM Filtering**:

    - Function: Ensuring the quality of synthetic paired data
    - Mechanism: For each candidate pair of images, a VLM sequentially performs: (a) identifying the common subject of the two images; (b) describing the details of each; (c) determining whether they refer to the "same" entity. Chain-of-Thought reasoning significantly improves the accuracy of the decisions.
    - Design Motivation: Grid images generated by T2I models can be noisy (not always strictly preserving identity). Filtering via VLMs converts the problem from unsupervised learning to supervised learning.

### Loss & Training
Standard diffusion denoising loss. Weighting of the two-frame loss is applied, as both the first frame (reconstruction) and the second frame (generation) are optimized simultaneously.

## Key Experimental Results

### Main Results

| Method | Identity Pres. ↑ | Diversity ↑ | Test-Time Optimization |
|------|----------|---------|-----------|
| DreamBooth | Highest | Medium | Required (Slow) |
| IP-Adapter+ | Low | High | Not Required |
| InstantID | Medium (Face Only) | Medium | Not Required |
| DSD (Ours) | Close to DreamBooth | High | Not Required |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| W/o VLM Filtering | Degraded Quality | Noisy data hurts training |
| W/o LLM Diverse Prompts | Poor Diversity | Model prefers repetitive themes |
| ControlNet Architecture | Poor Identity Pres. | Not suited for non-aligned identity editing |
| IP-Adapter Architecture | Poor Identity Pres. | Image encoder bottleneck |

### Key Findings
- Zero-shot methods can approach DreamBooth-level identity preservation—self-distilled data + supervised training closes the performance gap.
- The parallel two-frame architecture can perform both identity preservation (non-structure-preserving) and structure-preserving editing (e.g., relighting, depth control).
- Critical role of data diversity: Referencing LLM + LAION captions significantly boosts data coverage.

## Highlights & Insights
- **Leveraging Emergent Capabilities for Self-Evolution**: Allowing the model to leverage its own implicit knowledge to generate training data for self-improvement. This "self-distillation" paradigm holds extensive potential for future applications.
- **Fully Automated Data Engineering with LLMs + VLMs**: A fully automated pipeline spanning prompt generation $\rightarrow$ image generation $\rightarrow$ quality filtering, which can be easily repurposed for other data construction scenarios.
- **Unified Image-to-Image Architecture**: Viewing the generation as a two-frame video enables the model to perform identity preservation for both human faces and generic objects, as well as tasks like relighting.

## Limitations & Future Work
- The upper bound of generation quality is constrained by the teacher model.
- Error accumulation across the automated pipeline (noise is introduced at each step: LLM prompt $\rightarrow$ T2I $\rightarrow$ VLM filtering).
- Currently evaluated primarily on static images; video consistency remains unexplored.

## Related Work & Insights
- **vs DreamBooth/LoRA**: Requires per-instance optimization, whereas DSD is zero-shot and ready out-of-the-box.
- **vs IP-Adapter/InstantID**: Suffers from a lack of supervised paired data, which DSD Status compensates for via self-distillation.
- **vs BootComp**: Shares a similar concept of constructing synthetic data but is restricted to the fashion domain, whereas DSD is designed for general-purpose identity preservation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A complete pipeline combining self-distilled paired data and a parallel two-frame architecture, leveraging emergent capabilities to bridge the supervision gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared against various methods alongside multi-task evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and a highly systematic methodology.
- Value: ⭐⭐⭐⭐⭐ A significant breakthrough in zero-shot personalized generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Z-Magic: Zero-shot Multiple Attributes Guided Image Creator](z-magic_zero-shot_multiple_attributes_guided_image_creator.md)
- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] Emuru: Zero-Shot Styled Text Image Generation, but Make It Autoregressive](zero-shot_styled_text_image_generation_but_make_it_autoregressive.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [\[ECCV 2024\] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](../../ECCV2024/image_generation/multigen_zero-shot_image_generation_from_multi-modal_prompts.md)

</div>

<!-- RELATED:END -->
