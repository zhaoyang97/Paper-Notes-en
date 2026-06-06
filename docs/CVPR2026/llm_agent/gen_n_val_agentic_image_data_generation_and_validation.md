---
title: >-
  [Paper Note] Gen-n-Val: Agentic Image Data Generation and Validation
description: >-
  [CVPR 2026][LLM Agent][Data Augmentation] This paper proposes Gen-n-Val, an agentic synthetic data generation and validation framework that leverages an LLM to optimize Layer Diffusion prompts for generating high-quality…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "Data Augmentation"
  - "Synthetic Data"
  - "Agentic Data Generation"
  - "Long-Tail Distribution"
  - "Instance Segmentation"
date: 2026-05-08
content_hash: 110cc651fd05c964
---

# Gen-n-Val: Agentic Image Data Generation and Validation

**Conference**: CVPR 2026
**arXiv**: [2506.04676](https://arxiv.org/abs/2506.04676)  
**Code**: [GitHub](https://github.com/aiiu-lab/Gen-n-Val)  
**Area**: LLM Agent
**Keywords**: Data Augmentation, Synthetic Data, Agentic Data Generation, Long-Tail Distribution, Instance Segmentation

## TL;DR
This paper proposes Gen-n-Val, an agentic synthetic data generation and validation framework that leverages an LLM to optimize Layer Diffusion prompts for generating high-quality single-object transparent images, and employs a VLLM to filter low-quality samples. The framework reduces invalid synthetic data from 50% to 7%, achieving a 7.6% mAP improvement on LVIS rare-category instance segmentation.

## Background & Motivation

1. **Background**: Large-scale datasets (e.g., LVIS with 1,203 categories) exhibit severe long-tail distributions—rare categories appear in fewer than 10 images. Synthetic data is an important remedy for data scarcity. Existing approaches include Copy-Paste augmentation and diffusion-model-based generation (e.g., X-Paste, MosaicFusion).
2. **Limitations of Prior Work**: MosaicFusion generates segmentation masks using cross-attention maps, but approximately 50% of data is filtered out, and among the remaining data, roughly 50% still suffers from: (1) a single mask covering multiple objects; (2) inaccurate segmentation masks; (3) incorrect category labels. Directly using Layer Diffusion with standard prompts yields approximately 44% invalid data, as monotonous and vague descriptions lead to low diversity and extraneous objects.
3. **Key Challenge**: High-quality synthetic data requires "single object + precise mask + correct category + high diversity," yet standard prompts cannot simultaneously satisfy all these requirements, and manually designed filtering rules are inefficient and prone to omission.
4. **Goal**: Design an automated agentic pipeline to generate high-quality synthetic data for balancing long-tail datasets.
5. **Key Insight**: An LLM serves as the prompt agent to generate detailed and specific prompts (incorporating object category, style, color, lighting, etc.), while a VLLM serves as the validation agent to filter substandard images; the system prompts of both agents are optimized via TextGrad.
6. **Core Idea**: Layer Diffusion natively outputs an alpha channel that provides precise masks (eliminating the need for additional segmentation models); LLM-optimized prompts ensure single-object content and high diversity; VLLM validation serves as a final safeguard to catch remaining invalid samples.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Open-Vocabulary Prompt Generation**—the LLM agent, with its system prompt optimized via TextGrad, generates detailed LD prompts; (2) **Foreground Image Generation**—Layer Diffusion generates transparent single-object RGBA images guided by the optimized prompts; (3) **Image Filtering**—the VLLM validation agent inspects generated image quality and discards substandard samples. The validated foreground instances are then randomly pasted onto background images.

### Key Designs

1. **TextGrad-Optimized LD Prompt Agent**:
    - **Function**: Generate detailed prompts that guide Layer Diffusion to produce high-quality single-object images.
    - **Mechanism**: Three LLMs collaborate: the LD prompt agent $A_{p_{LD}}$ generates LD prompt $p_{LD}$ from system prompt $p_{\text{sys}}$; the prompt evaluator $E_{\text{prompt}}$ assesses the quality of the generated prompt and outputs a textual loss $L$; TextGrad's textual gradient descent optimizes $p_{\text{sys}}^*$. The prompt verifier $V_{\text{prompt}}$ compares prompt quality before and after optimization to decide whether to accept the update. The process iterates until the verifier accepts or the maximum iteration count is reached. The optimized prompt incorporates detailed attributes including object category, action, environment, style, color, texture, lighting, and viewpoint.
    - **Design Motivation**: Standard prompts (e.g., "a photo of a single \<object>") are too vague, leading to extraneous objects and low diversity. TextGrad automatically discovers what prompt descriptions enable Layer Diffusion to generate optimal single-object images.

2. **Layer Diffusion Foreground Generation**:
    - **Function**: Generate transparent foreground object images with precise alpha masks.
    - **Mechanism**: Layer Diffusion encodes the alpha transparency channel into the latent distribution of Stable Diffusion, directly outputting RGBA images. The alpha channel natively provides precise segmentation masks without requiring additional models such as SAM. After generation, median filtering is applied to the alpha channel to remove isolated noise pixels and smooth mask edges.
    - **Design Motivation**: Obtaining masks via cross-attention maps (MosaicFusion) or additional segmentation models (X-Paste) yields unstable quality and incurs additional runtime. The alpha channel provides a "free" precise mask.

3. **VLLM Data Validation Agent**:
    - **Function**: Automatically filter substandard synthetic images.
    - **Mechanism**: A VLLM (Meta-LLaMA-3.2-11B-Vision-Instruct) serves as the validation agent, with its system prompt likewise optimized via TextGrad. Validation criteria are encoded into the system prompt: (1) **Single object**—the image contains only one instance of the target category; (2) **Single viewpoint**—the object is depicted from a single angle; (3) **Completeness**—the object is fully visible; (4) **Clean background**—the background is blank and free of distractors. Images failing validation are discarded.
    - **Design Motivation**: Even after prompt optimization, approximately 7% of samples remain invalid; VLLM validation serves as the final line of defense to ensure data quality.

### Loss & Training
TextGrad optimization employs textual gradients (LLM-generated feedback text) in lieu of numerical gradients to optimize system prompts. The LLM used is Meta-LLaMA-3.1-8B-Instruct; the VLLM used is Meta-LLaMA-3.2-11B-Vision-Instruct.

## Key Experimental Results

### Main Results

**LVIS Instance Segmentation**:

| Method | $\text{mAP}_\text{mask}$ | $\text{mAP}_\text{mask}^\text{rare}$ | Invalid Data Rate |
|---|---|---|---|
| Mask R-CNN (baseline) | 21.7 | 9.6 | — |
| MosaicFusion | 23.1 | 15.2 | ~50% |
| Gen2Det | 23.6 | 15.3 | — |
| **Gen-n-Val** | **25.6** | **17.2 (+7.6)** | **~7%** |

**COCO Instance Segmentation (YOLO11m)**:

| Method | mAP | $\text{mAP}_\text{rare}$ |
|---|---|---|
| YOLO11m (baseline) | 10.3 | 6.5 |
| Copy-Paste | 10.4 | 6.7 |
| **Gen-n-Val** | **14.5** | **10.1 (+3.6)** |

### Ablation Study

| Configuration | Invalid Data Rate | Notes |
|---|---|---|
| Standard prompt + LD | ~44% | No prompt optimization |
| TextGrad-optimized prompt + LD | ~7% | Prompt agent effective |
| + VLLM validation | <1% | Validation agent further reduces invalid rate |
| MosaicFusion | ~50% | Baseline method |

### Key Findings
- **Invalid data reduced from 50% to 7%**: Prompt optimization is the dominant contributor, with VLLM validation providing an additional quality guarantee.
- **Most significant gains on rare categories (+7.6 mAP)**: Validates the substantial value of synthetic data in balancing long-tail distributions.
- **Scalability**: Injecting more synthetic data (from 1,874 to 727,393 instances) yields continuous improvement (+0.9 → +7.6 $\text{mAP}_\text{rare}$).

## Highlights & Insights
- **The alpha channel of Layer Diffusion as a "free mask"** is the most critical technical choice: it eliminates dependency on additional segmentation models and fundamentally guarantees perfect alignment between the mask and the object.
- **TextGrad-optimized dual-agent prompts** constitute an elegant automation strategy: both "what constitutes a good generation prompt" and "what constitutes a qualified image" are delegated to LLMs for automatic iterative optimization, removing the need for manually designed rules.
- The insight that **data quality > data quantity** merits attention: reducing the invalid data rate from 50% to 7% yields more substantial gains than simply increasing data volume.

## Limitations & Future Work
- Dependent on the generation quality of Layer Diffusion—results for certain rare categories (e.g., specific foods or tools) may be suboptimal.
- TextGrad optimization requires multiple LLM calls, incurring non-trivial computational overhead.
- Validation is limited to object detection and instance segmentation; extension to semantic segmentation, keypoint detection, and other tasks has not been explored.
- The Copy-Paste compositing approach may produce unnatural scene layouts.
- Future work could explore more realistic object placement in 3D scenes.

## Related Work & Insights
- **vs. MosaicFusion**: MosaicFusion generates masks via cross-attention, with a ~50% invalid rate. Gen-n-Val uses alpha channels and agentic validation, achieving only ~7% invalid rate with higher mask quality.
- **vs. X-Paste**: X-Paste uses an additional segmentation model to obtain masks, requiring 4.3× the GPU time of MosaicFusion. Gen-n-Val's alpha channel approach is more efficient.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of Layer Diffusion + dual agents + TextGrad is novel and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ LVIS + COCO benchmarks, multiple detectors, and scalability analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline is clearly presented; failure cases are illustrated intuitively.
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical and efficient solution for data augmentation in long-tail datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](../../ACL2026/llm_agent/supplement_generation_training_for_enhancing_agentic_task_performance.md)
- [\[AAAI 2026\] A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators](../../AAAI2026/llm_agent/a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)
- [\[ICCV 2025\] Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions](../../ICCV2025/llm_agent/embodied_image_captioning_self-supervised_learning_agents_for_spatially_coherent.md)
- [\[CVPR 2026\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_openvoc.md)
- [\[AAAI 2026\] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching](../../AAAI2026/llm_agent/pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)

</div>

<!-- RELATED:END -->
