---
title: >-
  [Paper Note] E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition
description: >-
  [ACL 2026][Object Detection][Multimodal Named Entity Recognition] This paper proposes E2E-GMNER, the first end-to-end GMNER framework that unifies entity recognition, semantic classification, visual grounding, and implicit knowledge reasoning within a single multimodal large language model. The framework employs CoT reasoning to adaptively assess the utility of visual and knowledge cues, and introduces Gaussian Risk-aware Bounding box Perturbation (GRBP) to enhance the robustness of generative bounding box prediction.
tags:
  - ACL 2026
  - Object Detection
  - Multimodal Named Entity Recognition
  - End-to-End Generation
  - Visual Grounding
  - Gaussian Perturbation
  - CoT Reasoning
date: 2026-05-08
content_hash: 343c241c6d9c7aea
---

# E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition

**Conference**: ACL 2026
**arXiv**: [2604.17319](https://arxiv.org/abs/2604.17319)
**Code**: [https://github.com/Finch-coder/E2E-GMNER](https://github.com/Finch-coder/E2E-GMNER)
**Area**: Object Detection
**Keywords**: Multimodal Named Entity Recognition, End-to-End Generation, Visual Grounding, Gaussian Perturbation, CoT Reasoning

## TL;DR

This paper proposes E2E-GMNER, the first end-to-end GMNER framework that unifies entity recognition, semantic classification, visual grounding, and implicit knowledge reasoning within a single multimodal large language model. The framework employs CoT reasoning to adaptively assess the utility of visual and knowledge cues, and introduces Gaussian Risk-aware Bounding box Perturbation (GRBP) to enhance the robustness of generative bounding box prediction.

## Background & Motivation

**Background**: Grounded Multimodal Named Entity Recognition (GMNER) requires jointly identifying entities in text, predicting their semantic types, and localizing each entity to the corresponding visual region in an image. Existing methods such as H-Index, TIGER, and RiVEG predominantly adopt pipeline-based architectures.

**Limitations of Prior Work**: (1) Pipeline architectures decouple textual entity recognition and visual grounding into independent modules (e.g., standalone NER taggers and external object detectors), leading to error accumulation and precluding joint optimization. (2) Existing methods resolve text-visual ambiguity through implicit cross-modal alignment, lacking explicit mechanisms to determine when visual evidence or external knowledge is genuinely useful, such that noisy visual cues can degrade performance. (3) In generative bounding box prediction, single hard-target supervision is sensitive to annotation noise and coordinate discretization errors.

**Key Challenge**: The tension between end-to-end unification and the task-specific requirements of each subtask — how can a single model simultaneously optimize entity recognition, semantic classification, and visual grounding, which are fundamentally different tasks?

**Goal**: Design the first end-to-end GMNER framework to eliminate error accumulation inherent in pipeline systems.

**Key Insight**: GMNER is formulated as a conditional generation task via instruction tuning, leveraging the unified generative capacity of multimodal large language models.

**Core Idea**: End-to-end generation, CoT adaptive reasoning, and Gaussian soft supervision work in concert to address the three core challenges of GMNER.

## Method

### Overall Architecture

Given an image-text pair and a task instruction, a LoRA-adapted multimodal LLM first performs CoT reasoning (visual cue analysis followed by background knowledge analysis), then autoregressively generates structured entity records in the format (entity name | semantic type | bounding box coordinates). During training, GRBP replaces hard bounding box supervision.

### Key Designs

1. **End-to-End Generative GMNER**:

    - **Function**: Eliminate error accumulation from pipeline architectures.
    - **Mechanism**: GMNER is formulated as conditional generation: input = [instruction; (image, text)], output = [reasoning sequence $R$; entity record set $\{(e_i, c_i, b_i)\}$]. Each entity record is serialized as "entity name | type | [x1, y1, x2, y2]", and all records are concatenated as the final prediction. Training uses standard autoregressive MLE loss.
    - **Design Motivation**: A single generative process enables information flow between entity recognition and visual grounding, enabling true joint optimization.

2. **Adaptive Reasoning via CoT Instruction Tuning**:

    - **Function**: Enable the model to autonomously determine when visual evidence or background knowledge is beneficial.
    - **Mechanism**: Prior to generating entity records, the model outputs a reasoning sequence $R$ comprising visual cue analysis (whether the image contains visual evidence corresponding to textual entities) and background knowledge analysis (whether external knowledge is needed for disambiguation). During training, reasoning sequences are generated by a stronger external LLM via API and used as supervision; during inference, the model generates reasoning entirely autonomously without relying on external models.
    - **Design Motivation**: Prevent the model from blindly exploiting noisy visual cues or irrelevant knowledge — the model is encouraged to "reason before acting."

3. **Gaussian Risk-aware Bounding Box Perturbation (GRBP)**:

    - **Function**: Improve robustness of generative bounding box prediction against annotation noise and discretization errors.
    - **Mechanism**: During training, GT bounding boxes are stochastically perturbed: Gaussian noise is added to center coordinates ($\delta_x, \delta_y \sim \mathcal{N}(0, \beta^2)$), and width/height are scaled by Gaussian factors. An IoU guard ensures that the perturbed box satisfies IoU $\geq \tau$ with respect to the original box. This replaces hard-target supervision with Gaussian-weighted soft targets — larger perturbations correspond to lower probability — maintaining empirical risk minimization while tolerating small geometric deviations.
    - **Design Motivation**: Generative bounding box prediction discretizes coordinates into token sequences, where minor deviations produce disproportionately large training losses. GRBP mitigates this through soft supervision.

### Loss & Training

Standard autoregressive MLE loss: $\mathcal{L} = -\sum_t \log p_\theta(y_t | y_{<t}, \text{Instruction}, I, T)$, where bounding box coordinates participate in training as soft targets after GRBP perturbation.

## Key Experimental Results

### Main Results

On the Twitter-GMNER and Twitter-FMNERG benchmarks:

| Method | Twitter-GMNER (GMNER) | Twitter-GMNER (MNER) |
|---|---|---|
| GMDA (pipeline) | 58.61 | — |
| GEM (pipeline + MLLM) | 59.83 | 83.15 |
| **E2E-GMNER** | **Most competitive** | **Most competitive** |

### Ablation Study

| Configuration | Effect | Note |
|---|---|---|
| w/o CoT reasoning | Degraded | Adaptive visual/knowledge utilization is important |
| w/o GRBP | Degraded | Robustness of bounding box prediction is impaired |
| Hard supervision vs. GRBP soft supervision | GRBP superior | Tolerates annotation noise |
| End-to-end vs. pipeline | End-to-end superior | Eliminates error accumulation |

### Key Findings

- The end-to-end framework achieves highly competitive performance on the primary GMNER task, validating the effectiveness of unified optimization.
- CoT reasoning enables the model to actively disregard noisy visual cues rather than being misled by them, which is critical for improving entity localization precision.
- The IoU guard in GRBP ensures perturbations remain bounded, balancing the flexibility of soft supervision with localization accuracy.
- Inference requires no external models, preserving efficient end-to-end inference.

## Highlights & Insights

- The significance of the first end-to-end GMNER framework lies not only in performance gains, but also in demonstrating that entity recognition and visual grounding can effectively co-operate within a unified generative framework rather than requiring sequential processing.
- GRBP introduces the concept of data augmentation into supervision target design: rather than augmenting input data, it "augments" labels — generating soft supervision signals through probabilistic perturbation of GT bounding boxes. This idea is transferable to other generative grounding tasks.
- CoT reasoning serves as an "attention gating" mechanism: by having the model assess the reliability of visual/knowledge signals before utilizing them, it constitutes a more intelligent multimodal fusion strategy than simple cross-attention.

## Limitations & Future Work

- Performance on certain entity categories may still fall short of specialized pipeline methods, particularly those employing powerful external detectors.
- CoT reasoning training relies on external LLMs (e.g., GPT-4o) to generate reasoning sequences, introducing additional data preparation costs.
- GRBP hyperparameters ($\beta, \gamma, \tau$) require tuning and may need different settings across datasets.
- Evaluation is currently limited to Twitter image-text pair datasets; generalization to other domains (news, e-commerce) remains unexplored.

## Related Work & Insights

- **vs. RiVEG (Li et al., 2024)**: Uses MLLMs as an auxiliary component but retains a pipeline architecture; E2E-GMNER achieves true end-to-end processing.
- **vs. MAKAR (Lin et al., 2025)**: Employs a multi-agent MLLM system to address semantic ambiguity but still contains pipeline components; E2E-GMNER is more streamlined.
- **vs. MQSPN (Tang et al., 2025)**: Uses set prediction to mitigate exposure bias but does not address noise sensitivity in bounding box prediction; E2E-GMNER's GRBP directly tackles this challenge.

## Rating
- Novelty: ⭐⭐⭐⭐ First end-to-end GMNER framework with innovative GRBP soft supervision
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks with comprehensive ablation study
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; method description is detailed
- Value: ⭐⭐⭐⭐ Provides an effective demonstration of the end-to-end paradigm for multimodal NER

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](../../ICLR2026/object_detection/traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](../../CVPR2026/object_detection/evaluating_fewshot_pill_recognition_under_visual_d.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[CVPR 2026\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](../../CVPR2026/object_detection/mitigating_memorization_in_texttoimage_diffusion_v.md)
- [\[ICCV 2025\] LMM-Det: Make Large Multimodal Models Excel in Object Detection](../../ICCV2025/object_detection/lmm-det_make_large_multimodal_models_excel_in_object_detection.md)

</div>

<!-- RELATED:END -->
