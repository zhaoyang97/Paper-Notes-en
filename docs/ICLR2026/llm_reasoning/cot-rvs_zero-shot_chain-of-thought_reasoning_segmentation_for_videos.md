---
title: >-
  [Paper Note] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos
description: >-
  [ICLR 2026][LLM Reasoning][Reasoning Video Segmentation] This paper proposes CoT-RVS, a fully training-free multi-agent framework that leverages the zero-shot CoT reasoning capabilities of pretrained MLLMs for temporal-semantic correlation analysis and keyframe selection, achieving substantial improvements over fine-tuned methods on reasoning video segmentation tasks (Refer-DAVIS J&F 79.1 vs. 71.2; ReasonVOS J&F 65.5 vs. 49.9).
tags:
  - ICLR 2026
  - LLM Reasoning
  - Reasoning Video Segmentation
  - Chain-of-Thought
  - Zero-Shot
  - Keyframe Selection
  - Multimodal Large Language Models
date: 2026-05-08
content_hash: 4bc7555a0eaa3cc1
---

# CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos

**Conference**: ICLR 2026
**arXiv**: [2505.18561](https://arxiv.org/abs/2505.18561)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Reasoning Video Segmentation, Chain-of-Thought, Zero-Shot, Keyframe Selection, Multimodal Large Language Models

## TL;DR

This paper proposes CoT-RVS, a fully training-free multi-agent framework that leverages the zero-shot CoT reasoning capabilities of pretrained MLLMs for temporal-semantic correlation analysis and keyframe selection, achieving substantial improvements over fine-tuned methods on reasoning video segmentation tasks (Refer-DAVIS J&F 79.1 vs. 71.2; ReasonVOS J&F 65.5 vs. 49.9).

## Background & Motivation

- **State of the Field**: Reasoning Video Object Segmentation (Reasoning VOS) requires models to generate target mask sequences based on complex, implicit text queries (e.g., "which player made the three-point shot"), making it one of the most challenging tasks in video understanding.
- **Limitations of Prior Work**: Existing methods (VISA/VideoLISA/HyperSeg) fine-tune MLLMs to generate segmentation tokens but perform poorly on temporally sensitive queries. The fundamental limitation is their lack of inter-frame temporal reasoning—these methods focus on intra-frame semantic understanding but cannot effectively reason about "what happens at which point in time."
- **Root Cause**: CoT-based reasoning segmentation in the image domain (Seg-Zero/ThinkFirst) has achieved notable success, yet the video domain additionally requires temporal "thinking" capabilities. Directly extending image-domain approaches to video is infeasible, as target objects in videos may undergo occlusion, motion, appearance, or disappearance over time.
- **Starting Point**: Rather than performing any fine-tuning, CoT-RVS exploits the zero-shot CoT capabilities of pretrained MLLMs (e.g., GPT-4o, Gemma3) by designing task-specific prompts that guide temporal-semantic reasoning—a direction well-aligned with the trend of test-time compute.
- **Core Idea**: The MLLM analyzes keyframe candidates through a self-questioning CoT process, establishing correspondences along two dimensions—semantic (which objects in the frame match the query) and temporal (in which frame the target is most observable)—ultimately selecting the optimal keyframe for each instance.

## Method

### Overall Architecture

A three-module collaborative multi-agent framework that decomposes reasoning video segmentation into three sub-tasks: keyframe selection → frame-level segmentation → video tracking:
- MLLM Keyframe Selector $\mathcal{F}_{key}$: responsible for temporal-semantic correlation reasoning
- Reasoning Image Segmentation Model $\mathcal{F}_{seg}$: generates key masks on selected keyframes
- Video Processor $\mathcal{F}_{vid}$ (SAM2): propagates masks along the temporal axis

### Key Designs

**1. MLLM Keyframe Selector (Core Innovation)**:
- Uniformly samples $T' = \lfloor T/\xi \rfloor$ keyframe candidates
- Automatically synthesizes a CoT question-answering sequence for each candidate frame in a coarse-to-fine manner: general semantics ("what is in the frame") → temporal reasoning ("whether this is a better keyframe") → detail confirmation ("whether a new target object appears")
- Final output includes: a list of target instances, corresponding keyframe indices, and intra-frame target descriptions (e.g., "the player in black jersey who is shooting")
- Designed as a Reasoning VIS framework (k ≥ 1 instances), with Reasoning VOS as the special case of k = 1
- Compatible with both closed-source (GPT-4o) and open-source (Gemma3-12B/LLaVA1.5-7B) MLLMs

**2. Reasoning Image Segmentation**: Models such as Seg-Zero are applied to generate key masks on the selected keyframe based on the textual description.

**3. Video Processor**: SAM2 propagates key masks to all frames; a greedy post-processing step ensures non-overlapping masks across multiple instances: $m_{i,t} = \bigcap_{j=1}^{i-1} \neg m_{j,t} \cap \hat{m}_{i,t}$

**4. Online Inference Extension (Online CoT-RVS)**:
- Periodically invokes the MLLM every $\xi$ frames to determine whether the current frame should replace the existing keyframe
- Greedy update strategy: if the new frame is better, the target and mask are updated; otherwise, historical information is retained
- Represents the first streaming reasoning video segmentation approach, suitable for real-time video stream scenarios

### Loss & Training

Entirely training-free—all modules use pretrained weights with no fine-tuning of any kind.

## Key Experimental Results

### Main Results

| Dataset | Metric | CoT-RVS (GPT-4o) | GLUS | SAMWISE | VideoLISA (Po) | VISA-13B |
|--------|------|:----------------:|:----:|:-------:|:-------------:|:--------:|
| MeViS | J&F | **52.2** | 51.3 | 49.5 | 44.4 | 44.5 |
| Refer-DAVIS | J&F | **79.1** | — | 70.6 | 68.8 | 70.4 |
| ReasonVOS | J&F | **65.5** | 49.9 | — | 47.5 | — |

### Ablation Study

| Configuration | MeViS J&F | Refer-DAVIS J&F | Notes |
|------|:---------:|:---------------:|------|
| CoT-RVS-GPT-4o | 52.2 | 79.1 | Best closed-source configuration |
| CoT-RVS-Gemma3-12B | 44.2 | 74.6 | Best open-source configuration |
| CoT-RVS-LLaVA1.5-7B | 45.9 | 73.9 | Lightest open-source configuration |
| w/o CoT (direct prompt) | — | ~65 | CoT reasoning contributes ~14-point gain |
| Online CoT-RVS (GPT-4o) | — | 77.8 | Online version approaches offline performance |

### Key Findings

- CoT-RVS outperforms GLUS by +15.6 points on ReasonVOS, with a particularly pronounced advantage on temporally sensitive queries (e.g., three-point shots, specific action moments), validating the central value of temporal reasoning.
- The open-source Gemma3 variant still surpasses fine-tuned methods such as VISA and VideoLISA without any API cost, suggesting that the general reasoning capabilities of pretrained MLLMs have been underestimated.
- The online version (Online CoT-RVS) lags behind the offline version by only ~1.3 points while supporting streaming processing, offering significant practical utility.

## Highlights & Insights

- **Training-Free Breakthrough**: CoT-RVS is the first zero-shot reasoning VOS framework compatible with both closed-source and open-source MLLMs, challenging the prevailing paradigm that reasoning segmentation requires fine-tuning.
- **Value of Temporal Reasoning**: The CoT process enables MLLMs to genuinely "reason" about inter-frame temporal-semantic relationships—a capability that is fundamentally absent in fine-tuning-based segmentation token approaches.
- **Flexibility of Modular Design**: The segmentation model (LISA/Seg-Zero) and video processor (SAM2/Cutie) are interchangeable, allowing future improvements in individual modules to directly benefit the overall system.
- **Practicality of Online Extension**: Online reasoning VOS solutions are rare; this extension is meaningful for real-time scenarios such as surveillance and autonomous driving.

## Limitations & Future Work

- The GPT-4o variant incurs high inference costs (multiple API calls per video), making large-scale deployment impractical.
- The open-source Gemma3 variant trails the closed-source GPT-4o by 8 points on MeViS, indicating that the visual reasoning capability of MLLMs remains a bottleneck.
- Uniform frame sampling may miss critical motion frames; adaptive sampling strategies (e.g., motion-detection-guided sampling) could be more effective.
- The greedy post-processing for multi-instance scenarios is relatively simple and cannot handle severe occlusion cases.
- The possibility of end-to-end joint training of the CoT reasoning module with segmentation and tracking modules has not been explored.

## Related Work & Insights

- Compared to fine-tuning-based methods such as VISA and VideoLISA, CoT-RVS replaces fine-tuning with zero-shot reasoning, representing a fundamentally different technical paradigm.
- CoT-RVS shares lineage with Seg-Zero/ThinkFirst (image-domain CoT segmentation) but additionally incorporates temporal-dimension reasoning.
- The work demonstrates the promising applicability of the test-time compute trend to visual tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Zero-shot CoT applied to reasoning video segmentation is a novel concept)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 datasets + multiple MLLMs + online extension + ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Framework description is clear with complete formulations)
- Value: ⭐⭐⭐⭐ (Demonstrates the feasibility of training-free CoT for video segmentation)
- **vs. VISA/VideoLISA**: These methods fine-tune MLLMs to generate segmentation tokens; CoT-RVS is entirely training-free.
- **vs. Seg-Zero/ThinkFirst**: CoT reasoning image segmentation methods; this work is the first to extend them to the video temporal domain.
- **vs. SAM2**: Serves as a plug-and-play video tracking module, demonstrating its potential for integration with reasoning systems.

## Rating
- Novelty: ⭐⭐⭐⭐ Zero-shot CoT for temporal reasoning in video is pioneering
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 benchmarks + online extension + module substitution ablation
- Writing Quality: ⭐⭐⭐⭐ Illustrative examples and clear framework description
- Value: ⭐⭐⭐⭐ The training-free paradigm is practically meaningful, though it relies on strong MLLMs

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization](../../CVPR2026/llm_reasoning/graze_grounded_refinement_and_motion-aware_zero-shot_generation.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)
- [\[ICLR 2026\] Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)
- [\[ICLR 2026\] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)

<!-- RELATED:END -->
