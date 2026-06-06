---
title: >-
  [Paper Note] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization
description: >-
  [ACL 2026][Multimodal VLM][Overthinking] The GPRO framework is proposed to address the overthinking problem in LVLMs. By using a meta-reasoning controller to dynamically route computation into three paths (fast…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Overthinking"
  - "perception-reasoning separation"
  - "meta-reasoning controller"
  - "adaptive computation"
  - "multi-objective reinforcement learning"
date: 2026-05-08
content_hash: fc3496d0ce9db8c4
---

# Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization

**Conference**: ACL 2026  
**arXiv**: [2601.04442](https://arxiv.org/abs/2601.04442)  
**Code**: None  
**Area**: Multimodal VLM / Adaptive Computation  
**Keywords**: Overthinking, perception-reasoning separation, meta-reasoning controller, adaptive computation, multi-objective reinforcement learning

## TL;DR

The GPRO framework is proposed to address the overthinking problem in LVLMs. By using a meta-reasoning controller to dynamically route computation into three paths (fast, perception re-check, and reasoning reflection) at each token generation step, it simultaneously improves both accuracy and efficiency.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) demonstrate strong reasoning capabilities through chain-of-thought mechanisms. However, this "slow thinking" approach often leads to overthinking, where lengthy reasoning chains are generated even for simple questions.

**Limitations of Prior Work**: (1) Overthinking not only wastes computational resources but also occasionally introduces errors. (2) Existing adaptive reasoning methods overlook a critical bottleneck: visual perception failure. Large-scale analysis indicates that the frequency of perception failure in LVLM errors is more than twice that of reasoning errors.

**Key Challenge**: When an error stems from "misseeing" rather than "misthinking", increasing reasoning depth is not only useless but may introduce further errors. Existing methods focus solely on reasoning adaptation and completely ignore perception adaptation.

**Goal**: To design an adaptive computation framework that simultaneously accounts for both perception uncertainty and reasoning uncertainty.

**Key Insight**: Drawing from the dual-process theory in cognitive science (Kahneman), humans flexibly switch between fast intuition, visual re-checking, and deep reasoning when solving problems.

**Core Idea**: Distinguish between perception errors and reasoning errors through large-scale failure attribution supervision (790k samples) to train a meta-reasoning controller for three-way dynamic computation allocation.

## Method

### Overall Architecture

GPRO inserts GPR modules into alternating layers of the Transformer decoder, replacing standard FFN layers. Each GPR module contains a meta-reasoning controller and three computation paths. The controller decides which path to activate at each token generation step based on internal states.

### Key Designs

1. **Meta-reasoning Controller**:
    - **Function**: Making path selection decisions at each token generation step.
    - **Mechanism**: A 2-layer lightweight Transformer receives three signals: the current hidden state $h_t$ (semantic context), prediction entropy $U_t$ (uncertainty measure), and global image features $V_g$ (visual complexity). It outputs a discrete action $a_t \in \{\text{fast}, \text{perception}, \text{reasoning}\}$.
    - **Design Motivation**: The three signals are complementary: hidden states reflect "what is currently being thought," entropy reflects "how uncertain the model is," and image features reflect "how complex the visual input is."

2. **Three Computation Paths**:
    - **Function**: Providing specialized processing for different types of computational needs.
    - **Mechanism**: The Fast Path uses the original FFN (low-cost direct generation). The Slow Perception Path revisits visual features via cross-attention: $$\text{Perc}(h_t, V) = \text{CrossAttn}(h_t, V, V)$$. The Slow Reasoning Path performs internal self-reflection via a meta-Transformer: $$\text{Reas}(h_t, H_{<t}) = \text{MetaTrans}(h_t, H_{<t})$$.
    - **Design Motivation**: Perception errors require "re-looking at the image," while reasoning errors require "re-thinking." Divide-and-conquer is more efficient than uniform processing.

3. **Large-scale Failure Attribution Supervision**:
    - **Function**: Providing training signals for the controller to distinguish between perception and reasoning failures.
    - **Mechanism**: Qwen2.5-VL was run on approximately 790k samples to collect error cases. GPT-4 was then used to attribute each error as either a "visual perception failure" or a "reasoning error" to construct a labeled training set.
    - **Design Motivation**: Standard benchmarks only provide the correctness of the final answer, lacking signals regarding "which cognitive stage failed."

### Loss & Training

Multi-objective PPO training is utilized with a reward function $R(\tau) = R_{task} + \alpha_c R_{cost} + \alpha_l R_{cal}$. The Task Reward is +1 for correctness; the Cost Reward penalizes the activation of slow paths; the Calibration Reward ensures that uncertainty scores align with actual errors (high before errors, low before correct outputs).

## Key Experimental Results

### Main Results (Qwen2.5-VL-7B Base)

| Method | MathVision Acc | MathVerse Acc | MathVista Acc | Avg. Response Length |
|------|---------------|---------------|---------------|------------|
| Base Qwen2.5-VL-7B | 24.1 | 38.5 | 65.1 | ~350 |
| Mulberry | Gain over base | Gain over base | Gain over base | Longer |
| GPRO-7B | Significant Gain | Significant Gain | Significant Gain | **Greatly Shortened** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Remove Perception Path | Significant Acc Drop | Perception re-check is crucial for error correction. |
| Remove Reasoning Path | Slight Acc Drop | Reasoning self-reflection provides auxiliary support. |
| Remove Calibration Reward | Degraded Path Selection | Uncertainty calibration is a key signal for the controller. |
| Error Attribution Analysis | Perc > Reas (2:1) | Validates the core argument that "perception is the primary bottleneck." |

### Key Findings
- GPRO simultaneously improves accuracy and efficiency (shorter responses) across 5 benchmarks, breaking the "more accurate = longer" assumption.
- Visual perception failures are indeed the primary source of errors for LVLMs (accounting for over 2/3), rather than insufficient reasoning.
- The three-way controller learns meaningful routing strategies: simple questions take the Fast Path, while visual ambiguity triggers the Perception Path.

## Highlights & Insights
- "The root of overthinking may not be thinking too little, but seeing unclearly"—this insight shifts the perspective on LVLM reasoning optimization.
- The construction method for large-scale failure attribution data is reusable—using a strong model to label error types of a weaker model is a general strategy for supervised generation.
- The three-path computational architecture elegantly engineers the dual-process theory from cognitive science.

## Limitations & Future Work
- GPT-4's failure attribution may contain its own biases; more reliable attribution methods are needed.
- The meta-reasoning controller increases model complexity, requiring additional engineering for deployment.
- While validated on 3B and 7B models, the applicability to larger-scale models has not been tested.
- Future work could explore finer-grained perception paths (e.g., region-level re-check vs. full-image re-check).

## Related Work & Insights
- **vs. Adaptive Reasoning Methods (e.g., FAST)**: Ours incorporates perception adaptation for the first time, adjusting both reasoning and perception depth.
- **vs. Mixture-of-Experts**: MoE selects in the parameter dimension, while GPRO selects in the computation type dimension.
- **vs. Vision-R1/LMM-R1**: These methods enhance reasoning through RL but do not distinguish between perception and reasoning errors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Adaptive computation with perception-reasoning separation is a new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 benchmarks, ablations, and attribution analysis.
- Writing Quality: ⭐⭐⭐⭐ Strong motivation and clear architectural description.
- Value: ⭐⭐⭐⭐⭐ Paradigmatic impact on LVLM reasoning optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[CVPR 2026\] Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](../../CVPR2026/multimodal_vlm/overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)

</div>

<!-- RELATED:END -->
