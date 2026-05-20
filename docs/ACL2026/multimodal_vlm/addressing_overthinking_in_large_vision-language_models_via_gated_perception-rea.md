---
title: >-
  [Paper Note] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization
description: >-
  [ACL 2026][Multimodal VLM][overthinking] This paper proposes the GPRO framework, which addresses the overthinking problem in LVLMs by inserting a meta-reasoning controller that dynamically routes computation at each toke…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "overthinking"
  - "perception-reasoning disentanglement"
  - "meta-reasoning controller"
  - "adaptive computation"
  - "multi-objective reinforcement learning"
date: 2026-05-08
content_hash: 6a5f603285e3fb35
---

# Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization

**Conference**: ACL 2026
**arXiv**: [2601.04442](https://arxiv.org/abs/2601.04442)  
**Code**: None  
**Area**: Multimodal VLM / Adaptive Computation
**Keywords**: overthinking, perception-reasoning disentanglement, meta-reasoning controller, adaptive computation, multi-objective reinforcement learning

## TL;DR

This paper proposes the GPRO framework, which addresses the overthinking problem in LVLMs by inserting a meta-reasoning controller that dynamically routes computation at each token generation step to one of three paths (fast / perception re-examination / reasoning reflection), simultaneously improving both accuracy and efficiency.

## Background & Motivation

**Background**: Large vision-language models (LVLMs) have demonstrated strong reasoning capabilities through chain-of-thought mechanisms, but this "slow-thinking" paradigm frequently leads to overthinking — generating lengthy reasoning chains even for simple questions.

**Limitations of Prior Work**: (1) Overthinking wastes computational resources and can sometimes introduce errors. (2) Existing adaptive reasoning methods overlook a critical bottleneck — visual perception failure. Large-scale analysis reveals that perception failures account for more than twice the frequency of reasoning errors among LVLM mistakes.

**Key Challenge**: When an error originates from "misseeing" rather than "misreasoning," increasing reasoning depth is not only unhelpful but may introduce additional errors. Existing methods focus solely on reasoning adaptation, entirely neglecting perception adaptation.

**Goal**: Design an adaptive computation framework that jointly accounts for both perceptual and reasoning uncertainty.

**Key Insight**: Drawing on dual-process theory from cognitive science (Kahneman), humans flexibly switch among fast intuition, visual re-examination, and deep reasoning when solving problems.

**Core Idea**: Leverage large-scale failure attribution supervision (790K samples) to distinguish perception errors from reasoning errors, and train a meta-reasoning controller to perform three-way dynamic computation allocation.

## Method

### Overall Architecture

GPRO inserts GPR modules at alternating layers of the Transformer decoder, replacing standard FFN layers. Each GPR module consists of a meta-reasoning controller and three computation paths. At each token generation step, the controller determines which path to activate based on the model's internal state.

### Key Designs

1. **Meta-Reasoning Controller**:

    - **Function**: Makes path-selection decisions at each token generation step.
    - **Mechanism**: A lightweight 2-layer Transformer receives three signals — the current hidden state $h_t$ (semantic context), predictive entropy $U_t$ (uncertainty measure), and global image features $V_g$ (visual complexity) — and outputs a discrete action $a_t \in \{\text{fast}, \text{perception}, \text{reasoning}\}$.
    - **Design Motivation**: The three signals are complementary — the hidden state reflects "what is currently being processed," entropy reflects "how uncertain the model is," and image features reflect "how visually complex the input is."

2. **Three Computation Paths**:

    - **Function**: Provide specialized processing tailored to different types of computational demands.
    - **Mechanism**: The Fast Path applies the original FFN (low-cost direct generation); the Slow Perception Path re-examines visual features via cross-attention, $\text{Perc}(h_t, V) = \text{CrossAttn}(h_t, V, V)$; the Slow Reasoning Path performs internal self-reflection via a meta-Transformer, $\text{Reas}(h_t, H_{<t}) = \text{MetaTrans}(h_t, H_{<t})$.
    - **Design Motivation**: Perception errors require "looking at the image again," while reasoning errors require "rethinking," making a divide-and-conquer approach more efficient than unified processing.

3. **Large-Scale Failure Attribution Supervision**:

    - **Function**: Provides training signals that enable the controller to distinguish perception failures from reasoning failures.
    - **Mechanism**: Error cases are collected by running Qwen2.5-VL on approximately 790K samples; GPT-4 then attributes each error as either "visual perception failure" or "reasoning error," constructing a labeled training set.
    - **Design Motivation**: Standard benchmarks only provide final answer correctness, lacking signals indicating "which cognitive stage failed."

### Loss & Training

Multi-objective PPO training with reward function $R(\tau) = R_{task} + \alpha_c R_{cost} + \alpha_l R_{cal}$. The Task Reward assigns +1 for correct answers; the Cost Reward penalizes slow-path activations; the Calibration Reward ensures uncertainty scores are aligned with actual errors (high uncertainty before errors, low before correct predictions).

## Key Experimental Results

### Main Results (Qwen2.5-VL-7B backbone)

| Method | MathVision Acc | MathVerse Acc | MathVista Acc | Avg. Response Length |
|--------|---------------|---------------|---------------|----------------------|
| Base Qwen2.5-VL-7B | 24.1 | 38.5 | 65.1 | ~350 |
| Mulberry | Above base | Above base | Above base | Longer |
| GPRO-7B | Significant gain | Significant gain | Significant gain | **Substantially shorter** |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|-----------|---------|
| Remove Perception Path | Notable accuracy drop | Perceptual re-examination is critical for error correction |
| Remove Reasoning Path | Slight accuracy drop | Reasoning self-reflection provides auxiliary benefit |
| Remove Calibration Reward | Path selection degrades | Uncertainty calibration is a key signal for the controller |
| Failure attribution analysis | Perception > Reasoning 2:1 | Validates the core claim that perception is the primary bottleneck |

### Key Findings
- GPRO simultaneously improves accuracy and efficiency (shorter responses) across 5 benchmarks, challenging the assumption that "more accurate = longer."
- Visual perception failure is confirmed as the dominant source of LVLM errors (accounting for over two-thirds of failures), rather than insufficient reasoning.
- The three-way controller learns meaningful routing strategies — simple questions are directed to the Fast Path, visually ambiguous inputs to the Perception Path.

## Highlights & Insights
- "The root cause of overthinking may not be insufficient deliberation, but insufficient perception" — this insight reframes the direction of reasoning optimization in LVLMs.
- The large-scale failure attribution pipeline is generalizable — using a stronger model to annotate error types of a weaker model is a broadly applicable supervision generation strategy.
- The three-path computation architecture elegantly operationalizes dual-process theory from cognitive science.

## Limitations & Future Work
- GPT-4's failure attributions may themselves be biased, necessitating more reliable attribution methods.
- The meta-reasoning controller increases model complexity, requiring additional engineering effort at deployment.
- Validation is limited to 3B and 7B models; applicability to larger-scale models remains untested.
- Future work may explore finer-grained perception paths (e.g., region-level re-examination vs. full-image re-examination).

## Related Work & Insights
- **vs. Adaptive reasoning methods (FAST, etc.)**: GPRO is the first to incorporate perception adaptation, adjusting not only reasoning depth but also perceptual depth.
- **vs. Mixture-of-Experts**: MoE selects along the parameter dimension; GPRO selects along the computation-type dimension.
- **vs. Vision-R1 / LMM-R1**: These methods enhance reasoning via RL but do not distinguish between perception and reasoning errors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Perception-reasoning disentanglement for adaptive computation constitutes a genuinely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, ablation studies, and attribution analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is well-argued; architectural description is clear.
- Value: ⭐⭐⭐⭐⭐ Has paradigm-level implications for LVLM reasoning optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[CVPR 2026\] Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](../../CVPR2026/multimodal_vlm/overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
