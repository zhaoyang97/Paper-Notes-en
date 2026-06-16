---
title: >-
  [Paper Note] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models
description: >-
  [ACL 2026][Hallucination Detection][Diffusion Language Model] This study presents the first systematic comparison of hallucination patterns between diffusion Large Language Models (dLLMs) and their autoregressive (AR) counterparts. It reveals that current dLLMs exhibit a higher propensity for hallucination and identifies three diffusion-specific failure modes: Premature Terminati
tags:
  - ACL 2026
  - Hallucination Detection
  - Diffusion Language Model
date: 2026-05-08
content_hash: 9de82504540ebb46
---
# Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.10556](https://arxiv.org/abs/2604.10556)  
**Code**: [github.com/ZeroLoss-Lab/Lost-in-Diffusion](https://github.com/ZeroLoss-Lab/Lost-in-Diffusion)  
**Area**: Hallucination Detection  
**Keywords**: Diffusion Large Language Models, Hallucination, Non-autoregressive generation, Failure modes, Scaling Inference-time Compute

## TL;DR

This study presents the first systematic comparison of hallucination patterns between diffusion Large Language Models (dLLMs) and their autoregressive (AR) counterparts. It reveals that current dLLMs exhibit a higher propensity for hallucination and identifies three diffusion-specific failure modes: Premature Termination, Incomplete Denoising, and Contextual Intrusion.

## Background & Motivation

**Background**: Diffusion Large Language Models (dLLMs) are rapidly emerging as a non-autoregressive generation paradigm. Open-source models such as LLaDA, Dream, and SDAR have achieved performance levels comparable to AR-LLMs on general benchmarks. Theoretically, the global planning and bidirectional visibility of dLLMs could mitigate the "snowballing effect" and "reversal curse" found in AR models.

**Limitations of Prior Work**: (1) The **reliability** of dLLMs (particularly regarding hallucinations) remains largely unexplored, as existing research focuses primarily on architectural optimization and inference acceleration; (2) The stochastic nature of the diffusion process may exacerbate decoding randomness, which is a known root cause of hallucinations; (3) There is a lack of a fair comparative framework with controlled variables.

**Key Challenge**: While dLLM global context planning should theoretically reduce hallucinations through backtrack correction, the inherent noise in diffusion might intensify them—the empirical validity of either perspective remains unclear.

**Goal**: To answer a core question through strictly controlled comparative experiments: Does the diffusion mechanism mitigate or exacerbate hallucinations?

**Key Insight**: The study designs two sets of meticulously controlled comparisons—(I) architectural alignment (LLaDA-8B vs. LLaMA-3-8B) and (II) parameter alignment (Dream-7B vs. Qwen2.5-7B, where Dream is directly initialized from Qwen weights)—to isolate the impact of the generation mechanism.

**Core Idea**: Although dLLMs have narrowed the performance gap in general tasks, their unique hallucination mechanisms pose critical challenges to model reliability. Dynamic sequence editing capabilities are required to realize the full potential of non-autoregressive generation.

## Method

### Overall Architecture

A paired comparison framework is constructed to isolate the influence of the generation paradigm through two sets of control experiments. The HalluLens benchmark is used to evaluate extrinsic hallucinations across three dimensions: precise knowledge recall, long-text factual consistency, and knowledge boundary detection.

### Key Designs

1. **Pairwise Comparison Framework**: Group I features architectural alignment (LLaDA-8B vs. LLaMA-3-8B), sharing similar architectures and parameter scales with comparable general performance. Group II features parameter alignment (Dream-7B vs. Qwen2.5-7B), where Dream is initialized directly from Qwen weights, ensuring any difference in hallucination is primarily attributable to the diffusion generation process. Pre-trained (non-instruction-tuned) checkpoints are prioritized to isolate post-training noise. The design motivation is to eliminate confounding factors from training data and model capacity.

2. **Standard Diffusion Inference Settings**: The study adopts canonical diffusion settings, where the number of denoising steps $T$ is set equal to the sequence length $L$ ($T=L$) to maximize iterative refinement capabilities. The temperature is set to 0 to ensure reproducibility. LLaDA utilizes high-confidence decoding, while Dream employs minimum entropy decoding. The design motivation is to fully characterize the native generation behavior of dLLMs without using semi-autoregressive or block-level acceleration methods.

3. **Inference-time Compute Dynamic Analysis**: The impact of different denoising steps $T \in \{128, 256, 512, 1024\}$ is evaluated on the LongWiki task, revealing distinct behaviors in the two dLLMs. LLaDA exhibits **early saturation** due to quasi-autoregressive decoding (linear noise scheduler + high-confidence decoding forcing an approximate left-to-right generation). In contrast, Dream achieves true non-sequential refinement through minimum entropy decoding, demonstrating **positive scaling** characteristics. The design motivation is to verify whether the theoretical "compute-for-quality" capability of dLLMs holds in practice.

### Loss & Training

This is an analytical work and does not involve model training. Evaluation is conducted using the automated LLM evaluator from HalluLens, with reliability verified through manual annotation on hierarchical subsets. Intrinsic hallucination tasks (e.g., summarization) are excluded as they rely heavily on instruction-following capabilities, which would introduce confounding factors.

## Key Experimental Results

### Main Results

| Model | PreciseWikiQA HR ↓ | PreciseWikiQA CR ↑ | LongWiki F1@32 ↑ | NonExistRefusal FA ↓ |
|------|-------------------|-------------------|------------------|---------------------|
| LLaMA-3-8B (AR) | 85.94 | 10.30 | 0.306 | 73.35 |
| LLaDA-8B (Ours) | **95.13** | **3.92** | 0.272 | **87.10** |
| Qwen2.5-7B (AR) | 89.06 | 9.06 | **0.387** | 94.05 |
| Dream-7B (Ours) | 92.54 | 6.04 | 0.340 | **98.50** |

### Frequency of Diffusion-Specific Failure Modes (Manual Annotation of 200 cases)

| Model | Premature Termination (PT) | Incomplete Denoising (ID) | Contextual Intrusion (CI) |
|------|-------------|----------------|----------------|
| LLaDA-8B | 18.0% | **60.0%** | 38.0% |
| Dream-7B | 13.0% | 44.0% | **58.0%** |

### Key Findings

- **dLLMs consistently underperform AR counterparts across all three tasks**: In precise knowledge recall, LLaDA-8B's accuracy is only 3.92% (vs. 10.30% for LLaMA-3-8B). In non-existent entity refusal, Dream-7B's false acceptance rate reaches 98.50%.
- **Divergence in inference-time compute dynamics**: LLaDA's F1@32 stagnates at ~0.27 across all steps (early saturation), while Dream shows monotonic increases from 128 to 1024 steps, demonstrating positive scaling.
- Early saturation is attributed to LLaDA's quasi-autoregressive generation order—despite theoretical bidirectional visibility, it is forced to approximate left-to-right generation.
- **Discovery of three diffusion-specific failure modes** is highly insightful:
    - **Premature Termination**: Fragments denoised independently fail to align grammatically, forcing the model to insert EOS or broken delimiters.
    - **Incomplete Denoising**: When facing rare entities, the latter part of the sequence anchors on meaningless tokens; bidirectional attention attempts to rationalize these connections, leading to total collapse.
    - **Contextual Intrusion**: High-frequency tokens (numbers, code keywords) are occasionally denoised early; bidirectional attention is then forced to construct a logical path toward these "pseudo-anchors," hijacking the original query.

## Highlights & Insights

- **First systematic quantification of dLLM hallucination issues**, filling a significant research gap—the dLLM community previously focused almost exclusively on performance parity while neglecting reliability.
- Rigorous design of the paired comparison framework, particularly the Dream-Qwen pair (sharing initialization weights), which provides an ideal controlled variable.
- **Taxonomy of three failure modes** provides valuable vocabulary and a framework for understanding dLLM generation behavior.
- Divergent dynamics in inference-time compute reveal that decoding strategies, rather than architecture alone, critically influence dLLM behavior.

## Limitations & Future Work

- Inability to completely isolate the influence of the generation paradigm—even lightweight diffusion adaptation requires weight updates, which may introduce subtle confounding factors.
- Use of canonical diffusion settings ($T=L$); acceleration methods used in practical deployments may alter hallucination characteristics.
- Exclusion of instruction-tuned models and intrinsic hallucination tasks limits evaluation coverage.
- Evaluation is restricted to 7-8B scale models; hallucination behavior in larger-scale dLLMs may differ.
- Future Direction: Dynamic sequence editing capabilities (insertion, deletion, re-masking) are key to dLLMs achieving their full potential.

## Related Work & Insights

- **Snowballing Effect & Reversal Curse**: Known hallucination mechanisms in AR-LLMs are not entirely eliminated in dLLMs but are instead transformed into new forms, such as "solidified" early denoising errors.
- **Re-masking methods** (e.g., LLaDA 2.1) are attempting to address token-level errors, directly relating to the failure modes identified in this study.
- **HalluLens Benchmark**: Provides a standardized tool for dLLM hallucination evaluation but requires expansion to cover more task types.
- Insight: For dLLMs to achieve reliable generation, the core challenge is not denoising precision but **global consistency maintenance**—the lack of an editing mechanism is a fatal weakness when independently denoised fragments conflict.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic comparison of dLLM hallucinations; the discovery of three failure modes is highly original and insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The paired comparison framework is rigorously designed with manual annotation validation, though model and task coverage could be broader.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Precise problem definition, deep analysis, and clear, vivid descriptions of failure modes; the Limitations section is sincere and constructive.
- **Value**: ⭐⭐⭐⭐⭐ Sounds a reliability alarm for the dLLM community; the three failure modes provide clear directions for future improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/hallucination/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
