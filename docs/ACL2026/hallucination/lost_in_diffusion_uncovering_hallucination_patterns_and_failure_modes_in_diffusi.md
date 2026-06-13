---
title: >-
  [Paper Note] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models
description: >-
  [ACL 2026][Hallucination Detection][Diffusion Language Models] This work provides the first systematic comparison of hallucination patterns between diffusion large language models (dLLMs) and their autoregressive (AR) co…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "Diffusion Language Models"
  - "Hallucination"
  - "Non-autoregressive Generation"
  - "Failure Modes"
  - "Inference-time Compute"
date: 2026-05-08
content_hash: 313f68bb37bebacc
---

# Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.10556](https://arxiv.org/abs/2604.10556)  
**Code**: [github.com/ZeroLoss-Lab/Lost-in-Diffusion](https://github.com/ZeroLoss-Lab/Lost-in-Diffusion)  
**Area**: Hallucination Detection  
**Keywords**: Diffusion Language Models, Hallucination, Non-autoregressive Generation, Failure Modes, Inference-time Compute

## TL;DR

This work provides the first systematic comparison of hallucination patterns between diffusion large language models (dLLMs) and their autoregressive (AR) counterparts. It reveals that current dLLMs have a higher propensity for hallucinations and identifies three diffusion-specific failure modes: Premature Termination, Incomplete Denoising, and Contextual Invasion.

## Background & Motivation

**Background**: Diffusion Large Language Models (dLLMs) are rapidly emerging as a non-autoregressive generation paradigm. Open-source models such as LLaDA, Dream, and SDAR have achieved performance comparable to AR-LLMs on general benchmarks. Theoretically, the global planning and bidirectional visibility of dLLMs could mitigate the "snowball effect" and "reversal curse" found in AR models.

**Limitations of Prior Work**: (1) The **trustworthiness** of dLLMs (especially regarding hallucinations) remains almost unexplored, with existing research focusing primarily on architecture optimization and inference acceleration; (2) The stochastic nature of the diffusion process may exacerbate decoding randomness, which is a known root cause of hallucinations; (3) There is a lack of a fair comparison framework with controlled variables.

**Key Challenge**: While dLLMs' global context planning should theoretically reduce hallucinations (via backtracking and correction), the inherent noise in diffusion might intensify them—the validity of these competing theories lacks empirical evidence.

**Goal**: To answer the core question through strictly controlled comparative experiments: Does the diffusion mechanism alleviate or exacerbate hallucinations?

**Key Insight**: This study designs two sets of meticulously controlled paired comparisons: (I) Architecture alignment (LLaDA-8B vs. LLaMA-3-8B) and (II) Parameter alignment (Dream-7B vs. Qwen2.5-7B, where Dream is directly initialized from Qwen weights), maximizing the isolation of the generation mechanism's impact.

**Core Idea**: Although dLLMs have narrowed the performance gap in general tasks, their unique hallucination mechanisms pose critical challenges to model reliability. Dynamic sequence editing capabilities are required to realize the full potential of non-autoregressive generation.

## Method

### Overall Architecture

A paired comparison framework was constructed to isolate the impact of generation paradigms. The HalluLens benchmark was used to evaluate extrinsic hallucinations across three dimensions: precise knowledge recall, long-text factual consistency, and knowledge boundary detection.

### Key Designs

1.  **Paired Comparison Framework**: Group I focuses on architecture alignment (LLaDA-8B vs. LLaMA-3-8B), sharing similar architectures and parameter scales with comparable general performance. Group II focuses on parameter alignment (Dream-7B vs. Qwen2.5-7B), where Dream is initialized from Qwen weights. Any differences in hallucinations can primarily be attributed to the diffusion generation process. Pre-trained (non-instruction tuned) checkpoints are prioritized to isolate post-training noise. **Design Motivation**: To eliminate confounding factors related to training data and model capacity.

2.  **Standard Diffusion Inference Setting**: A canonical diffusion setup is adopted where the number of denoising steps $T$ is set equal to the sequence length $L$ ($T=L$), maximizing the model's iterative refinement capability. Temperature is set to 0 for reproducibility. LLaDA uses high-confidence decoding, while Dream uses minimum entropy decoding. **Design Motivation**: To fully characterize the native generation behavior of dLLMs without using semi-autoregressive or block-level acceleration methods.

3.  **Dynamics Analysis of Inference-time Compute**: Evaluation on the LongWiki task across different denoising steps $T \in \{128, 256, 512, 1024\}$ reveals distinct behaviors. LLaDA exhibits **early saturation** due to quasi-autoregressive decoding (linear noise scheduler + high-confidence decoding forcing an approximate left-to-right generation). Dream exhibits **positive scaling** characteristics, as minimum entropy decoding enables true non-sequential refinement. **Design Motivation**: To verify whether the theoretical "trading compute for quality" capability of dLLMs holds in practice.

### Loss & Training

This is an analytical study and does not involve model training. Evaluation utilizes the automatic LLM evaluator from HalluLens, with reliability verified through human annotation on hierarchical subsets. Intrinsic hallucination tasks (e.g., summarization) were excluded as they rely heavily on instruction-following capabilities, which would introduce confounding factors.

## Key Experimental Results

### Main Results

| Model | PreciseWikiQA HR ↓ | PreciseWikiQA CR ↑ | LongWiki F1@32 ↑ | NonExistRefusal FA ↓ |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA-3-8B (AR) | 85.94 | 10.30 | 0.306 | 73.35 |
| LLaDA-8B (dLLM) | **95.13** | **3.92** | 0.272 | **87.10** |
| Qwen2.5-7B (AR) | 89.06 | 9.06 | **0.387** | 94.05 |
| Dream-7B (dLLM) | 92.54 | 6.04 | 0.340 | **98.50** |

### Frequency of Diffusion-Specific Failure Modes (Human Annotation, 200 cases)

| Model | Premature Termination (PT) | Incomplete Denoising (ID) | Contextual Invasion (CI) |
| :--- | :--- | :--- | :--- |
| LLaDA-8B | 18.0% | **60.0%** | 38.0% |
| Dream-7B | 13.0% | 44.0% | **58.0%** |

### Key Findings

*   **dLLMs consistently underperform compared to AR counterparts across all three tasks**: In precise knowledge recall, LLaDA-8B's accuracy is only 3.92% (vs. 10.30% for LLaMA-3-8B). In non-existent entity refusal, Dream-7B's false acceptance rate reaches 98.50%.
*   **Divergent dynamics of inference-time compute**: LLaDA's F1@32 stagnates at ~0.27 across all steps (early saturation), while Dream increases monotonically from 128 to 1024 steps, demonstrating positive scaling.
*   Early saturation is attributed to LLaDA's quasi-autoregressive generation order—despite theoretical bidirectional visibility, it is practically forced to approximate a left-to-right generation.
*   The discovery of **three diffusion-specific failure modes** is highly insightful:
    *   **Premature Termination**: Fragments denoised independently fail to align syntactically, forcing the model to insert EOS or break delimiters.
    *   **Incomplete Denoising**: When facing rare entities, the rear sequence anchors on meaningless tokens; bidirectional attention's attempt to rationalize these connections leads to overall collapse.
    *   **Contextual Invasion**: Occasional denoising of high-frequency tokens (numbers, code keywords) forces bidirectional attention to construct a logical path to these pseudo-anchors, hijacking the original query.

## Highlights & Insights

*   **First systematic quantification of hallucination issues in dLLMs**, filling a significant research gap as the dLLM community previously focused almost exclusively on performance parity rather than reliability.
*   The paired comparison framework is rigorously designed, especially the Dream-Qwen pairing (sharing initialized weights), providing an nearly ideal control variable.
*   The **taxonomy of three failure modes** provides valuable vocabulary and a framework for understanding dLLM generation behavior.
*   The divergent dynamics of inference-time compute reveal that decoding strategies (rather than the architecture itself) critically influence dLLM behavior.

## Limitations & Future Work

*   The impact of the generation paradigm cannot be completely isolated—even the most lightweight diffusion adaptation requires weight updates, which may introduce subtle confounding factors.
*   Using canonical diffusion settings ($T=L$) means that acceleration methods used in actual deployment might alter hallucination characteristics.
*   Instruction-tuned models and intrinsic hallucination tasks were excluded, limiting the evaluation coverage.
*   Only 7-8B scale models were evaluated; the hallucination behavior of larger dLLMs might differ.
*   Future Work: Dynamic sequence editing capabilities (insertion, deletion, re-masking) are key for dLLMs to reach their full potential.

## Related Work & Insights

*   **Snowball Effect & Reversal Curse**: Known hallucination mechanisms in AR-LLMs are not completely eliminated in dLLMs; instead, they transform into new forms where early denoising errors are "solidified."
*   **Re-masking Methods** (e.g., LLaDA 2.1) are attempting to address token-level errors, which directly correlates with the failure modes discovered in this work.
*   **HalluLens Benchmark**: Provides a standardized tool for dLLM hallucination evaluation but needs expansion to cover more task types.
*   Insight: For dLLMs to achieve reliable generation, the core challenge is not denoising precision but **maintenance of global consistency**—the lack of an editing mechanism is a fatal weakness when independently denoised fragments conflict.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ First systematic comparison of dLLM hallucinations; the discovery of three failure modes is highly original and insightful.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ The paired comparison framework is rigorous and includes human annotation verification, though model and task coverage could be broader.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definition, in-depth analysis, and vivid descriptions of failure modes; the Limitations section is sincere and constructive.
*   **Value**: ⭐⭐⭐⭐⭐ Sounds a reliability alarm for the dLLM community and points toward future improvements via the identified failure modes.

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
