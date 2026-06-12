---
title: >-
  [Paper Note] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought
description: >-
  [NeurIPS 2025][LLM Reasoning][Multimodal CoT] This paper proposes "Visual Thoughts" as a unified framework for interpreting the effectiveness of multimodal chain-of-thought reasoning (MCoT). The core mechanism underlying…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Multimodal CoT"
  - "Visual Thoughts"
  - "T-MCoT"
  - "I-MCoT"
  - "Visual Information Transfer"
date: 2026-05-08
content_hash: 35b3836e60c36511
---

# Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought

**Conference**: NeurIPS 2025
**arXiv**: [2505.15510](https://arxiv.org/abs/2505.15510)  
**Code**: None  
**Area**: LLM Reasoning / Multimodal
**Keywords**: Multimodal CoT, Visual Thoughts, T-MCoT, I-MCoT, Visual Information Transfer

## TL;DR

This paper proposes "Visual Thoughts" as a unified framework for interpreting the effectiveness of multimodal chain-of-thought reasoning (MCoT). The core mechanism underlying performance gains in both textual MCoT (T-MCoT) and interleaved multimodal MCoT (I-MCoT) is the caching and transfer of visual information into the reasoning process. The paper defines four forms of visual thought expressions and reveals their role as image-to-reasoning intermediaries in deep Transformer layers.

## Background & Motivation

**Background**: Multimodal CoT (MCoT) for large vision-language models (LVLMs) encompasses two major paradigms: (1) T-MCoT (Textual MCoT), which accepts multimodal inputs and produces purely textual reasoning steps; and (2) I-MCoT (Interleaved MCoT), which generates reasoning outputs interleaving text and images. Both paradigms have their proponents—some argue that I-MCoT better mirrors human cognition, while others find T-MCoT superior in mathematical settings.

**Limitations of Prior Work**: No unified explanation exists for the effectiveness of these two paradigms. In particular, it remains unclear: (1) why different MCoT paradigms work; (2) which paradigm is preferable under what conditions; and (3) what the underlying mechanism is by which MCoT improves reasoning. No single framework addresses all these questions simultaneously.

**Key Challenge**: T-MCoT and I-MCoT each excel on different tasks, yet a theoretical framework to explain this divergence—and to guide paradigm selection—is lacking.

**Goal**: To provide a unified perspective for understanding how different MCoT paradigms enhance multimodal reasoning in LVLMs.

**Key Insight**: Drawing an analogy from computer systems—raw images resemble external storage (requiring reprocessing on each access), while visual thoughts resemble a cache (distilling critical visual information for fast retrieval).

**Core Idea**: The central value of MCoT lies in generating "visual thoughts"—distilling and caching task-relevant visual information into the reasoning chain, thereby reducing dependence on raw images and enabling more efficient and deeper subsequent reasoning.

## Method

### Overall Architecture

The paper formally defines visual thoughts as a special type of MCoT reasoning step that extracts information from visual inputs and transfers it to subsequent reasoning steps. It then systematically explores four forms of visual thought expression, validates their effectiveness through controlled experiments, and employs attention analysis to reveal internal mechanisms.

### Key Designs

1. **Formal Definition and Validation of Visual Thoughts**:
    - **Function**: Define visual thoughts and verify their necessity through ablation experiments.
    - **Mechanism**: Three controlled conditions are designed—(1) standard I-MCoT (with visual thoughts); (2) removal of the visual thought cache, forcing the model to re-analyze the raw image (w/o VT); (3) replacement of image-form visual thoughts with textual descriptions (text-form VT). Results show that removing VT degrades performance (even below direct query-based reasoning), while restoring VT consistently improves reasoning.
    - **Design Motivation**: To rule out the hypothesis that MCoT is beneficial merely by adding reasoning steps—the visual information cached in VT is the true driver.

2. **Systematic Exploration of Four Visual Thought Expressions**:
    - **Function**: Define and compare four visual thought expressions across two modalities (text and image).
    - **Mechanism**: **Natural Language (N-LANG)**—prompting the LVLM to generate an image description as a reasoning prefix; **Structured Language (S-LANG)**—generating a scene graph in JSON format; **Edited Image (E-IMG)**—applying visual tools (grounding/segmentation/depth) to edit the original image; **Generated Image (G-IMG)**—using DALL-E 3 to synthesize a new image conditioned on the query as a reasoning aid. The four forms vary in terms of clarity and conciseness.
    - **Design Motivation**: To systematically cover a 2×2 combination of text vs. visual modality and free-form vs. structured format.

3. **Internal Information Flow Analysis in Transformers**:
    - **Function**: Reveal how visual thoughts transfer visual information within LVLMs.
    - **Mechanism**: Attention map analysis shows that in deep Transformer layers, visual thought tokens become the primary intermediary through which input image information is relayed to reasoning tokens. In standard reasoning, attention to image tokens decays with depth; with visual thoughts, information first flows to VT tokens and then propagates to deep reasoning tokens—enabling higher-level visual understanding.
    - **Design Motivation**: To explain the effectiveness of VT from the perspective of internal model mechanisms, beyond surface-level performance metrics.

## Key Experimental Results

### Main Results

| Model | Method | MMVP | V*Bench | M3CoT | CoMT | AVG |
|-------|--------|------|---------|-------|------|-----|
| LLaVA-1.5-7B | w/o VT | 43.42 | 44.44 | 26.83 | 16.00 | 34.36 |
| LLaVA-1.5-7B | N-LANG | 52.63 | 46.67 | 32.52 | 17.50 | 38.58 |
| LLaVA-1.5-7B | S-LANG | 52.63 | 51.11 | 31.71 | 20.50 | 39.50 |
| LLaVA-1.5-7B | E-IMG | 50.00 | 48.89 | 34.15 | 23.00 | 40.10 |
| LLaVA-1.5-7B | G-IMG | 48.68 | 55.56 | 39.02 | 25.00 | 42.27 |
| Qwen2-VL-7B | w/o VT | 55.26 | 80.00 | 74.80 | 18.00 | 56.11 |
| Qwen2-VL-7B | S-LANG | 68.42 | 85.56 | 79.67 | 20.00 | 60.41 |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Image-form VT vs. text-form VT | CoMT-Selection accuracy | Image VT exceeds text VT by 47.83%, especially in complex scenes |
| VT vs. plain caption | Accuracy on complex scenes | VT improvement exceeds 7% when brief captions lose detail |
| Remove VT (w/o VT) | All tasks | Performs worse than direct query-based reasoning—VT positions are wasted |
| Different VLM scales | Consistency of performance gain | All four VLMs benefit; gains correlate with model capability |

### Key Findings
- G-IMG (generated image) achieves the best performance on LLaVA (AVG 42.27 vs. w/o VT 34.36)—generating new images can highlight task-critical information.
- Image-form VT shows particularly pronounced advantages in complex scenes, reflecting the natural superiority of the visual modality for transmitting visual information.
- VT is distinct from simple captioning: captions are effective only in simple scenarios, whereas VT yields substantially larger gains in complex settings.
- Attention analysis confirms that VT tokens serve as the bridge through which image information is propagated to deeper layers.

## Highlights & Insights
- The "cache" analogy is highly intuitive—conceptualizing visual thoughts as a cache layer for image information avoids redundant processing of raw images.
- The systematic comparison of four VT expression forms provides practical guidance for MCoT method selection.
- The internal attention analysis goes beyond surface-level performance—explaining VT as an image-to-reasoning intermediary from an information flow perspective.
- The unified framework bridges the T-MCoT vs. I-MCoT debate: what matters is not the form but the clarity and efficiency of visual information transfer.

## Limitations & Future Work
- The four VT expression forms require additional tools (DALL-E 3, visual models, etc.), increasing inference cost.
- Experiments are conducted primarily on 7B-scale models; whether larger models (e.g., GPT-4V) exhibit the same VT dependency remains uncertain.
- The attention analysis is descriptive; causal intervention experiments are lacking to confirm whether VT's intermediary role is indeed causal.
- Automatic VT selection strategies are not explored—choosing the appropriate VT expression for a given scenario still requires manual decision-making.

## Related Work & Insights
- **vs. Visual Sketchpad**: Visual Sketchpad is a representative I-MCoT approach; this paper subsumes it together with T-MCoT under the unified visual thoughts framework.
- **vs. Textual CoT**: Textual CoT enhances reasoning capability but does not improve visual information acquisition; the unique value of VT lies in strengthening visual information transfer.
- **vs. Description-then-Reason**: The simple "describe then reason" paradigm is merely a special case of VT (N-LANG); other expression forms may be more effective.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposing a unified perspective for understanding MCoT constitutes a valuable conceptual contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four VT forms × four VLMs × multiple benchmarks × attention analysis—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Framework definitions are rigorous; the cache analogy is intuitive.
- Value: ⭐⭐⭐⭐ Provides a unified analytical language and a systematic comparison baseline for MCoT research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[ICCV 2025\] Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization](../../ICCV2025/llm_reasoning/unsupervised_visual_chain-of-thought_reasoning_via_preference_optimization.md)
- [\[NeurIPS 2025\] Note 1: Is CoT a Hallucination? A Data Distribution Perspective](is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)

</div>

<!-- RELATED:END -->
