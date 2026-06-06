---
title: >-
  [Paper Note] Uncovering Visual Counting Bottlenecks in Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Visual Counting] By decomposing visual counting into three cognitive stages, this work discovers that VLM counting failures do not originate from visual perception or quantity understanding…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Counting"
  - "Systematic Generalization"
  - "Symbolic Mapping"
  - "VLM"
  - "Out-of-Distribution Generalization"
date: 2026-05-08
content_hash: 605fee74786ee570
---

# Uncovering Visual Counting Bottlenecks in Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.30170](https://arxiv.org/abs/2605.30170)  
**Code**: https://github.com/Russellpang/semproj  
**Area**: Multimodal VLM  
**Keywords**: Visual Counting, Systematic Generalization, Symbolic Mapping, VLM, Out-of-Distribution Generalization

## TL;DR
By decomposing visual counting into three cognitive stages, this work discovers that VLM counting failures do not originate from visual perception or quantity understanding, but from the symbolic mapping stage's inability to project visual representations onto the correct text tokens, reflecting the lack of a unified cross-modal numerical representation space.

## Background & Motivation

**Background**: Large VLMs perform excellently in interpolation tasks but struggle with systematic generalization, particularly in visual counting tasks.

**Limitations of Prior Work**: When the number of objects in an image exceeds the training distribution, VLM performance collapses immediately from near-perfect accuracy to near-random guessing, yet the specific causes of this failure remain unclear.

**Key Challenge**: Models can perfectly learn recursive counting rules in the text domain (counting to 99), but fail to generalize to 50 objects after being trained on only 49 in the visual domain—indicating a severe rupture between text and visual capabilities.

**Goal**: (1) Identify the specific bottleneck of counting failures; (2) Rule out visual perception or quantitative reasoning as the root cause; (3) Locate the failure at the symbolic mapping stage.

**Key Insight**: Decompose visual counting into three stages—visual individuation, quantity awareness, and symbolic mapping—verifying each step using linear probing techniques in both synthetic environments and state-of-the-art foundation models.

**Core Idea**: Utilize a decoupled diagnostic framework (Vision Gap and Language Gap) to prove that models retain correct internal visual quantity representations but fail to map them to corresponding text tokens, supporting the "Fragmented Number Hypothesis."

## Method

### Overall Architecture
A two-tier experimental design—first, strictly control the training distribution in a synthetic laboratory (custom-trained lightweight Toy VLM + Go board dataset), then validate the conclusions on a state-of-the-art foundation model (Qwen3-VL-32B). Failure causes are diagnosed across multiple dimensions using latent number probing, systematically excluding Hypotheses A and B to locate Hypothesis C.

### Key Designs

1.  **Decoupled Training Curriculum**:
    - **Function**: Simulates VLM pre-training dynamics and creates cross-modal distribution shifts.
    - **Mechanism**: Phase 1 language pre-training enables the decoder to master recursive successor functions; Phase 2 visual alignment restricts visual training to $N \le 49$. This creates a critical visual extrapolation (VE) regime (50–99)—where the model knows the labels but has never seen the corresponding visual density.
    - **Design Motivation**: Isolates cross-modal issues by creating a mismatch between text knowledge and visual experience, rather than relying on noisy real-world datasets.

2.  **Latent Number Probing Tool**:
    - **Function**: Detects the presence of objects in visual encoder outputs via a linear classifier $f_{probe}: \mathbb{R}^d \to \{0,1\}$, aggregated to obtain the latent number $N_H = \sum_{i=1}^L f_{probe}(z_i)$.
    - **Mechanism**: Evaluate probes trained only on in-distribution (ID, $N \le 49$) data within the extrapolation regime, defining the Vision Gap ($|N_H - N_G|$, perception error) and the Language Gap ($|N_H - N_P|$, language module alignment error).
    - **Design Motivation**: By checking the information content of visual representations directly, it bypasses the language decoder to precisely locate where the failure occurs.

3.  **Verifying Quantity Awareness via Comparative Counting Task**:
    - **Function**: Transforms the enumeration task into a binary classification—the model judges whether the cardinalities of two inputs are identical.
    - **Mechanism**: The model compares the equality of two quantities without needing to generate specific number tokens, bypassing the symbolic expression bottleneck to test if quantity awareness is preserved.
    - **Design Motivation**: If the model maintains $>90\%$ accuracy in the comparative task (even when explicit counting fails in the VE regime), it proves the quantity signal is not lost, and failure originates solely in the generation stage.

## Key Experimental Results

### Main Results

| Evaluation Set | Visual Counting Accuracy | Text Counting Accuracy | Meaning |
| :--- | :--- | :--- | :--- |
| In-distribution (ID, 0-49) | 100% | 100% | Perfect within training range |
| Visual Extrapolation (VE, 50-99) | 0% | 100% | Language ability does not map to vision automatically |
| Full Extrapolation (FE, 100-120) | 0% | ~99% | Text prior alone is insufficient |

### Ablation Study

| Stage | Vision Gap | Language Gap | Conclusion |
| :--- | :--- | :--- | :--- |
| Visual Individuation (H.A) | $\approx 0$ (Linear separability maintained) | Soars ($>0$) | Not a perceptual failure |
| Quantity Awareness (H.B) | Low | High, Comparative task $>90\%$ | Not a loss of reasoning |
| Symbolic Mapping (H.C) | Low | High, prediction collapses to "attractors" | **Confirmed Symbolic Mapping Bottleneck** |

### Key Findings
- Visual encoders maintain robust, linearly separable quantity representations in the extrapolation regime, excluding perceptual blindness.
- Models can accurately judge if quantities from different modalities are equal in comparative tasks even when they fail at explicit counting.
- Counting failures are not random noise but are structured—predictions collapse into "attractors" (visual training boundary 49, text priors 90/99, or low-frequency hallucinations like 9).
- Attention heads activated for visual vs. text counting are almost entirely non-overlapping (95.7% distinct), indicating the model uses two isolated "counting subroutines."
- Validations on Qwen3-VL show that the same separation phenomenon persists even after trillion-token pre-training, indicating this is an architectural property rather than a scaling issue.

## Highlights & Insights
- **Sophisticated Decomposition Framework**: Transforms the diagnosis of counting failure from a black box into a three-stage analysis, pinpointing symbolic mapping through orthogonal experiments.
- **Creative Use of Latent Probes**: Establishes causal links by combining linear probes with intervention analysis rather than merely detecting information presence.
- **Theoretical Insight via "Fragmented Number Hypothesis"**: Reveals that the fundamental VLM bottleneck lies in representation unity rather than computational capability.

## Limitations & Future Work
- The Go board task in synthetic experiments, while strictly controlled, is simplified.
- Validated only on the Qwen3-VL foundation model; the situation for other VLM architectures remains unknown.
- The paper diagnoses the problem but does not provide a definitive solution.
- The universality of this bottleneck in higher-order reasoning tasks beyond counting remains to be verified.

## Related Work & Insights
- **vs. Systematic Generalization Literature**: Previous work focused on visual distribution shifts; this paper is the first to systematically decompose multimodal generalization in VLMs, identifying representation ruptures between language and vision.
- **vs. VLM Benchmarking**: Existing evaluations only report accuracy metrics; this work delves into internal mechanisms (linear probing + circuit analysis) to reveal structural defects masked by accuracy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (First to strictly decompose VLM counting failure into three stages using causal diagnostic tools and proposing the Fragmented Number Hypothesis.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Dual-level validation with synthetic and real models + latent probes + intervention analysis + circuit tracing.)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Clear logical chain with sequential hypothesis testing.)
- **Value**: ⭐⭐⭐⭐⭐ (Provides profound understanding of multimodal model reliability, guiding VLM design and safety research.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)
- [\[ICML 2026\] Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination](are_vlms_seeing_or_just_saying_uncovering_the_illusion_of_visual_re-examination.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](../../ACL2026/multimodal_vlm/efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)

</div>

<!-- RELATED:END -->
