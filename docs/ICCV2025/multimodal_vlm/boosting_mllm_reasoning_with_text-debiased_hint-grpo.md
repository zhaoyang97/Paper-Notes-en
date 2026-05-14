---
title: >-
  [Paper Note] Boosting MLLM Reasoning with Text-Debiased Hint-GRPO
description: >-
  [ICCV 2025][Multimodal VLM][MLLM reasoning] This paper identifies two critical issues in applying GRPO to MLLM reasoning — low data utilization (invalid gradients when all sampled outputs for a hard question are incorrec…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "MLLM reasoning"
  - "GRPO reinforcement learning"
  - "data utilization"
  - "text bias"
  - "hint-guided training"
date: 2026-05-08
content_hash: 4eba8e5a05428b76
---

# Boosting MLLM Reasoning with Text-Debiased Hint-GRPO

**Conference**: ICCV 2025
**arXiv**: [2503.23905](https://arxiv.org/abs/2503.23905)
**Code**: [github.com/hqhQAQ/Hint-GRPO](https://github.com/hqhQAQ/Hint-GRPO)
**Area**: Multimodal Learning / MLLM Reasoning
**Keywords**: MLLM reasoning, GRPO reinforcement learning, data utilization, text bias, hint-guided training

## TL;DR

This paper identifies two critical issues in applying GRPO to MLLM reasoning — low data utilization (invalid gradients when all sampled outputs for a hard question are incorrect) and text bias (the model ignores visual input and relies solely on textual reasoning) — and proposes two corresponding solutions: Hint-GRPO (adaptively providing reasoning hints) and text-debiasing calibration (enhancing image conditioning at test time). The approach achieves significant reasoning improvements across 11 datasets on 3 base MLLMs.

## Background & Motivation

MLLM reasoning methods fall into two categories:
- **PRM (Process Reward Methods)**: supervise intermediate reasoning steps, but DeepSeek-R1 notes that PRMs struggle to accurately assess step quality and suffer from severe reward hacking.
- **ORM (Outcome Reward Methods)**: supervise only final answers (e.g., GRPO), which are easier to evaluate accurately and generalize more broadly.

GRPO performs well in LLM reasoning but exhibits two key problems when applied to MLLMs:

**Problem 1 — Low Data Utilization**: GRPO samples $G$ outputs per question. When all outputs are incorrect ($r_i = 0$ for all $i$), the normalized advantage $\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})} = 0$, leaving only the KL regularization term in the gradient. Empirically, the fraction of valid samples for Qwen2-VL-7B on mathematical reasoning is only 40–50%.

**Problem 2 — Text Bias**: As GRPO training progresses, MLLM accuracy improves even when image conditioning is removed — indicating the model has learned to reason from text alone while ignoring visual input. This occurs because many queries are sufficiently described by text, but the model fails when text alone is insufficient.

## Method

### Overall Architecture

Hint-GRPO extends standard GRPO by adaptively providing reasoning hints for hard questions to improve data utilization, combined with test-time text-debiasing calibration to strengthen image-conditioned reasoning. The pipeline is as follows: invalid GRPO samples → multiple output groups with varying hint levels → selection of the most appropriate group for training → test-time CFG-style calibration to enhance visual perception.

### Key Designs

1. **Hint-GRPO — Hint-Guided Training**:

    - **Data construction**: Based on the LLaVA-CoT dataset, GPT-4o is used to decompose long reasoning chains into structured steps; multiple-choice questions are converted to fill-in-the-blank format to eliminate random guessing.
    - **Hint injection modes**:
        - $\mathcal{I}_{query}$ (hint injected into query): appends the hint to the query text → train-test inconsistency, poor performance.
        - $\mathcal{I}_{answer}$ (hint injected into answer): keeps the query unchanged and prepends the hint as the beginning of the model output, letting the model complete the remaining reasoning steps → train-test consistent, significantly better performance.
    - **Adaptive hint strategy**:
        - Define hint ratio $\alpha \in [0,1]$; use the first $L \cdot \alpha$ steps from $L$ correct reasoning steps as the hint.
        - Extend GRPO's single output group to $M$ groups ($M=3$), each with a different hint ratio $\alpha_i = \frac{i-1}{M}$.
        - Iterate from low to high hint ratios and select the first group containing at least one correct output for training.
        - This avoids both low utilization and excessive hinting that impedes reasoning learning.
        - Incurs only a 20.5% increase in training time over standard GRPO.

2. **Text-Debiasing Calibration (Test Time)**:

    - Inspired by Classifier-Free Guidance (CFG) in image generation.
    - Computes token logits both with and without image conditioning: $\hat{\pi}_\theta(o_t | q_{img})$ and $\hat{\pi}_\theta(o_t)$.
    - Calibration formula:
    $$\hat{\pi}_\theta^{calibrated}(o_t | q_{img}) = \hat{\pi}_\theta(o_t | q_{img}) + \gamma \cdot (\hat{\pi}_\theta(o_t | q_{img}) - \hat{\pi}_\theta(o_t))$$
    - $\gamma=0.8$ controls image conditioning strength, pushing calibrated logits away from the image-free distribution and toward the image-conditioned distribution.

3. **Data Utilization Metric**:

    - Defines the valid sample ratio $S_{valid} = \frac{1}{B}\sum_{k=1}^B \mathbb{1}\{\text{std}(\mathbf{r}(z_k)) \neq 0\}$.
    - Hint-GRPO substantially raises $S_{valid}$ from the 40–50% baseline.

### Loss & Training

- Based on the standard GRPO objective with a KL regularization term to constrain policy drift.
- AdamW optimizer, learning rate 5e-5, trained for 2 epochs on 8 GPUs.
- Accelerated with DeepSpeed ZeRO-3; vLLM used for generation (1 GPU for generation, 7 GPUs for training).
- System prompt formats outputs as `<think>...</think> <answer>...</answer>`.

## Key Experimental Results

### Main Results (Geometric Reasoning — Geo170K Accuracy)

| Method | Geo170K | MathVista | MMStar | MathVerse | Math-Vision | Avg |
|--------|---------|-----------|--------|-----------|-------------|-----|
| Qwen2-VL-7B (base) | 30.63 | 44.50 | 40.52 | 27.92 | 10.89 | 30.40 |
| SFT | 37.53 | 41.66 | 37.07 | 14.47 | 2.86 | 25.50 |
| Mulberry (PRM) | 33.55 | 52.17 | 42.24 | 17.68 | 6.06 | 32.08 |
| Open-R1-Multimodal | 35.68 | 45.55 | 40.52 | 28.78 | 11.43 | 31.56 |
| R1-V | 38.72 | 47.26 | 41.38 | 28.12 | 12.51 | 33.19 |
| GRPO | 38.46 | 48.82 | 42.24 | 30.10 | 12.02 | 33.92 |
| Hint-GRPO | 45.62 | 52.77 | 43.97 | 31.68 | 14.38 | 37.60 |
| **Debiased Hint-GRPO** | **46.68** | **54.19** | **45.69** | **32.18** | **14.99** | **38.55** |

### Ablation Study

| Configuration / Hyperparameter | Geo170K Acc |
|-------------------------------|-------------|
| GRPO + $\mathcal{D}_{original}$ (with multiple-choice) | 35.81 |
| GRPO + $\mathcal{D}_{new}$ (fill-in-the-blank) | 38.46 |
| Hint-GRPO + $\mathcal{I}_{query}$ | 41.64 |
| **Hint-GRPO + $\mathcal{I}_{answer}$** | **45.62** |
| Fixed $\alpha=0.25$ | Good but suboptimal |
| Fixed $\alpha=0.75$ | Harmful |
| Random $\alpha$ | Better than fixed |
| **Adaptive $\alpha$ ($M=3$)** | **Best** |
| $\gamma=0$ (no calibration) | 45.62 |
| $\gamma=0.8$ (calibration) | **46.68** |
| $\gamma=1.6$ (over-calibration) | 44.69 |

### General Multimodal Reasoning (Llama-3.2-11B-Vision)

| Method | MMStar | MMBench | MMVet | MathVista | AI2D | Hallusion | Avg |
|--------|--------|---------|-------|-----------|------|-----------|-----|
| Base | 49.8 | 65.8 | 57.6 | 48.6 | 77.3 | 40.3 | 56.6 |
| LLaVA-o1 | 57.6 | 75.0 | 60.3 | 54.8 | 85.7 | 47.8 | 63.5 |
| **Ours** | **60.7** | **75.8** | **64.2** | **56.8** | **86.6** | **50.7** | **65.8** |

### Key Findings

- SFT degrades OOD performance on geometric reasoning, suggesting it encourages rote memorization rather than generalizable reasoning.
- Converting multiple-choice to fill-in-the-blank improves GRPO accuracy from 35.81 to 38.46 by eliminating lucky-guess shortcuts.
- Answer-injected hints substantially outperform query-injected hints (45.62 vs. 41.64), attributable to train-test consistency.
- $\gamma=0.8$ is the optimal calibration strength; excessive calibration ($\gamma=1.6$) over-corrects and degrades performance.
- Increasing adaptive groups $M$ from 1 to 3 yields consistent gains; $M=4$ saturates, and $M=3$ is selected for efficiency.

## Highlights & Insights

- The formal analysis of low data utilization is clear and rigorous: gradient decomposition demonstrates that all-zero advantages render samples entirely uninformative.
- The design philosophy of Hint-GRPO is elegant — framing hint provision as "prepending partial reasoning as a prefix" preserves train-test consistency.
- The discovery of text bias is insightful: an increasing no-image accuracy during GRPO training serves as a meaningful warning signal.
- The transfer of CFG from image generation to text debiasing in MLLM reasoning is a compelling cross-domain adaptation.

## Limitations & Future Work

- Hint-GRPO depends on high-quality step-level reasoning data (LLaVA-CoT decomposed by GPT-4o), which is costly to obtain.
- Gains on general multimodal reasoning are less pronounced than on geometric reasoning; more robust accuracy estimation methods (e.g., relaxed matching via IoU) may be needed.
- Text-debiasing calibration requires two forward passes, increasing inference latency.
- Multi-group sampling introduces additional generation overhead.

## Related Work & Insights

- The adaptive hint strategy could be further refined by drawing on curriculum learning principles.
- Text bias likely affects other multimodal tasks (e.g., visual question answering) as well.
- The low data utilization problem in GRPO may be alleviated through improved sampling strategies such as importance sampling.

## Rating
- Novelty: ⭐⭐⭐⭐ — Both identified problems and proposed solutions are creative, though the core contribution remains an engineering improvement upon GRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 base models, 11 datasets, and detailed ablations; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is thorough and gradient derivations are clearly presented.
- Value: ⭐⭐⭐⭐⭐ — Directly improves GRPO-based MLLM reasoning training; the method is simple, practical, and readily reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[AAAI 2026\] AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs](../../AAAI2026/multimodal_vlm/abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)
- [\[AAAI 2026\] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](../../AAAI2026/multimodal_vlm/astar_boosting_multimodal_reasoning_with_automated_structure.md)
- [\[ICLR 2026\] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](../../ICLR2026/multimodal_vlm/diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)

</div>

<!-- RELATED:END -->
