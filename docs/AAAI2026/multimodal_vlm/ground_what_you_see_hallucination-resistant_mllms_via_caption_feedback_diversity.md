---
title: >-
  [Paper Note] Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization
description: >-
  [AAAI2026][Multimodal VLM][MLLM hallucination] This paper identifies three root causes of hallucination in RL-based MLLM training—visual misinterpretation, limited exploration diversity, and sample conflict—and addresses each with Caption Reward, reward-variance-guided sample selection, and NTK-similarity-based InfoNCE regularization, achieving significant hallucination reduction across multiple benchmarks.
tags:
  - AAAI2026
  - Multimodal VLM
  - MLLM hallucination
  - reinforcement-learning
  - GRPO
  - caption reward
  - NTK similarity
  - InfoNCE
date: 2026-05-08
content_hash: 02d4fbd3ef9e26c5
---

# Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization

**Conference**: AAAI2026
**arXiv**: [2601.06224](https://arxiv.org/abs/2601.06224)
**Code**: [ZJU-OmniAI/OMNEX-VL](https://github.com/ZJU-OmniAI/OMNEX-VL)
**Area**: Multimodal VLM
**Keywords**: MLLM hallucination, reinforcement-learning, GRPO, caption reward, NTK similarity, InfoNCE

## TL;DR

This paper identifies three root causes of hallucination in RL-based MLLM training—visual misinterpretation, limited exploration diversity, and sample conflict—and addresses each with Caption Reward, reward-variance-guided sample selection, and NTK-similarity-based InfoNCE regularization, achieving significant hallucination reduction across multiple benchmarks.

## Background & Motivation

MLLMs have demonstrated strong performance on visual question answering and video understanding, yet **hallucination**—generating fluent responses that are factually inconsistent with visual inputs—severely limits real-world deployment, particularly in safety-critical scenarios. Recent work has introduced reinforcement learning (RL) into MLLM training to enhance reasoning ability (e.g., DeepSeek-R1-style GRPO training); however, RL training can paradoxically **amplify hallucinations**, causing models to fall into semantically redundant reasoning loops or produce mismatches between the thinking process and the final answer.

The authors systematically analyze three root causes of hallucination during RL training:

1. **Visual Misinterpretation**: The model generates inaccurate visual descriptions early in the reasoning chain, anchoring subsequent reasoning to erroneous information; alternatively, the model attends to inputs at a coarse level and produces redundant, irrelevant reasoning.
2. **Limited Exploration**: Insufficient sampling diversity during policy optimization leads to overconfident outputs and overfitting.
3. **Sample Conflict**: Gradient updates for one sample inadvertently corrupt the model's predictions for unrelated samples, introducing spurious correlations.

## Core Problem

How can RL-based MLLM training simultaneously address inaccurate visual grounding, insufficient policy exploration, and cross-sample gradient interference as sources of hallucination?

## Method

The overall framework consists of three modules applied jointly on top of GRPO:

### 1. Visual-Grounded Reasoning Enhancement

**Redefining the reasoning paradigm**: Two new stages are inserted before the standard thinking → answer pipeline:

- **Planning stage**: Proactively locates visual regions relevant to the question.
- **Caption stage**: Generates concise textual descriptions of those regions as intermediate anchoring points for subsequent reasoning.

The complete pipeline becomes: planning → caption → thinking → answer.

**Caption Reward**: The model-generated caption is extracted and, together with the question (without the image), fed into a separate LLM. A positive reward is granted if this LLM can correctly answer the question from the caption alone; otherwise the reward is zero. This ensures that the caption accurately reflects visual content, preventing erroneous descriptions from propagating through the reasoning chain.

### 2. Reward Variance-Guided Sample Selection

By deriving gradients of the GRPO loss with respect to logits, the paper establishes the following key conclusions:

- Positive advantage ($A_{i,t} > 0$) → distribution sharpening: reinforcing correct answers.
- Negative advantage ($A_{i,t} < 0$) → distribution flattening: suppressing incorrect high-confidence outputs.

Based on this, the paper categorizes samples into three types:

| Type | Characteristic | Training Effect |
|------|---------------|-----------------|
| Easy | High mean, low variance | Risk of overfitting; distribution overly sharpened |
| Hard | Low mean, low variance | Model cannot learn; distribution remains flat |
| **Medium** | **High variance** | **Most valuable**: explores then converges; ideal learning trajectory |

In practice, 64 responses are generated per input; the mean and variance of their rewards are computed, and only the top 50% by variance are retained for RL training.

### 3. Conflict-Aware Regularization

The paper employs Neural Tangent Kernel (NTK) analysis to characterize cross-sample gradient interference:

$$\Delta\log\pi^t(\mathbf{y}_u|\mathbf{x}_o) \propto \eta \cdot \mathcal{A}^t(x_o) \cdot \mathcal{K}^t(x_o, x_u) \cdot \nabla_z\log\pi^t(y_u|x_u) \cdot A_{u,t}$$

where $\mathcal{K}^t(x_o, x_u)$ denotes the NTK similarity. When NTK similarity is excessively high, updating on one sample substantially alters the model's predictions for unrelated samples.

**Key insight**: Simply minimizing NTK similarity is inadvisable (as it would suppress beneficial interactions); rather, the similarity should be **regulated to an appropriate range**.

The procedure is as follows:

- The cosine similarity of last-layer log-probability gradients is used to approximate NTK similarity.
- Sample pairs are divided into positive pairs (similarity too low, to be brought closer) and negative pairs (similarity too high, to be pushed apart) based on threshold $\tau$.
- An InfoNCE loss is applied: $\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\sum_{j\in\mathcal{P}(i)}\exp(\text{sim}(f_i,f_j))}{\sum_{k=1}^{B}\mathbb{I}_{[k\neq i]}\exp(\text{sim}(f_i,f_k))}$
- The optimal threshold is empirically determined as $\tau = 0.54$.

## Key Experimental Results

Based on Qwen-VL-2.5-7B, results are compared across multiple benchmarks:

| Model | MMVU | VideoHallucer | POPE | MMBench |
|-------|------|---------------|------|---------|
| Qwen-VL-2.5-7B (baseline) | 57.6 | 46.5 | 84.4 | 86.3 |
| + SFT | 62.7 | 43.5 | 82.2 | 83.9 |
| + GRPO | 62.1 | **42.3↓** | 83.6 | 86.8 |
| + **Ours** | **65.6** | **50.8** | **88.7** | **88.6** |
| GPT-4o | 75.4 | 53.3 | 86.9 | 83.4 |

Key findings:

- Standard GRPO training **degrades** VideoHallucer score (46.5→42.3), confirming that RL does introduce hallucinations.
- The proposed method achieves 88.7% on POPE (surpassing GPT-4o's 86.9%) and 88.6% on MMBench (surpassing GPT-4o's 83.4%).
- 65.6% on MMVU represents the highest score among all open-source models.

Ablation Study:

- Removing Caption + Caption Reward: MMVU 65.6→62.6, POPE 88.7→85.2.
- Training with only Easy/Hard samples: performance is consistently lower than with Medium samples.
- Removing InfoNCE Loss: MMVU 65.6→63.8, POPE 88.7→86.8.

## Highlights & Insights

1. **Thorough problem analysis**: The paper theoretically derives the effect of positive and negative advantage on output distributions during RL training, providing a principled justification for sample selection.
2. **Novel NTK perspective**: This work is the first to connect NTK similarity to cross-sample conflict in MLLM training and proposes the idea of "regulation rather than elimination."
3. **Elegant Caption Reward design**: Visual grounding quality is assessed indirectly by asking whether a caption alone suffices to answer the question, requiring no additional annotation.
4. **Three orthogonal and complementary modules**: The modules respectively target the entry point of the reasoning chain (caption), sample selection (variance), and the optimization process (NTK regularization).

## Limitations & Future Work

1. **Area miscategorization**: The paper addresses MLLM hallucination mitigation and would be better classified under multimodal_vlm than object_detection.
2. **High computational cost**: Generating 64 responses per sample for variance estimation, combined with NTK similarity computation and InfoNCE loss, substantially increases training overhead.
3. **Threshold $\tau$ requires tuning**: The NTK similarity threshold of 0.54 is obtained via empirical search and may require adjustment for different datasets.
4. **Limited benchmark coverage**: Evaluation is primarily conducted on video understanding and hallucination detection benchmarks; broader visual reasoning tasks (e.g., visual grounding, referring expression comprehension) are not assessed.
5. **Gap with GPT-4o remains**: Notable gaps persist on MMVU (65.6 vs. 75.4) and VideoHallucer (50.8 vs. 53.3).

## Related Work & Insights

| Method | Core Idea | Difference from Ours |
|--------|-----------|----------------------|
| Vision-R1 | Progressive thinking suppression training + decoupled reward | Does not address error propagation from visual descriptions |
| R1-VL | StepGRPO step-level consistency | Does not consider cross-sample gradient conflict |
| Video-R1 | Temporal consistency reward | Targets only video temporality; does not address general hallucination |
| Standard GRPO | Group relative policy optimization | This paper demonstrates that standard GRPO exacerbates hallucination |

The key distinction of this work lies in **simultaneously** addressing hallucination across three dimensions: reasoning paradigm, sampling strategy, and optimization regularization.

### Broader Connections

- **Generalizability of Caption Reward**: Using an independent model to verify the quality of intermediate reasoning artifacts can be extended to other scenarios requiring process-level supervision.
- **Variance-guided selection relates to curriculum learning**: Medium samples are essentially those at the model's learning boundary, analogous to the zone of proximal development.
- **NTK regularization is applicable to other multi-task/multi-sample training settings**: Any scenario requiring reduced cross-sample gradient interference may benefit from this approach.

## Rating

- Novelty: ⭐⭐⭐⭐ — The NTK perspective and Caption Reward design are genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations are comprehensive, though benchmark coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is clear and the three-module logic is coherent.
- Value: ⭐⭐⭐⭐ — Systematically addresses MLLM hallucination in RL training with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)
- [\[CVPR 2026\] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](../../CVPR2026/multimodal_vlm/where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)
- [\[NeurIPS 2025\] Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling](../../NeurIPS2025/multimodal_vlm/enhancing_the_outcome_reward-based_rl_training_of_mllms_with_self-consistency_sa.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)

</div>

<!-- RELATED:END -->
