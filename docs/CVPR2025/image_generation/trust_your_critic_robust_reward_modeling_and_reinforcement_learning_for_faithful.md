---
title: >-
  [Paper Note] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation
description: >-
  [CVPR 2025][Image Generation][Reward Model] This work proposes the FIRM framework, which trains specialized reward models (FIRM-Edit-8B / FIRM-Gen-8B) via "difference-first" (for editing) and "plan-and-score" (for generation) data construction pipelines. Combined with a "Base-and-Bonus" reward strategy (CME/QMA) to resolve reward hacking in RL, it achieves SOTA results on both image editing and T2I generation tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Reward Model"
  - "Reinforcement Learning"
  - "Image Editing"
  - "T2I Generation"
  - "Reward Hacking"
  - "MLLM"
date: 2026-05-08
content_hash: 88795ca5c02b7fbc
---

# Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation

**Conference**: CVPR 2025  
**arXiv**: [2603.12247](https://arxiv.org/abs/2603.12247)  
**Code**: [https://github.com/VisionXLab/FIRM-Reward](https://github.com/VisionXLab/FIRM-Reward)  
**Area**: Image Generation and Editing / Reward Modeling  
**Keywords**: Reward Model, Reinforcement Learning, Image Editing, T2I Generation, Reward Hacking, MLLM

## TL;DR

This work proposes the FIRM framework, which trains specialized reward models (FIRM-Edit-8B / FIRM-Gen-8B) via "difference-first" (for editing) and "plan-and-score" (for generation) data construction pipelines. Combined with a "Base-and-Bonus" reward strategy (CME/QMA) to resolve reward hacking in RL, it achieves SOTA results on both image editing and T2I generation tasks.

## Background & Motivation

**Rise of RL in Image Generation/Editing**: Methods such as DDPO and DPOK model the diffusion denoising process as an MDP and directly optimize the policy using PPO; Edit-R1 utilizes GRPO to achieve progress in editing tasks.

**Unreliable Critic as the Core Bottleneck**: General MLLMs (e.g., Qwen3-VL) acting as zero-shot reward models suffer from severe hallucinations, object omission, and weak spatial reasoning, thereby providing noisy and misleading reward signals for RL.

**Counter-intuitive Finding — MLLM as Evaluator vs. Descriptor**: While MLLMs frequently miss fine-grained details when directly judging editing quality, they perform exceptionally well in "problem-solving" (e.g., describing differences between two images). This inspires a "difference-first" evaluation design.

**Severity of Reward Hacking**: Simple linear combinations of rewards (such as $0.5 \cdot \text{Execution} + 0.5 \cdot \text{Consistency}$) lead the model to find shortcuts, such as outputting images almost identical to the inputs to obtain high Consistency scores. In T2I, models generate black silhouettes for short prompts to technically satisfy text conditions.

**Scaling Reward Models $\neq$ Better**: Although Qwen3-VL-32B is larger than Qwen3-VL-8B, it results in a $-0.07$ performance drop on ImgEdit. Simply scaling up general VLMs does not guarantee better reward modeling.

**Lack of a Unified Editing and Generation Evaluation Benchmark**: Existing benchmarks either cover only editing or only generation, and lack evaluation sets with rigorous human annotation and balanced score distributions.

## Method

### Overall Architecture

FIRM comprises four core components: Data Construction Pipeline $\rightarrow$ Reward Model Training $\rightarrow$ Evaluation Benchmark $\rightarrow$ RL Reward Strategy.

### Key Designs

#### 1. FIRM-Edit Data Pipeline ("Difference-First")

1. **Two-Level Difference Analysis**: Provide the original and edited images to a SOTA MLLM, requiring it to identify both obvious and subtle modifications to generate a unified difference report.
2. **Difference-Conditioned Evaluation**: Input the difference description, image pair, and editing instruction together into an MLLM evaluator $\rightarrow$ output Execution (1-5 score) and Consistency (1-5 score).
3. **Core Insight**: Prompting the model to "see differences" before scoring is significantly more accurate than direct scoring, effectively transforming the evaluation problem into a description task that the model excels at.
4. **Low-Score Sample Balancing**: Intentionally construct low-quality matches by rewriting instructions, ensuring a uniform distribution of scores across the 1-5 range.
5. **Data Sources**: OpenGPT-4o-Image, GPT-Image-Edit, ShareGPT-4o-Image, ImgEdit $\rightarrow$ 370K samples in total.

#### 2. FIRM-Gen Data Pipeline ("Plan-and-Score")

1. **Stage 1 — Explicit Criteria Planning**: The LLM (Qwen3-32B) extracts an evaluation checklist from the generation prompt, covering subject accuracy, style alignment, negative constraints, etc.
2. **Stage 2 — Structured Analysis & Scoring**: The MLLM (Qwen3-VL-235B) inspects the generated image item-by-item according to the checklist $\rightarrow$ aggregates into a final score.
3. **Multi-Model Generation for Diversity**: Generate images from the same prompt using 5 models of different architectures/capacities (Ovis, Z-image, Flux.1-dev, SDXL, SD1.5) to prevent the reward model from overfitting to the artifacts of a single generator.
4. **293K samples in total**

#### 3. "Base-and-Bonus" Reward Strategy

**Editing: Consistency-Modulated Execution (CME)**:
$$R_{\text{CME}} = \text{Execution} \cdot (0.6 + 0.4 \cdot \text{Consistency})$$
- Execution operates as a necessary condition: If Execution is low, the total reward remains suppressed regardless of how high Consistency is.
- Consistency serves as a bonus: Refining structural fidelity on the premise that a valid edit was executed.

**Generation: Quality-Modulated Alignment (QMA)**:
$$R_{\text{QMA}} = \text{InsFollowing} \cdot (0.4 + 0.6 \cdot \text{Quality})$$
- Instruction following is the foundation, with image quality acting as a modulating factor.
- Prevents hacking behaviors such as generating black silhouettes for short prompts.

#### FIRM-Bench

- 807 human-annotated samples: FIRM-Bench-Edit (301 Execution + 256 Consistency) + FIRM-Bench-Gen (250 Instruction Following).
- Strict data isolation (no overlap with training data) and a uniform distribution of ground truth scores.
- Evaluation Metric: MAE (Mean Absolute Error between predicted scores and human annotations).

## Key Experimental Results

### FIRM-Bench Reward Model Evaluation (MAE ↓)

| Model | Edit Exec. | Edit Cons. | Edit Total | Gen Total |
|------|-----------|-----------|--------|--------|
| GPT-5 | 0.62 | 0.73 | 0.67 | 0.52 |
| Gemini-3-Pro | 0.54 | 0.57 | **0.55** | **0.40** |
| Qwen3-VL-235B | 0.72 | 0.91 | 0.81 | 0.56 |
| Qwen3-VL-8B | 0.66 | 1.12 | 0.87 | 0.63 |
| **FIRM-Edit-8B** | **0.53** | **0.73** | **0.62** | — |
| **FIRM-Gen-8B** | — | — | — | **0.51** |

### Image Editing RL Performance (Table 3)

| Model | GEdit-Bench Overall | ImgEdit Overall |
|------|-------------------|-----------------|
| Qwen-Image-Edit-2509 (baseline) | 7.54 | 4.35 |
| + RL w/ Qwen3-VL-8B | 7.69 (+0.15) | 4.36 (+0.01) |
| + RL w/ Qwen3-VL-32B | 7.65 (+0.11) | 4.28 (−0.07) |
| + **FIRM-Qwen-Edit** | **7.84 (+0.30)** | **4.42 (+0.07)** |
| GPT-Image | 7.53 | 4.20 |
| UniWorld-Qwen | 7.76 | 4.48 |

### T2I Generation RL Performance (Table 4)

| Model | GenEval | DPG-Bench | TIIF | UniGen Short | UniGen Long |
|------|---------|-----------|------|-------------|-------------|
| SD3.5-Medium (baseline) | 0.52 | 84.08 | 70.17 | 60.71 | 64.67 |
| + RL w/ Qwen3-VL-8B | 0.76 | 86.87 | 75.99 | 67.17 | 74.50 |
| + **FIRM-SD3.5** | **0.77** | **87.16** | **77.12** | **69.56** | **76.22** |
| BAGEL | 0.82 | 85.07 | 71.50 | 59.91 | 71.26 |

### Reward Strategy Ablation (Table 5, based on FIRM-Edit-8B)

| Method | GEdit Overall | ImgEdit |
|------|-------------|---------|
| Baseline (w/o RL) | 7.54 | 4.35 |
| Weighted (0.5+0.5) | 1.06 | 2.17 |
| Weighted (0.6+0.4) | 6.51 | 3.73 |
| Edit-R1 (Non-CoT Logits) | 4.06 | 2.75 |
| **CME (Ours)** | **7.84** | **4.42** |

### Key Findings

- **FIRM-Edit-8B outperforms GPT-5 with only 8B parameters** (MAE 0.62 vs 0.67) — specialized training vastly outperforms scaling up general capabilities.
- **Reaches the performance of UniWorld (which uses 27K samples) with only 2,400 training samples (150 steps × 16 samples)** — precise reward signals significantly enhance sample efficiency.
- **Simple linear combinations of rewards cause catastrophic hacking**: under uniform 0.5+0.5 weights, GEdit score plummets from 7.54 to 1.06.
- **FIRM-Gen-8B shows more pronounced advantages on complex prompts**: +11.55 on UniGen Long vs. +9.83 for Qwen3-VL-8B — more complex scenarios demand more precise rewards.
- **Opposite patterns in editing vs. generation reward curves**: In editing, FIRM scores are lower than general VLMs (since VLMs ignore subtle changes and give falsely high scores). In generation, FIRM scores are higher than general VLMs (since VLMs hallucinate and give falsely low scores).

## Highlights & Insights

1. **"Difference-First" Evaluation Paradigm**: Exploiting the property of MLLMs where "problem-solving exceeds evaluation", this approach prompts the model to describe differences before scoring, simply and effectively bypassing the hallucination issues of direct evaluation.
2. **In-Depth Analysis of Reward Hacking**: Instead of only identifying "lazy editing" and "black silhouette" hacking patterns, the work fundamentally resolves them via multiplicative rewards (rather than additive) — setting the total reward to zero if Execution is zero, entirely blocking shortcuts.
3. **"Scaling Up is Not Always Better" Finding**: Qwen3-VL-32B degrades on ImgEdit, revealing the fundamental limitations of general VLMs in domain-specific evaluation.
4. **End-to-End Coverage of the Entire RL Stack**: Innovations span every stage, from data $\rightarrow$ model $\rightarrow$ benchmark $\rightarrow$ reward strategy $\rightarrow$ downstream training.

## Limitations & Future Work

- **Reward models initialized from Qwen3-VL-8B**: Bound by the capability ceiling of the base model, potential blind spots may persist for extremely complex spatial reasoning.
- **Limited scale of FIRM-Bench**: The human-annotated set of 807 samples may not fully cover all edge cases.
- **Manual tuning still required for CME/QMA weights**: The optimal values of $w_1, w_2$ may vary across different tasks and models.
- **Video editing/generation unexplored**: While the framework is theoretically generalizable to video, it has yet to be validated.
- **Weaker T2I base model**: Utilizing SD3.5-Medium as the baseline, validation was not performed on stronger models (such as FLUX or DALL-E 3 level).

## Related Work & Insights

- **vs. EditScore**: EditScore also trains an editing reward model but solely covers editing; FIRM covers both editing and generation while proposing a multiplicative reward strategy.
- **vs. Edit-R1**: Edit-R1 utilizes logits from general MLLMs as rewards, which leads to a noisy signal that plummets GEdit from 7.54 to 4.06. In contrast, FIRM's specialized reward model provides precise signals.
- **vs. T2I-R1**: T2I-R1 employs GRPO + CoT reasoning but does not focus on reward model quality; FIRM demonstrates that the accuracy of the reward model is the fundamental bottleneck in RL.
- **Insight**: "Problem deconstruction" outperforms "model scaling" — rather than pursuing larger general VLMs for evaluation, it is superior to design evaluation workflows that VLMs are naturally better at resolving (difference description $\rightarrow$ scoring, checklist $\rightarrow$ scoring).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — "Difference-first" pipeline + multiplicative reward strategy, offering deep insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers both editing and generation tasks, evaluates on multiple benchmarks, compares against multiple baselines, and includes exhaustive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, presenting a complete logical chain from problem identification to resolution.
- **Value**: ⭐⭐⭐⭐⭐ — FIRM-Edit-8B/Gen-8B can be directly used as critics for RL training, with code and models open-sourced.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Leveraging Verifier-Based Reinforcement Learning in Image Editing](../../CVPR2026/image_generation/leveraging_verifier-based_reinforcement_learning_in_image_editing.md)
- [\[ICLR 2026\] EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](../../ICLR2026/image_generation/editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)
- [\[CVPR 2026\] The Image as Its Own Reward: Reinforcement Learning with Adversarial Reward for Image Generation](../../CVPR2026/image_generation/the_image_as_its_own_reward_reinforcement_learning_with_adversarial_reward_for_i.md)
- [\[CVPR 2025\] Visual-ERM: Reward Modeling for Visual Equivalence](visual-erm_reward_modeling_for_visual_equivalence.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)

</div>

<!-- RELATED:END -->
