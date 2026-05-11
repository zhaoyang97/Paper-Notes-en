---
title: >-
  [Paper Note] Training Large Language Models To Reason In Parallel With Global Forking Tokens
description: >-
  [ICLR 2026][parallel reasoning] This paper proposes Set Supervised Fine-Tuning (SSFT), which aligns global forking tokens with diverse reasoning trajectories via bipartite matching, enabling LLMs to globally steer distinct reasoning modes from a single control token. SSFT substantially outperforms standard SFT and GRPO on mathematical reasoning and code generation tasks.
tags:
  - ICLR 2026
  - parallel reasoning
  - global forking tokens
  - set supervised fine-tuning
  - bipartite matching
  - test-time compute
date: 2026-05-08
content_hash: c1f80cc84cca0a6b
---

# Training Large Language Models To Reason In Parallel With Global Forking Tokens

**Conference**: ICLR 2026
**arXiv**: [2510.05132](https://arxiv.org/abs/2510.05132)
**Code**: [Sheng-J/SSFT](https://github.com/Sheng-J/SSFT)
**Area**: Code Intelligence
**Keywords**: parallel reasoning, global forking tokens, set supervised fine-tuning, bipartite matching, test-time compute

## TL;DR

This paper proposes Set Supervised Fine-Tuning (SSFT), which aligns global forking tokens with diverse reasoning trajectories via bipartite matching, enabling LLMs to globally steer distinct reasoning modes from a single control token. SSFT substantially outperforms standard SFT and GRPO on mathematical reasoning and code generation tasks.

## Background & Motivation

- LLMs improve reasoning by scaling test-time compute (generating more tokens), but **sequential scaling** suffers from "overthinking"—performance degrades beyond a certain sequence length.
- **Parallel sampling** (e.g., self-consistency, Best-of-N) offers an orthogonal scaling dimension, but relies on the model generating **diverse and correct** solutions.
- Research shows that only a few **forking tokens** in Chain-of-Thought reasoning lead to divergent reasoning paths; as problems become harder and generations longer, the probability of sampling these critical tokens drops significantly.
- Common diversity-enhancing techniques (e.g., temperature scaling) face a **diversity–accuracy trade-off**: theoretical work shows that simply raising temperature cannot guarantee greater diversity unless the model is explicitly trained for coverage.

## Core Problem

How can LLMs be trained on diverse reasoning trajectories such that a set of **global control tokens** placed at the beginning of generation steer the model into distinct reasoning modes—achieving high diversity and high accuracy in parallel reasoning without relying on sampling intermediate forking tokens?

## Method

### 1. Problem Formulation: Parallel Reasoning as Set Prediction

- A set of **global forking tokens** $\boldsymbol{g} = \{g^{(i)}\}_{i=1}^{N}$ is defined, instantiated as `<think 1>`, `<think 2>`, ..., `<think N>` tags.
- Given a problem $\mathbf{x}$ and $M$ distinct correct reasoning trajectories $\mathbf{R} = \{\mathbf{r}^{(j)}\}_{j=1}^{M}$, the goal is for each $g^{(i)}$ to uniquely trigger a distinct reasoning trajectory.
- Parallel reasoning is framed as a **set-of-next-token-prediction** problem satisfying two requirements:
    - **Permutation invariance**: the loss is independent of the ordering of $\mathbf{R}$ and $\boldsymbol{g}$.
    - **Non-shared forking tokens**: distinct reasoning trajectories must not be conditioned on the same $g^{(i)}$.

### 2. SSFT: Set SFT via Optimal Bipartite Matching

Each training step proceeds in two stages:

**Stage 1: Optimal Matching.** A cost matrix is constructed where each entry is the NTP loss of trajectory $\mathbf{r}^{(j)}$ conditioned on $g^{(i)}$ (length-normalized, stop-gradient). The **Hungarian algorithm** is applied to find the minimum-cost bipartite matching $\hat{\boldsymbol{\sigma}}$.

**Stage 2: Optimization.** Backpropagation is performed only over the $M$ matched sequences, minimizing the Hungarian loss:

$$\mathcal{L}_{\text{Hungarian}}(\boldsymbol{\theta}) = -\mathbb{E}_{\mathbf{x}, \mathbf{R}}[\sum_{j=1}^{M}\sum_{t=1}^{T_\mathbf{r}} \log \pi_\theta(r_t^{(j)} | \mathbf{x}, g^{(\hat{\sigma}(j))}, \mathbf{r}_{<t}^{(j)})]$$

**Computational optimization**: matching cost computation uses only the first $L$ tokens ($L \approx \lfloor \text{max\_seq\_len} / (MN) \rfloor$), allowing all matching costs to be computed in a single forward pass with negligible additional training overhead.

### 3. GFPO: Global Forking Policy Optimization

- Following SSFT, a small number of RL steps apply policy gradients exclusively to the output distribution of the global forking tokens.
- Since global forking tokens always appear at a fixed position in the generated sequence, the implementation requires only a few lines of Python slicing on top of existing GRPO code.
- Full generations are used solely to compute the advantage function for $g^{(i)}$ and do not participate in backpropagation.

### 4. Inference Protocol

- **Cons@k**: each of the $N$ distinct `<think i>` tags is used to prompt a separate generation, followed by majority voting.
- **Pass@1**: the GFPO model automatically samples the optimal $g^{(i)}$; alternatively, the $g^{(i^*)}$ that covers the most distinct trajectories during training is selected (based on a graph heuristic from Equation 4).

### Key Design Choices

- $N > M$ global forking tokens are retained (e.g., $N=6, M=4$); the extra tokens maximize discrimination between similar trajectories.
- The matching assignment $\hat{\sigma}$ varies dynamically per problem—the same teacher model may be matched to different forking tokens across different problems.
- Unlike training independent sub-models, SSFT allows positive transfer across different trajectories.

## Key Experimental Results

### Experimental Setup

- **Base model**: Qwen2.5-32B-Instruct
- **Training data**: 1,000 problems from s1k, each with 4 reasoning trajectories distilled from R1, Gemini Flash, Claude Opus 4.0/4.1, and GPT-OSS-120B
- **Evaluation**: AIME24/25, MATH-500, GPQA-Diamond, LiveCodeBench (OOD)

### Main Results (Pass@1)

| Model | AIME24 | AIME25 | MATH-500 | LCB (OOD) |
|---|---|---|---|---|
| SFT-mixed-distill-32B | 58.23 | 51.96 | 88.49 | 32.34 |
| SSFT-32B (random σ) | 61.77 | 55.10 | 89.95 | 35.33 |
| **SSFT-32B** | **64.06** | **58.13** | **90.02** | **38.92** |
| **SSFT-32B-GFPO** | **64.22** | **58.80** | 89.90 | **42.10** |

- SSFT improves over SFT trained on the same data by **+5.83 / +6.17** on AIME24/25, respectively.
- Cons@32 reaches **86.67%** on AIME25, a 10-point gain over SFT's 76.67%.
- On OOD code generation (LCB), SSFT-GFPO achieves 42.10%, a gain of +9.76.

### Diversity Analysis

- Different `<think i>` tags trigger **clearly distinct distributions of reasoning length and accuracy** (Figure 4).
- In models trained with random matching, different `<think i>` tags show no discernible differences (Figure 5).
- During training, only a small subset of matching assignments consistently receives weight, indicating that the model learns stable associations between forking tokens and reasoning modes.

### Robustness

- Results are consistent on code generation data (code1k): SSFT-32B-code achieves 52.07% on LCB vs. 47.13% for SFT.
- Consistent gains are observed on public datasets (OpenR1-93k) and smaller models (Qwen2.5-Math-7B).
- Gains are also observed on Qwen3-4B-Base and Llama3.1-8B-Instruct.

## Highlights & Insights

- **Elegant problem formulation**: Casting parallel reasoning as set prediction and applying bipartite matching—inspired by DETR—to language modeling for the first time yields a theoretically principled framework.
- **High practicality**: Global forking tokens are placed at fixed positions at the sequence start, enabling diverse reasoning guidance at inference without complex search. GFPO requires only a few lines of code.
- **Interpretable matching visualization**: The evolution of matching assignments during training clearly demonstrates the automatic learning of associations between forking tokens and reasoning modes.
- **Minimal computational overhead**: Using stop-gradient and only the first $L$ tokens for matching cost computation introduces almost no additional training cost.

## Limitations & Future Work

- The current setting of $N=6, M=4$ is relatively small in scale; the effectiveness and computational cost of larger bipartite graphs remain unexplored.
- Diverse reasoning trajectories are obtained via multi-teacher distillation, creating a dependency on the quality and diversity of teacher models.
- GFPO applies policy gradients only to forking tokens; whether this approach can be extended to additional controllable positions is not discussed.
- Evaluation focuses primarily on mathematics and code; performance on open-domain reasoning tasks (e.g., commonsense reasoning, multi-hop QA) is unknown.

## Related Work & Insights

| Method | Characteristics | Advantage of Ours |
|---|---|---|
| Temperature scaling | Increases diversity by adjusting temperature | Cannot guarantee coverage; high temperature degrades accuracy |
| Self-consistency / Best-of-N | Aggregates parallel samples | Does not explicitly train for diversity; relies on intermediate forking tokens |
| Multiverse (Yang et al., 2025b) | Converts sequential CoT to parallel CoT | No set loss; cannot prevent mode collapse |
| Concurrent work (Wen et al., 2025) | Multi-trajectory distillation with random tag assignment | Random assignment cannot learn stable associations between forking tokens and trajectories |
| DETR (Carion et al., 2020) | Global set loss for object detection | This work is the first to extend it to autoregressive language modeling |

### Broader Connections

- The paradigm of combining **set prediction with language modeling** has broad generalization potential: applications include multi-answer generation, multi-style writing, and multi-strategy search.
- The global forking token concept can be combined with **Mixture of Experts**—routing distinct reasoning modes to different experts.
- The trick of computing matching costs using only the first $L$ tokens suggests that the early tokens of a reasoning trajectory are sufficient to distinguish different reasoning strategies, an observation applicable to trajectory pruning and rapid evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (introduces set prediction to language modeling; global forking token concept is original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (multiple benchmarks, model scales, data sources, and extensive ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear formulations, rich visualizations, rigorous experimental logic)
- Value: ⭐⭐⭐⭐⭐ (practically useful and theoretically elegant; offers important guidance for parallel reasoning training)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [\[ICLR 2026\] Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[ICLR 2026\] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task](mathfimer_enhancing_mathematical_reasoning_by_expanding_reasoning_steps_through_.md)
- [\[ICLR 2026\] A Problem-Oriented Perspective and Anchor Verification for Code Optimization](a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)

</div>

<!-- RELATED:END -->
