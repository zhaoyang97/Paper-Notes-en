---
title: >-
  [Paper Note] Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling
description: >-
  [NeurIPS 2025][Multimodal VLM][Reinforcement Learning] To address the problem of "unfaithful reasoning trajectories induced by outcome-reward RL training in multimodal multiple-choice tasks…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Reinforcement Learning"
  - "Self-Consistency Sampling"
  - "Reasoning Faithfulness"
  - "Multimodal Reasoning"
  - "Outcome Reward"
date: 2026-05-08
content_hash: 2dd8e40b93d04774
---

# Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2511.10648](https://arxiv.org/abs/2511.10648)  
**Code**: [GitHub](https://github.com/GenuineWWD/SCS)  
**Area**: Multimodal VLM
**Keywords**: Reinforcement Learning, Self-Consistency Sampling, Reasoning Faithfulness, Multimodal Reasoning, Outcome Reward

## TL;DR

To address the problem of "unfaithful reasoning trajectories induced by outcome-reward RL training in multimodal multiple-choice tasks," this paper proposes Self-Consistency Sampling (SCS), which obtains consistency rewards via truncation-resampling and visual perturbation to penalize spurious reasoning. When combined with RLOO, SCS achieves an average improvement of 7.7 percentage points across six benchmarks.

## Background & Motivation

- **Background**: Outcome-reward RL methods (e.g., GRPO, RLOO, REINFORCE++) are the dominant paradigm for enhancing reasoning capabilities in MLLMs.
- **Limitations of Prior Work**: In multiple-choice questions (the primary format of multimodal reasoning benchmarks), a critical yet overlooked issue exists: **unfaithful trajectories gaming the reward** — the model guesses the correct option following an erroneous reasoning chain, yet receives the same full reward as a genuinely correct reasoning trace.
- **Key Challenge**: The paper reveals the severity of this issue through exploratory experiments:

  **Insufficient gains from multiple-choice format**: On Geometry3K, multiple-choice training yields only a 5.6% improvement, 6.4 points lower than the 12.0% gained with open-ended QA.

  **Truncation-continuation divergence**: Continuing from truncated reasoning trajectories at different positions frequently produces different final answer choices from the same prefix, indicating that unfaithful reasoning trajectories are pervasive.

  **Qualitative analysis**: The model frequently generates incorrect reasoning yet coincidentally selects the correct answer.

- **Goal**: Process Reward Models (PRMs) can mitigate this issue but are computationally expensive. SCS aims to identify and down-weight unreliable reasoning trajectories through self-consistency checks, without introducing any additional reward model.

## Method

### Overall Architecture

SCS models the reasoning process as a tree structure based on three key assumptions:

1. **Uniqueness of correct trajectories**: Exactly one correct reasoning trajectory exists in the reasoning tree.
2. **Leaf-node/option alignment**: Every trajectory ultimately points to one answer option.
3. **Relationship between correct/incorrect trajectories and options**: A correct trajectory necessarily leads to the correct option; an incorrect trajectory may either guess the correct option or select a wrong one.

Under these assumptions, when using only accuracy reward $r = r_{\text{acc}}$, unfaithful reasoning still has a non-zero probability of receiving full reward:

$$P(\tau^- \mid y^+) = \frac{P(y^+, \tau^- \mid x)}{P(y^+, \tau^- \mid x) + P(y^+, \tau^+ \mid x)}$$

### Key Designs

#### Consistency Reward

SCS introduces a consistency reward $r_{\text{con}}$ to penalize inconsistent reasoning patterns. Given a reasoning trajectory $\tau$, the model resamples $N$ times to obtain an answer set $\mathcal{A}$, and the consistency reward is defined as:

$$r_{\text{con}} = \frac{1}{N}(N - |\mathcal{A}|) \cdot c$$

where $c$ is a scaling coefficient. Intuitively, if the reasoning is correct, repeated sampling should consistently point to the same answer ($|\mathcal{A}|=1$, maximum reward); if the reasoning is incorrect, answers will diverge ($|\mathcal{A}|$ increases, reward decreases).

Consistency reward for a correct trajectory: $r_{\text{con}}^+ = \frac{1}{N}(N-1) \cdot c$ (maximum).
Consistency reward for an incorrect trajectory: $r_{\text{con}}^- = \frac{1}{N}(N-|\mathcal{A}^-|) \cdot c$, where $r_{\text{con}}^+ > r_{\text{con}}^-$ since $\mathbb{E}(|\mathcal{A}^-|) > 1$.

Total reward: $r = r_{\text{for}} + r_{\text{acc}} + r_{\text{con}}$ (format reward + accuracy reward + consistency reward).

#### Truncation-Resampling (TR)

The initial reasoning trajectory $\tau$ is truncated at ratio $k$ to yield an incomplete trajectory $\tau^<$, which serves as a prefix for $m$ resampling continuations, each generating a new answer $a_t$. All answers are collected to form $\mathcal{A}$.

**Core Idea**: If the reasoning process is faithful, continuations from the same prefix should yield consistent answers; if the reasoning is spurious, continuations will diverge.

#### Visual Perturbation (VP)

Slight Gaussian noise is added to the input image at each resampling step:

$$\tilde{\mathbf{x}}_i = \mathbf{x} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2), \quad \sigma_i \sim \mathcal{U}(\sigma_{\min}, \sigma_{\max})$$

The perturbation magnitude $\sigma_i$ is independently sampled from a uniform distribution at each step, exposing the model to diverse visual variants. This compels the policy to reason based on perturbed visual evidence; correct reasoning should remain robust to small perturbations.

### Loss & Training

SCS is compatible with multiple RL algorithms: RLOO, GRPO, REINFORCE++, and REINFORCE++-baseline. Training configuration:

- Base model: Qwen2.5-VL-7B-Instruct
- Training data: M³CoT (7.8k) + Geometry3K (2.1k) + ScienceQA (6.2k, filtered) — only image-containing multiple-choice questions retained
- Batch size: 128, with 16 trajectories sampled per prompt
- Learning rate: 1e-6, temperature: 1.0
- Truncation ratio $k$=0.8 (RLOO/REINFORCE++) or 0.4 (GRPO); resampling count $m$=4 (RLOO) or 8 (GRPO)
- 8 × A800 GPUs, approximately 24 hours of training

## Key Experimental Results

### Main Results

**Table 1: Effect of SCS with Different RL Algorithms (Qwen2.5-VL-7B-Instruct)**

| Method | SCS | Overall | M3CoT | MMMU | SciQA | WeMath | MathVerse | MathVision |
|--------|-----|---------|-------|------|-------|--------|-----------|------------|
| Baseline (pretrained) | - | 54.9 | 65.5 | 45.7 | 73.7 | 62.5 | 57.7 | 24.1 |
| SFT | - | 58.6 | 78.7 | 52.6 | 51.0 | 90.7 | 49.4 | 29.3 |
| RLOO | ✕ | 57.8 | 67.6 | 51.5 | 53.9 | 86.4 | 56.8 | 30.4 |
| RLOO | ✓ | **65.5 (+7.7)** | 75.7 | 59.1 | 68.8 | 88.1 | 67.1 | 34.0 |
| GRPO | ✕ | 63.6 | 72.6 | 57.2 | 66.6 | 88.3 | 64.2 | 32.8 |
| GRPO | ✓ | **64.5 (+0.9)** | 73.9 | 58.0 | 66.4 | 88.7 | 67.0 | 33.1 |
| REINFORCE++ | ✕ | 60.9 | 66.8 | 54.9 | 64.8 | 84.3 | 60.9 | 33.4 |
| REINFORCE++ | ✓ | **62.9 (+2.0)** | 65.7 | 54.6 | 76.1 | 85.4 | 61.6 | 34.0 |

**Table 2: Cross-Model Generalization (RLOO + SCS)**

| Model | w/o SCS | w/ SCS | Gain |
|-------|---------|--------|------|
| Qwen2.5-VL-7B-Instruct | 57.8 | 65.5 | +7.7 |
| Qwen2.5-VL-3B-Instruct | 54.7 | 57.9 | +3.2 |
| InternVL3-8B | 61.7 | 63.3 | +1.6 |

### Ablation Study

**Component Effectiveness (based on RLOO)**:

| TR | VP | Overall | Gain |
|----|----|---------|------|
| ✕ | ✕ | 57.8 | - |
| ✓ | ✕ | 63.0 | +5.2 |
| ✕ | ✓ | 62.8 | +5.0 |
| ✓ | ✓ | 65.5 | +7.7 |

Each component contributes approximately 5 points individually; their combination recovers the full 7.7-point gain, demonstrating complementarity.

**Hyperparameter Sensitivity**:

- Truncation ratio $k$: performance increases from 0.1 to 0.8, peaks at 0.8, then declines. Too small a ratio provides insufficient consistency signal; too large a ratio leaves inadequate exploration space.
- Resampling count $m$: performance increases from 2 to 4, peaks at $m=4$, then declines. Excessive resampling introduces randomness and increases computational overhead.
- Performance fluctuation within the tested hyperparameter ranges remains within 4 points, indicating that the method is relatively robust.

**Quantitative Analysis of Reasoning Reliability**:
100 correctly answered samples were randomly drawn from each benchmark and manually examined for reasoning faithfulness. After SCS training, the proportion of unfaithful reasoning decreased by approximately 15% (human evaluation: 25.0 → 21.2; o3-mini evaluation: 22.0 → 19.0).

### Key Findings

1. RLOO under vanilla RL even underperforms SFT (57.8 vs. 58.6), but with SCS jumps to 65.5 — SCS yields the largest gain for RLOO (+7.7).
2. The advantage of vanilla RL over SFT is marginal (approximately 2–3 points), indicating that standard outcome rewards are inefficient for multiple-choice tasks.
3. The consistency reward in SCS effectively functions as a "poor man's process reward," requiring no additional reward model.
4. The two components contribute nearly symmetrically (TR +5.2 vs. VP +5.0), yet exhibit synergistic effects when combined.

## Highlights & Insights

- The paper precisely identifies a widely overlooked problem: unfaithful reasoning trajectories under the multiple-choice format cause degeneration of outcome-reward RL.
- SCS is an elegant and lightweight design: it requires no additional reward model, does not alter the RL algorithm's structure, and only appends a consistency reward term.
- The theoretical derivation is rigorous: the mathematical formulation from tree-structure modeling to consistency reward follows a logically sound progression.
- The truncation-resampling mechanism is ingenious in that it exploits the "causal coherence" of reasoning trajectories — a faithful reasoning prefix should consistently lead to the correct conclusion.

## Limitations & Future Work

- Validation is limited to the multiple-choice format; effectiveness in open-ended QA settings remains unknown.
- Training data scale is relatively small (~16k samples); performance at larger scales awaits verification.
- The mathematical assumption underlying the consistency reward (uniqueness of the correct trajectory) may not hold for open-ended reasoning.
- Visual perturbation employs simple Gaussian noise; more sophisticated data augmentation strategies (geometric transformations, occlusion) may yield further improvements.
- SCS produces a smaller gain for GRPO (+0.9), possibly because GRPO's group-relative advantage estimation already partially suppresses unfaithful trajectories.

## Related Work & Insights

- Self-Consistency (Wang et al., 2022) improves reasoning consistency by taking the majority vote over multiple samples; SCS integrates the consistency idea into RL training rewards.
- Process Reward Models (PRMs) provide step-level feedback but are computationally expensive; SCS serves as a low-cost alternative.
- RLOO benefits the most, likely because its leave-one-out baseline already exhibits low variance, and SCS's consistency signal further stabilizes training.
- The truncation-resampling mechanism is generalizable to other scenarios requiring verification of reasoning faithfulness.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The problem is precisely identified, and the consistency reward design demonstrates strong originality.
- **Technical Depth**: ⭐⭐⭐⭐ — The theoretical derivation is complete, and the two-component design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 4 RL algorithms, 3 models, 6 benchmarks, and multiple ablation groups.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play, negligible computational overhead, well-suited for multiple-choice reasoning training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training](first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](../../AAAI2026/multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)

</div>

<!-- RELATED:END -->
