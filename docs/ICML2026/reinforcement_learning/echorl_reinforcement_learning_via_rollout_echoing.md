---
title: >-
  [Paper Note] EchoRL: Reinforcement Learning via Rollout Echoing
description: >-
  [ICML 2026][Reinforcement Learning][RLVR] This paper identifies that GRPO-like methods in the late stages of RLVR training suffer from "advantage degeneration…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "Advantage Degeneration"
  - "EchoClip"
  - "Step-Level Entropy"
  - "GRPO"
date: 2026-05-08
content_hash: d28df30472ad1b6e
---

# EchoRL: Reinforcement Learning via Rollout Echoing

**Conference**: ICML 2026  
**arXiv**: [2605.31228](https://arxiv.org/abs/2605.31228)  
**Code**: Mentioned in the paper, but the repository link is not explicitly provided  
**Area**: Reinforcement Learning / LLM Reasoning / RLVR / GRPO  
**Keywords**: RLVR, Advantage Degeneration, EchoClip, Step-Level Entropy, GRPO  

## TL;DR
This paper identifies that GRPO-like methods in the late stages of RLVR training suffer from "advantage degeneration," where gradients vanish because an entire group of rollouts succeeds, causing the relative advantage to drop to zero. The authors propose EchoRL: selecting the "most challenging yet successful" prefix, termed EchoClip, based on **step-level entropy peaks** from verified-success rollouts. This is added to the loss as an auxiliary SFT term, consistently yielding up to 5.6%/5.0% ID/OOD gains across 4 RLVR frameworks, 5 backbones, and 10 benchmarks.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) is currently the de facto mainstream for LLM reasoning post-training. GRPO has become the most widely used framework—powering models like DeepSeek-R1 and Qwen-Math—by removing the critic and replacing the value function estimation with group-relative advantage.

**Limitations of Prior Work**: As models become stronger, more prompts result in $G$ rollouts that are all verified-success. In such cases, rewards within the group are identical, leading to a standard deviation $\sigma_r=0$, which makes the normalized advantage $\hat{A}_i=(r_i-\mu_r)/\sigma_r=0$ for all $i$. The GRPO policy gradient $\nabla_\theta J \propto \mathbb{E}[\sum\nabla\log\pi_\theta\cdot w_{i,t}\cdot \hat{A}_i]$ is thus nullified by being multiplied by zero, **consuming significant compute without providing any learning signal**. This phenomenon is termed "advantage degeneration" by the authors.

**Key Challenge**: Rewards only reflect the final correctness, remaining completely blind to "how the answer was reached." A rollout using brute-force algebra and another using a "logarithmic differentiation trick" both receive a reward of 1, but the latter clearly contains a **more valuable reasoning path**. Current normalization methods discard this as noise. Existing solutions either follow the "rejection sampling/dynamic budget" route (e.g., DAPO, Reinforce-Rej/Ada)—discarding degenerate groups at the cost of data efficiency—or the "external golden trajectory supervision" route (e.g., LUFFY, UFT, SRFT), which introduces dependencies on expensive expert models.

**Goal**: To extract "buried usable signals" from the model’s own verified-success rollouts without relying on external experts or discarding rollouts, thereby maintaining non-zero gradients even during the advantage degeneration phase.

**Key Insight**: The authors conducted a diagnostic analysis comparing the token entropy distribution of expert golden trajectories versus the current policy’s verified rollouts. They found that **expert trajectories have higher overall entropy**, and within specific trajectories, **information-dense steps often coincide with sharp entropy peaks**. This translates the problem of "which step is important" into "which step has the maximum step-level entropy."

**Core Idea**: Use step-level entropy as a proxy to identify the "most difficult but successful" prefix in verified rollouts as an EchoClip. This is added to the RL objective as an auxiliary NLL loss. This SFT-style dense supervision does not depend on the advantage, naturally bypassing gradient vanishing.

## Method

### Overall Architecture

EchoRL is a plug-and-play module that can be integrated with any RLVR algorithm (GRPO, DAPO, LUFFY, UFT). For each prompt $q$, the standard process still involves sampling $G$ rollouts, calculating group-relative advantages, and performing PPO-style updates. EchoRL inserts two additional steps: (1) **EchoClip Mining**—selecting the most critical prefix $o_{echo}$ from the verified-success subset $V=\{o\mid r(o)=1\}$ based on step-level entropy; (2) **EchoRL Update**—adding the negative log-likelihood of this segment as an auxiliary supervision $\mathcal{J}_{EchoRL}$ to the main loss. The overall objective becomes $\mathcal{J}(\theta)=\mathcal{J}_{RLVR}(\theta)+\lambda\mathcal{J}_{EchoRL}(\theta)$, where $\lambda=0.001$ adjusts the magnitude. The mechanism's elegance lies in the fact that when $\hat{A}_i=0$ and $\nabla\mathcal{J}_{RLVR}\to 0$ due to group-wide success, $\nabla\mathcal{J}_{EchoRL}$ remains non-zero, preventing training from stalling.

### Key Designs

1. **Step-level Entropy as a Proxy for "Usable Learning Signal"**:
    - **Function**: Transforms the vague question of "which reasoning step is valuable" into a computable scalar.
    - **Mechanism**: Rollouts are split into a sequence of reasoning steps $(s_1,\dots,s_M)$ using natural delimiters (e.g., `\n`). Step-level entropy is defined as $\bar{H}(s_j)=\frac{1}{|s_j|}\sum_{x\in s_j}H_\theta(x\mid q,o_{<x})$, the average predictive entropy of all tokens in the step. Single-token entropy is avoided because short-term fluctuations (e.g., punctuation) make it too noisy.
    - **Design Motivation**: The authors established the "high step-level entropy = critical step" link through two points of evidence. Evidence 1: The overall entropy of external golden trajectories is significantly higher than self-generated rollouts, suggesting expert "hard steps" correspond to the model's "uncertainty zones." Evidence 2: **Ablative deletion** on the OpenR1-Math 45k subset showed that deleting steps from "high entropy to low entropy" caused accuracy to collapse rapidly, while deleting in reverse or randomly required removing many more steps for the same drop. This proves high-entropy steps carry most of the reasoning value.

2. **EchoClip Mining: Selecting a Unique Prefix from Verified Rollouts**:
    - **Function**: Locates a single "representative" critical path as the supervision source given a set of verified rollouts $V$.
    - **Mechanism**: Collect all steps from $V$ into a pool $\text{Steps}(V)$ and find the step with the maximum entropy $s^*=\arg\max_{s\in\text{Steps}(V)}\bar{H}(s)$. If $o^*\in V$ is the parent rollout containing $s^*$, the EchoClip is defined as $o_{echo}=\text{Prefix}(o^*, s^*)$, which is the prefix truncated at the end of $s^*$.
    - **Design Motivation**: Choosing the maximum entropy step rather than top-k ensures **precision**—targeting the single most difficult but successful moment is more effective than a vague multi-step average. Truncating at $s^*$ rather than using the full rollout avoids supervising redundant subsequent chains and preserves the model's freedom to generate its own sequels. Using a "prefix" makes this a standard prefix-LM training problem with minimal implementation cost.

3. **EchoRL Update: Wrapping EchoClip as a Prefix-NLL Auxiliary Loss**:
    - **Function**: Converts the extracted EchoClip into a stable dense gradient signal injected into the RL pipeline.
    - **Mechanism**: The auxiliary objective is defined as $\mathcal{J}_{EchoRL}(\theta)=-\frac{1}{L}\sum_{t=1}^{L}\log\pi_\theta((o_{echo})_t\mid q,(o_{echo})_{<t})$ where $L=|o_{echo}|$. This is weighted with the RLVR term: $\mathcal{J}(\theta)=\mathcal{J}_{RLVR}+\lambda\mathcal{J}_{EchoRL}$ with $\lambda=0.001$.
    - **Design Motivation**: This format intentionally mimics an SFT-style token-level NLL to bypass the advantage mechanism. Since SFT loss gradients do not depend on group variance, they remain effective even when $\sigma_r=0$. A small $\lambda$ ensures RL remains the primary driver while EchoRL serves as a "booster" during degeneration, preventing training from collapsing into pure SFT. Unlike LUFFY/UFT, which introduce full external golden trajectories, EchoRL uses the model's own prefix, requiring zero extra inference cost or expert dependency.

### Loss & Training

General objective: $\mathcal{J}(\theta)=\mathcal{J}_{RLVR}(\theta)+\lambda\mathcal{J}_{EchoRL}(\theta)$, $\lambda=0.001$. Rollout batch size 128, update batch size 64, 8 rollouts per prompt, temperature 0.6. Implemented on verl (text) and EasyR1 (multimodal). Base models include Qwen2.5-1.5B/7B/Math-7B, LLaMA-3.1-8B, and Qwen2.5-VL. Training sets: OpenR1-Math 45k (text) and Geometry3K (multimodal).

## Key Experimental Results

### Main Results: EchoRL on Qwen2.5-Math-7B (Excerpt)

| Method | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | ID Avg | ARC-c | GPQA | MMLU-Pro | OOD Avg |
|------|--------|--------|-----|----------|---------|----------|--------|-------|------|----------|---------|
| Qwen2.5-Math-7B | 11.4 | 4.9 | 31.3 | 43.6 | 7.4 | 15.6 | 19.0 | 18.2 | 11.1 | 16.9 | 15.4 |
| Qwen2.5-Math-7B-Instruct | 12.9 | 10.2 | 48.5 | 80.4 | 32.7 | 41.0 | 37.6 | 70.3 | 24.7 | 34.1 | 43.0 |
| SFT | 22.2 | 22.3 | 52.8 | 82.6 | 40.8 | 43.7 | 44.1 | 75.2 | 24.7 | 42.7 | 47.5 |
| GRPO | 25.8 | 16.4 | 61.2 | 80.4 | 39.7 | 43.7 | 44.5 | — | — | — | — |
| LUFFY | 29.4 | 23.1 | 65.6 | 87.6 | 37.5 | 57.2 | 50.1 | 80.5 | 39.9 | 53.0 | 57.8 |
| **LUFFY + EchoRL** | **33.4** | **25.7** | **67.5** | **88.9** | 39.0 | 55.1 | **51.9** | **83.6** | **45.3** | **54.1** | **61.0** |
| UFT | 24.8 | 18.1 | 60.5 | 82.6 | 40.1 | 47.8 | 45.7 | 82.2 | 38.9 | 49.6 | 56.9 |
| **UFT + EchoRL** | 27.0 | 21.3 | 62.0 | 84.4 | 40.8 | 49.6 | 47.6 | 82.7 | 43.4 | 53.5 | 59.9 |

LUFFY + EchoRL achieved a +1.8% Gain in ID and +3.2% Gain in OOD. Specifically, GPQA improved from 39.9 to 45.3 (+5.4), demonstrating maximum benefit for true OOD reasoning.

### Ablation Study: Step-level Entropy (Verifying "High Entropy = Critical")

| Deletion Strategy | Accuracy (10% steps removed) | Accuracy (30% steps removed) | Description |
|----------|------------------------|------------------------|------|
| High-entropy | Significant drop | Near random | High-entropy steps are key to reasoning |
| Low-entropy | Nearly unchanged | Still acceptable | Low-entropy steps are boilerplate |
| Random | Intermediate | Intermediate | Corroborates the above |

### Key Findings
- EchoRL provides positive Gains across **all** four base RLVR methods (GRPO/DAPO/LUFFY/UFT). Even for LUFFY/UFT, which already use external experts, EchoRL adds another 1–3% boost, indicating that "mining self high-entropy steps" and "using external trajectories" are complementary signals.
- OOD gains (up to +5.04%) are more pronounced than ID gains (up to +5.61%). Improvements of 3–5 points on ARC-c/GPQA/MMLU-Pro suggest EchoRL strengthens the **stability of general reasoning steps** rather than rote memorization.
- Computational overhead is negligible: Entropy for EchoClip is calculated using logits already available from the rollout phase. The loss adds one forward pass, keeping training time comparable to or even slightly lower than original algorithms (as it avoids DAPO's extra rollout overhead for rejection sampling).

## Highlights & Insights
- **"Entropy peak as critical step" is a universal and elegant proxy**: While others use verifiers, PRMs, or external experts to locate key steps, this work uses the model’s own predictive entropy—zero external dependency, near-free computation, and high interpretability (high entropy = model hesitation at a fork in the road). This can be extended to PRM training, best-of-N selection, or search-based decoding.
- **Using SFT to bypass advantage disappearance is a cost-effective trick**: Advantage degeneration is essentially a zero-denominator problem. while others attempt to restore the denominator (rejection sampling/dynamic budgets), EchoRL **bypasses it** by injecting gradients via an NLL term. This design pattern—using a secondary loss to maintain training momentum when the primary loss fails—can be generalized to any group-relative algorithm.
- **Selecting a prefix rather than the full trajectory is an underrated detail**: Unlike LUFFY, which forces the full expert trajectory into the loss, EchoRL only supervises the prefix, allowing the model to **generate freely afterwards**. This preserves RL exploration while providing guidance at critical junctures.

## Limitations & Future Work
- Step-level entropy reliability assumes the model is "reasonably calibrated." In early training, high entropy might be noise rather than a difficulty; the authors do not discuss if EchoRL should be delayed (warmup).
- The "max-entropy step from the whole group" selection reduces to "taking the only one" when there is only one verified-success rollout, potentially losing diversity. Top-k EchoClips or soft selection with temperature could be considered.
- $\lambda=0.001$ was fixed across all experiments. The optimal $\lambda$ across tasks and scales hasn't been swept; smaller models (e.g., 1.5B) might benefit from a higher prefix-NLL weight.
- Step splitting via natural delimiters works for well-formatted Chain-of-Thought but might fail on dense formulas or code blocks. Some details are in Appendix D, but task-specific tokenizers might be needed for production.

## Related Work & Insights
- **vs DAPO / Reinforce-Rej / Reinforce-Ada**: These discard degenerate prompts or increase rollout budgets, sacrificing data efficiency for stability. EchoRL reuses the same rollouts, actually improving data efficiency.
- **vs LUFFY / UFT / SRFT / SEELE / RelIFT**: These rely on expensive strong model demos. EchoRL is self-sufficient, and results show LUFFY + EchoRL still improves, suggesting "self-verified critical steps" and "external expert steps" are distinct, additive signals.
- **vs Process Reward Model (PRM)**: PRMs require training a step-level reward model with high annotation costs. EchoRL uses entropy as a zero-cost proxy for PRM scores.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "entropy-peak mining + prefix-NLL advantage bypass" is fresh in the RLVR context, and its engineering simplicity is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive across 5 backbones, 4 RLVR baselines, 10 benchmarks, ID+OOD evaluations, entropy ablations, and overhead analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly explained with specific examples (quartic polynomials); method diagrams and formulas are well-integrated.
- Value: ⭐⭐⭐⭐ Plug-and-play, zero extra compute, effective across frameworks. Highly valuable for industrial implementation (e.g., integration into verl/OpenRLHF).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](darts_distribution-aware_active_rollout_trajectory_shaping_for_accelerating_llm_.md)
- [\[ICLR 2026\] QuRL: Efficient Reinforcement Learning with Quantized Rollout](../../ICLR2026/reinforcement_learning/qurl_efficient_reinforcement_learning_with_quantized_rollout.md)
- [\[ICML 2026\] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)
- [\[ICML 2026\] InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](inftythink_effective_and_efficient_infinite-horizon_reasoning_via_reinforcement_.md)
- [\[ICML 2026\] Coupled Variational Reinforcement Learning for Language Model General Reasoning](coupled_variational_reinforcement_learning_for_language_model_general_reasoning.md)

</div>

<!-- RELATED:END -->
