---
title: >-
  [Paper Note] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models
description: >-
  [ICML 2026][LLM Reasoning][Looped Language Model] Addressing the characteristic of Looped Language Models (LoopLM) that iteratively process latent representations $T_{\max}$ times before outputting each token…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Looped Language Model"
  - "Implicit Reasoning"
  - "Trajectory-level Credit Assignment"
  - "GRPO"
  - "Process Reward"
date: 2026-05-08
content_hash: 980cb0fc5886ba70
---

# Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.10520](https://arxiv.org/abs/2602.10520)  
**Code**: https://github.com/jonwill8/RLTT.git  
**Area**: LLM Reasoning / Reinforcement Learning / Looped Transformer  
**Keywords**: Looped Language Model, Implicit Reasoning, Trajectory-level Credit Assignment, GRPO, Process Reward

## TL;DR
Addressing the characteristic of Looped Language Models (LoopLM) that iteratively process latent representations $T_{\max}$ times before outputting each token, this paper proposes RLTT: replacing the "reward only the terminal loop" strategy in GRPO with "weighting the next-token distribution of every loop by $\omega_t$." Without introducing external verifiers and with near-zero additional computational overhead, RLTT improves the average accuracy of Ouro-2.6B on MATH/AIME/BeyondAIME by +10.9%, while reducing training time by 10% and spontaneously shortening response lengths.

## Background & Motivation

**Background**: The mainstream "long-chain reasoning" route relies on explicit chain-of-thought (CoT) tokens, using RL with verifiable rewards (e.g., GRPO) to backpropagate gradients based on final answer correctness. An alternative route is **implicit reasoning**: LoopLMs like Ouro and Huginn execute the same transformer weights $T_{\max}$ times per output token to "think" in latent space, achieving performance comparable to explicit CoT with fewer parameters.

**Limitations of Prior Work**: Direct application of GRPO to LoopLM yields minimal gains—the Ouro paper admitted that RL offered almost no improvement over SFT. LSRL attempted to provide process rewards for intermediate states using GPT-4.1 nano, but required decoding hidden states into text and calling external APIs, leading to extremely high engineering overhead and costs.

**Key Challenge**: The GRPO policy gradient $\nabla_\theta \log P_\theta^{(T_{\max})}(y_j\mid x,y_{<j}) \hat{A}_i$ is a function only of the final loop's distribution, **implicitly assuming a single decision step per token**. However, a LoopLM actually undergoes a full trajectory $h_j^{(1)}\to\cdots\to h_j^{(T_{\max})}$. Reward signals must penetrate all intermediate loops to backpropagate to early representations, creating a "credit assignment bottleneck."

**Goal**: Design a policy gradient framework that can directly replace GRPO, requires no external verifier, and applies RL signals to the implicit distributions of all loops simultaneously.

**Key Insight**: The LoopLM architecture offers a neglected convenience—**the hidden state $h_j^{(t)}$ of each loop can be projected onto the vocabulary via the shared language modeling head $g(\cdot)$**, producing $T_{\max}$ "latent thought distributions" for free. Since these distributions are already calculated during the forward pass, incorporating them into the policy gradient adds almost no extra computation.

**Core Idea**: Use a weight sequence $\{\omega_t\}_{t=1}^{T_{\max}}$ ($\sum \omega_t = 1$) to calculate the weighted sum of $\log P_\theta^{(t)}$ for each loop, replacing the terminal-only $\log P_\theta^{(T_{\max})}$ in GRPO—thereby assigning credit directly to the "thought trajectory" rather than just the "thought endpoint."

## Method

### Overall Architecture
RLTT is a **plug-and-play replacement for GRPO**, requiring only that the LoopLM produces a complete next-token distribution in every loop. The pipeline is isomorphic to GRPO: sample $g$ rollouts $\{y_i\}$ for prompt $x$, and calculate normalized advantage $\hat{A}_i = (r_i - \mu)/\sigma$ using binary correctness rewards $r_i \in \{0,1\}$. The only difference is the policy gradient form: RLTT incorporates the log-prob of each token's every loop into the gradient weighted by $\omega_t$, while retaining GRPO-style KL regularization $\beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$ (calculated only for the final loop to save memory).

The final loss (Eq. (5)–(7)) is:

$$J_{\text{RLTT}}(\theta) = -\mathbb{E}\Big[\frac{1}{g|y_i|}\sum_{i,j,t} \omega_t \log P_\theta^{(t)}(y_{i,j}\mid x,y_{<j})\hat{A}_i\Big] + \beta D_{\mathrm{KL}}$$

The engineering cost is not compute, but **VRAM**: log-probs for all $T_{\max}$ loops must be retained to construct the weighted objective. Consequently, `ppo_max_token_len_per_gpu` is reduced from 16384 (GRPO) to 8192, compensated by mini-steps.

### Key Designs

1.  **Trajectory-level Policy Gradient (RLTT PG)**:
    - **Function**: Replaces the terminal-only $\nabla_\theta \log P$ calculation in GRPO with a weighted sum across all $T_{\max}$ loops.
    - **Mechanism**: Replaces the single-loop gradient with $\sum_{t=1}^{T_{\max}} \omega_t \nabla_\theta \log P_\theta^{(t)}(y_j\mid x,y_{<j})\hat{A}_i$, making the gradient a direct function of all loops' implicit distributions. This ensures: (i) reward signals no longer rely solely on backpropagation through the terminal loop; (ii) the entire trajectory $P_\theta^{(1)}\to\cdots\to P_\theta^{(T_{\max})}$ is pushed toward high-advantage predictions, directly encouraging "early convergence."
    - **Design Motivation**: Terminal-only credit assumes single-step decisions, mismatching the multi-step implicit reasoning of LoopLM; trajectory-level credit treats the effective credit assignment horizon as 1 step, making each update more "information-dense."

2.  **Three Loop Weighting Strategies $\{\omega_t\}$**:
    - **Function**: Determines the contribution of each loop to the total credit.
    - **Mechanism**: (a) **Exit PDF**—$\omega_t = p_{\text{exit}}(t\mid x)$, reusing the learned early-exit head probabilities from Ouro as loop reliability; (b) **Progressive**—$\omega_t = t^\alpha / \sum_s s^\alpha$, weighting later loops higher; (c) **Uniform**—$\omega_t = 1/T_{\max}$, forcing the model to form and maintain the correct distribution as early as possible.
    - **Design Motivation**: Performance differences across strategies were minimal (Appendix A.3), suggesting that gains come from "exposing the full trajectory" itself rather than any specific scheduling.

3.  **Experimental Protocol Strictly Aligned with GRPO**:
    - **Function**: Rules out "parameter tuning luck" as the cause for RLTT's superiority.
    - **Mechanism**: Rollout budget, optimizer, reward function, advantage normalization, training steps (140), and KL coefficients are all aligned. RLTT's `ppo_max_token_len_per_gpu` is halved due to VRAM constraints. During evaluation, GRPO is intentionally given a larger token budget (3072 vs. 2048 for RLTT) to rule out advantages from shorter responses.

## Key Experimental Results

### Main Results

| Model | MATH-500 | AIME24 | AIME26 | BeyondAIME | GSM8K | Math Avg | Non-Math Avg |
|---|---|---|---|---|---|---|---|
| Ouro-1.4B-Thinking | 73.2 | 16.7 | 13.3 | 4.0 | 90.7 | 39.6 | 58.7 |
| + GRPO | 77.4 | 16.7 | 16.7 | 6.0 | 91.7 | 41.7 | 59.5 |
| **+ RLTT** | **81.2** | **26.7** | **20.0** | **12.0** | 90.3 | **46.0 (+5.8)** | **64.8 (+5.3)** |
| Ouro-2.6B-Thinking | 75.6 | 13.3 | 6.67 | 5.0 | 93.6 | 38.8 | 64.5 |
| + GRPO | 79.0 | 16.7 | 16.7 | 6.0 | 93.9 | 42.5 | 65.2 |
| **+ RLTT** | **86.0** | **33.3** | **26.7** | **16.0** | **94.0** | **51.2 (+10.9)** | **71.8 (+6.6)** |
| Qwen3-4B + GRPO | 62.2 | 3.33 | 3.33 | 0.0 | 89.8 | 31.7 | 58.7 |

Ouro-1.4B+RLTT (46.0%) surpasses Qwen3-4B+GRPO (31.7%) in math average. 2.6B+RLTT shows gains of +16.6/+10.0/+10.0% over GRPO on AIME24/26/BeyondAIME respectively. Paired t-tests show RLTT is significantly better than GRPO on 8/9 benchmarks for 2.6B (p<0.05).

### Ablation Study

| Metric | GRPO | RLTT | Note |
|---|---|---|---|
| Training Time (140 steps) | 54.42 hrs | 49.05 hrs | RLTT spontaneously shortens responses → -10% time |
| Min/Step | 23.3 ± 8.31 | 21.1 ± 9.87 | Same as above |
| Response Length Trend | Stable | Continual Decrease | Emergent behavior; no brevity incentive |
| Terminal Entropy Drop | Slow | Steeper & Sustained | Pass@k rules out entropy collapse |
| Loop Weighting Policy | n/a | Diff < 1% | Gains come from trajectory exposure |
| GPQA (zero-shot) | 19.7 (2.6B) | 38.4 (2.6B) | Nearly doubled multi-hop reasoning |
| 1-2 loop Eval | Massive degradation | Superior to GRPO | Early loop reasoning is significantly enhanced |

### Key Findings
- **Spontaneous response shortening without explicit rewards**: Rewards only consider correctness, but RLTT distributes credit across every implicit distribution, forcing "early convergence" in latent space. This eliminates the need for long-tail token-level corrections.
- **GPQA doubling as strong transfer evidence**: Despite training only on MATH, the massive improvement on GPQA (fact-based multi-hop) indicates RLTT makes reasoning trajectories converge more robustly within token budgets.
- **Theoretical Support (Appendix A.10 Theorem A.5)**: Under assumptions of uncertainty reduction through loop refinement, RLTT is proven to require fewer tokens on average than GRPO to reach a correct solution.
- **Improved GSNR**: On the hardest tasks (AIME24/BeyondAIME), the Gradient Signal-to-Noise Ratio (GSNR) of RLTT is statistically superior to GRPO, providing direct evidence for the "richer gradient signal" hypothesis.

## Highlights & Insights
- **Architecture-Aligned RL**: Identifies the fundamental mismatch between GRPO's "single-step decision assumption" and LoopLM's multi-step implicit reasoning. The zero-cost fix of reusing existing per-loop logits serves as a paradigm for aligning RL algorithms with model architecture semantics.
- **Process Rewards without External Verifiers**: Unlike LSRL, which decodes states for GPT-4.1 nano to score, RLTT uses the model's own next-token distributions as process signals.
- **Byproducts as Diagnostic Signals**: Improvements in response length, entropy, and training time—none of which were explicitly optimized—suggest that RLTT changes *how* the model thinks rather than just *what* it learns.

## Limitations & Future Work
- VRAM usage doubles due to retaining per-loop log-probs, limiting per-GPU token packing.
- Method is specific to LoopLM and not applicable to standard non-recurrent LLMs.
- Training/Inference uses fixed loop depth, sacrificing Ouro's native adaptive early-exit capability.
- **Future Work**: Use gradient checkpointing or FP8 to store log-probs; integrate RLTT with adaptive halting; generalize to other implicit reasoning architectures like Huginn or Coconut.

## Related Work & Insights
- **vs GRPO**: The direct baseline; RLTT transitions from "terminal-only" to "trajectory-weighted" credit, resulting in comprehensive wins.
- **vs LSRL (Ren, 2025)**: LSRL requires expensive GPT-4.1 nano decoding for every loop; RLTT is self-contained and achieved higher gains (+10.9% on Math vs. LSRL's +4.27% on GSM8K).
- **vs Ouro (Zhu et al., 2025b)**: Ouro introduced the architecture but struggled to make RL work; RLTT directly solves this "RL failure in LoopLM" problem.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple weighting idea, but perfectly aligned with "ready-to-use" LoopLM logits for a high gain-to-cost ratio.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across scales, benchmarks, t-tests, GSNR, and theoretical proofs.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the core idea could have been introduced earlier.
- Value: ⭐⭐⭐⭐⭐ First work to make RL definitively work on LoopLM without external verifiers; likely to become a strong baseline for latent reasoning + RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[ICML 2026\] GRPO is Secretly a Process Reward Model](grpo_is_secretly_a_process_reward_model.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](reward_modeling_from_natural_language_human_feedback.md)
- [\[ICML 2026\] DecepChain: Inducing Deceptive Reasoning in Large Language Models](decepchain_inducing_deceptive_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](../../ACL2026/llm_reasoning/large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)

</div>

<!-- RELATED:END -->
