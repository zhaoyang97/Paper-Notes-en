---
title: >-
  [Paper Note] Probing RLVR Training Instability through the Lens of Objective-Level Hacking
description: >-
  [ICML 2026][Reinforcement Learning][RLVR] The authors propose the "objective-level hacking" framework, attributing the phenomenon where the training-inference discrepancy in MoE models widens during RLVR training to bias…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "GRPO"
  - "MoE"
  - "training-inference discrepancy"
  - "objective-level hacking"
date: 2026-05-08
content_hash: 910010bb894bb7d9
---

# Probing RLVR Training Instability through the Lens of Objective-Level Hacking

**Conference**: ICML 2026  
**arXiv**: [2602.01103](https://arxiv.org/abs/2602.01103)  
**Code**: None  
**Area**: LLM Alignment / RLHF / RL Training Stability / MoE  
**Keywords**: RLVR, GRPO, MoE, training-inference discrepancy, objective-level hacking

## TL;DR
The authors propose the "objective-level hacking" framework, attributing the phenomenon where the training-inference discrepancy in MoE models widens during RLVR training to biased pseudo-signals introduced by token-level weight distortions. They verify on a 30B MoE through four sets of experiments that "bias (not variance) is the culprit."

## Background & Motivation

**Background**: RLVR (Reinforcement Learning from Verifiable Rewards, represented by algorithms like GRPO/DAPO/GSPO) has become the core post-training paradigm behind reasoning models such as OpenAI o1 and DeepSeek-R1, demonstrating stronger generalization and long-term gains in math, code, and Agent tasks compared to SFT.

**Limitations of Prior Work**: Particularly under MoE architectures, RLVR training frequently suffers from sudden collapses—validation performance drops, token entropy collapses, and gradient norms become abnormal. A puzzling concomitant phenomenon is the steadily growing *training-inference discrepancy*: the output token probabilities for the same weights become increasingly inconsistent between vLLM inference and Megatron training, even with weight synchronization at every step.

**Key Challenge**: This was originally expected to be transient noise caused by "infrastructure numerical precision differences." Why does it grow monotonically with training and eventually trigger irreversible collapse? Existing patches (TIS, various clip variants, GSPO sequence-level clip) can mitigate this, but the underlying mechanism remains unexplained.

**Goal**: To answer two specific questions—(1) Why does the training-inference discrepancy accumulate and grow instead of remaining constant? (2) Which common techniques (initial discrepancy, token-level clipping, custom token weighting) inadvertently inject biased signals into the optimization objective?

**Key Insight**: The authors elevate the concept of "reward hacking" from the verifier to the *optimization objective level*. Any fine-tuning of token-level weights is equivalent to adding a $\Delta\mathcal J(\theta)$ term to the original GRPO objective. If this term is positively correlated with a pseudo-signal (such as $\rho_{i,t}^{-1}$), the optimization will push toward widening the discrepancy, forming a positive feedback loop.

**Core Idea**: Use a unified formula $\mathcal J_{\text{dist}}=\mathcal J + \Delta_{\text{dist}}\mathcal J$ to describe the implicit bias of various "token-level weight distortions" on the optimization objective. Through active injection experiments, the authors prove that *bias is the key*, while variance-based noise does not trigger collapse.

## Method

### Overall Architecture
The framework consists of two parts: (i) The theoretical side decomposes the GRPO/GSPO objectives into $\mathcal J(\theta)+\Delta\mathcal J(\theta)$, expressing both initial discrepancy and token-level clipping as a unified form of "token weight perturbation" $\sum_{i,t} X_{i,t}(\theta)(\phi_{i,t}-1)$. (ii) The experimental side runs DAPO-Math-17k on Qwen3-30B-A3B MoE using verl + vLLM + Megatron. Through four sets of experiments—(a) comparing GRPO with TIS correction, (b) varying clip ranges, (c) actively injecting weight distortion for low-probability tokens, and (d) injecting unbiased variance noise—the causal chain of "bias ⇒ discrepancy growth ⇒ collapse" is isolated.

### Key Designs

1.  **Unified Formalization of Objective-Level Hacking**:
    - **Function**: Maps all "seemingly harmless" token-level modifications to the implicit bias of the optimization objective.
    - **Mechanism**: Starting from the ideal objective $\mathcal J(\theta)=\mathbb E_{\text{train}}[\sum_{i,t} X_{i,t}(\theta)]$ (where $X_{i,t}=r_{i,t}\hat A_{i,t}/(G|o_i|)$), the authors derive a first-order approximation after shifting rollout from $\pi_{\text{train}}$ to $\pi_{\text{infer}}$, yielding $\Delta\mathcal J(\theta)\simeq \sum_{i,t}\text{Cov}_{\text{train}}(X_{i,t},\rho_{i,t}^{-1})$, where $\rho_{i,t}=\pi_{\text{train}}/\pi_{\text{infer}}$. Similarly, token-level clipping is equivalent to a multiplicative weight $\phi_{i,t}\in\{0,1\}$, resulting in $\Delta_{\text{clip}}\mathcal J=\mathbb E_{\text{train}}[\sum X_{i,t}(\phi_{i,t}-1)]$.
    - **Design Motivation**: Seemingly disparate "training accidents" (numerical errors, clipping, custom weighting) are actually doing the same thing—surreptitiously re-weighting certain tokens. A unified representation allows them to be examined using the same experimental logic.

2.  **Active Injection Experiments as Causal Probes**:
    - **Function**: Directly verifies "which types of perturbations trigger discrepancy growth" without relying on specific patches.
    - **Mechanism**: Based on a stable GSPO baseline (where sequence-level clipping itself does not induce growth), the authors artificially weight low-probability tokens using $\varphi_{i,t}=\delta$ if $\pi_{\text{train}}<\pi_{\text{low}}=0.1$, and $=1$ otherwise, scanning $\delta\in\{1.2,2,3\}$. This explicitly constructs a biased $\Delta_{\text{inj}}\mathcal J$. A control experiment is designed with unbiased variance injection $\xi_{i,t}\sim\mathcal N(1,\sigma^2)$, where the derivation shows $\Delta\mathcal J_{\text{var}}\simeq 0$ (since $\xi-1$ is independent of $Y_{i,t}$).
    - **Design Motivation**: Observing correlations cannot prove causality; active injection allows "switching" the pseudo-signal. If $\delta=1.2$ can instantly trigger explosion while variance noise does not, it proves the problem is *bias* rather than noise itself.

3.  **Objective-Level Signal Monitoring + Positive Feedback Loop Description**:
    - **Function**: Converts the abstract hacking concept into a monitorable scalar in training logs and explains why collapse is irreversible.
    - **Mechanism**: A proxy variable $J=\sum_{i,t}\hat A_{i,t}(\varphi_{i,t}-1)$ is defined and plotted in real-time. A Pearson correlation coefficient between $J$ and steps significantly greater than 0 indicates the pseudo-signal is being "continuously optimized." Meanwhile, statistics for $\rho_{i,t}$ across different probability intervals show that $\rho$ for low-probability tokens continuously deviates from 1; survival bias makes them harder to recover, further amplifying the hacking and forming a "discrepancy ⇄ hacking" positive feedback loop.
    - **Design Motivation**: Provides an early warning metric for industrial RLVR—training can be halted when $J$ rises monotonically, rather than waiting for validation performance to crash.

### Loss & Training
No new loss functions are introduced; instead, existing GRPO/GSPO objectives are reformulated as "weighted objectives." All experiments are conducted on 4 nodes × 8 A100 GPUs, with 128 problems × 16 responses per step, 4 optimization epochs, and response lengths of 8K. GRPO clip is set to 0.2 by default, and GSPO sequence-level clip is 3e-4/4e-4.

## Key Experimental Results

### Main Results

Discrepancy and validation behavior under different stabilization strategies (summarized from qualitative conclusions in Figures 2, 4, and 5):

| Configuration | Discrepancy Growth | Validation | Remarks |
| :--- | :--- | :--- | :--- |
| GRPO + token clip (vanilla) | Significant rise | Inverse drop | Standard RLVR, poor stability |
| + TIS correction | Significantly slowed | Improvement | Only modifies objective, not infra |
| Token clip strong ($\epsilon=0.2$) | Fastest | Earliest collapse | Strong clip = Strong bias |
| Token clip weak ($\epsilon=0.28$) | Slower | Better | Weak clip is paradoxically more stable |
| GSPO (sequence clip) | No growth | Stable | No token-level bias |
| GSPO + injected $\delta=1.2$ | Triggers growth | Degradation | Even 20% weighting triggers explosion |
| GSPO + variance injection $\xi\sim\mathcal N(1,\sigma^2)$ | No growth | Stable | Bias is the culprit, not variance |

### Ablation Study

| Investigated Question | Design | Key Observation |
| :--- | :--- | :--- |
| Clip strength vs. discrepancy | Right clip $\in\{0.2, 0.24, 0.28\}$ | Stronger clip leads to faster discrepancy growth |
| Bias vs. Variance | Biased vs. Unbiased token weighting | Unbiased variance does not trigger growth (Eq. (22)) |
| Decreasing low-prob token weight | Symmetrical perturbation to increase | Also triggers discrepancy growth, showing the root cause is "distortion," not "direction" |

### Key Findings
- Token-level clipping, which "feels like it should stabilize training," actually accelerates discrepancy growth in MoE because it is essentially a form of token-level weight distortion, equivalent to injecting pseudo-signals.
- Discrepancy growth is a *positive feedback* process: hacking causes $\rho$ for low-probability tokens to deviate from 1, and this deviation amplifies the effective strength of the hacking. This is why models cannot be saved by swapping batches or rolling back checkpoints once collapse starts.
- Sequence-level algorithms (GSPO) are stable on MoE not because they have "looser clipping," but because they structurally avoid token-level weight distortion. This provides a specific criterion for future MoE-specific RL algorithm design: *never inject biased weights into the objective that depend on token probabilities.*

## Highlights & Insights
- Successfully translates the concept of "reward hacking" to "objective hacking," providing the first mechanical explanation of the training-inference discrepancy "infrastructure bug" as a product of "algorithmic-level biased objectives."
- The active injection experiment design is elegant: using GSPO as a stable base to explicitly toggle $\delta$ and $\sigma$ constitutes a clean bidirectional causal experiment.
- Offers a *monitorable* early signal for industrial RLVR, $J=\sum \hat A_{i,t}(\varphi_{i,t}-1)$, allowing practitioners to preemptively stop training rather than relying solely on post-hoc validation drops.

## Limitations & Future Work
- Experiments were completed only on the Qwen3-30B-A3B MoE model; the strength of the conclusions for dense models or other MoE routing strategies is unknown.
- The framework explains "what triggers collapse" but does not yet provide an "automatic de-biasing" solution. A potential direction is to incorporate both importance sampling correction and token weight distortion fixes into a unified objective.
- The numerical precision of A100 is lower than H100, which the authors admit amplified the initial discrepancy; hacking trigger thresholds may differ on higher-precision hardware.

## Related Work & Insights
- **vs DAPO / Dr.GRPO and other GRPO variants**: While they modify clipping, advantage, or length normalization based on empirical experience, this paper provides a unified formula $\mathcal J=\mathcal J+\Delta\mathcal J$ to explain *why* these modifications work.
- **vs GSPO (Zheng 2025)**: GSPO discovered sequence-level clipping was more stable at an engineering level; this paper provides a mechanical explanation via objective-level hacking, moving its advantages from "empirical finding" to "theoretical necessity."
- **vs TIS (Yao 2025)**: TIS directly applies importance sampling correction to the objective. This paper positions TIS as a "targeted treatment at the $\Delta\mathcal J$ layer" and shows that TIS cannot fully eliminate discrepancy in MoE settings.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes the concept of objective-level hacking and a unified formula, explaining the MoE RLVR collapse mechanism for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Conducted four sets of experiments on 30B MoE: control groups, intensity scans, active injection, and bias/variance decoupling.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations and intuitive descriptions of the positive feedback loop.
- Value: ⭐⭐⭐⭐ Provides specific guidelines for RLVR algorithm design and monitorable engineering signals.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](../../ICLR2026/reinforcement_learning/exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](../../ICLR2026/reinforcement_learning/how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)

</div>

<!-- RELATED:END -->
