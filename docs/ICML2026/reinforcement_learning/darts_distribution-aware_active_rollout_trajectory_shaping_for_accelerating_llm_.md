---
title: >-
  [Paper Note] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][GRPO] DARTS redefines the long-tail bottleneck of LLM RL rollout from "scheduling bypass" to "active distribution shaping." Through intra-prompt redundant sampling + dual-end length sa…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "GRPO"
  - "rollout acceleration"
  - "long-tail distribution"
  - "active shaping"
  - "dual-end sampling"
  - "adaptive redundancy allocation"
date: 2026-05-08
content_hash: 95898c19007a268d
---

# DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.30859](https://arxiv.org/abs/2605.30859)  
**Code**: Abstract notes "Source code available at: URL" (placeholder, to be confirmed after open-sourcing)  
**Area**: Reinforcement Learning / LLM Inference-Training / System Optimization  
**Keywords**: GRPO, rollout acceleration, long-tail distribution, active shaping, dual-end sampling, adaptive redundancy allocation

## TL;DR
DARTS redefines the long-tail bottleneck of LLM RL rollout from "scheduling bypass" to "active distribution shaping." Through intra-prompt redundant sampling + dual-end length sampling + variance-driven redundancy budget allocation, it explicitly compresses and tightens the model's rollout length distribution, achieving up to 1.77x speedup compared to VeRL on Qwen 3B–32B models without losing downstream accuracy.

## Background & Motivation

**Background**: Currently, LLMs using RL algorithms like GRPO / DAPO for "inference-time scaling" have become standard practice—each prompt is sampled $M$ times by the policy $\pi_\theta$ to generate responses $\{o_i^j\}_{j=1}^M$, calculate group-normalized advantage $A(o_i^j) = (r_i^j - \mu(\mathcal{Y}_i))/\sigma(\mathcal{Y}_i)$, and backpropagate gradients. The entire pipeline is split into rollout and training phases, with the rollout phase accounting for over 70% of the total training time, forming the primary bottleneck. The root of this slowness is the extreme long-tail distribution of rollout trajectory lengths—a few prompts trigger ultra-long trajectories that can be 5–10x longer than the median and over 20x longer than short responses. In synchronous on-policy systems, "the longest response stalls the entire batch," causing severe GPU idling.

**Limitations of Prior Work**: Existing mitigation strategies—such as Tail Batching in RollPacker and Partial Rollout used by Kimi/Moonshot—are essentially "prompt-level tail scheduling": over-sampling $N' > N$ prompts and waiting for $N$ prompts to complete while deferring or truncating the remaining long-tail prompts. These methods only adjust "scheduling timing" without touching the distribution itself; furthermore, they identify inter-prompt tails (variance between different prompts) while ignoring significant long tails within the same prompt. Another asynchronous route can completely overlap rollout and training but breaks on-policy semantics, leading to training instability and accuracy degradation.

**Key Challenge**: Scheduling routes cannot eliminate fundamental waste—the long-tail trajectories themselves are generated, and the token computation cost is already paid. The authors observe an overlooked fact: rollout lengths within a single prompt also exhibit severe long tails (max/mean > 10×), and a large portion of these tails consists of "ineffective verbosity"—meaningless chatter or error loops that do not contribute to accuracy.

**Goal**: Upgrade from "managing tail latency" to "eliminating tail distribution," specifically: (1) concentrate the model's rollout length distribution toward the short end without losing accuracy; (2) simultaneously preserve necessary long-chain reasoning; (3) make the distribution shaping mechanism prompt-aware and adaptive; (4) utilize system optimizations to amortize the extra sampling overhead introduced by distribution shaping.

**Key Insight**: The authors decompose long-tail trajectories into two modes based on "length–reward correlation"—Pattern I (Ineffective Verbose Tail, $\mathbb{E}[l|r>0] \le \mathbb{E}[l|r<0]$, where correct responses are shorter) and Pattern II (Necessary Deep Tail, $\mathbb{E}[l|r>0] > \mathbb{E}[l|r<0]$, where correct responses require long chains). Ideal distribution shaping should compress the former while preserving the latter.

**Core Idea**: Utilize a single "dual-end length sampling + adaptive redundancy budget" mechanism to automatically achieve both "suppressing verbosity (Pattern I)" and "preserving depth (Pattern II)" through the shift of the GRPO group baseline, without explicit classification.

## Method

### Overall Architecture
DARTS acts as a rollout-phase plugin on top of VeRL, consisting of three components: (1) Distribution-Aware Trajectory Sampling—performing intra-prompt redundant sampling to obtain a candidate pool $\mathcal{T}_i$ (size $M_i' \ge M$) for each prompt $q_i$, then selecting the training set $\mathcal{Y}_i$ via dual-end length sampling; (2) Adaptive Redundancy Allocation—using historical length variance $\tilde\sigma_L^2(q_i)$ as a proxy for long-tail severity to distribute the total redundancy budget $M_{\mathrm{total}}$ across prompts based on variance; (3) System-Level Optimization—variance-guided tail pruning + proactive early stopping + token-level streaming. The entire process does not modify the reward function or introduce extra hyperparameters (except for three stable system parameters $M_{\mathrm{up}}, M_{\mathrm{low}}, \lambda$).

### Key Designs

1. **Distribution-Aware Trajectory Sampling (Core Algorithm Layer)**:
    - **Function**: Explicitly pulls the model's rollout length distribution toward the short end while preserving a few long chains.
    - **Mechanism**: First performs intra-prompt redundant sampling, generating $M_i' \ge M$ candidates $\mathcal{T}_i = \{o_i^1,\ldots,o_i^{M_i'}\}$ for each prompt $q_i$. Then, it performs dual-end length sampling by sorting by length and selecting the top-$K$ shortest and top-$L$ longest ($K + L = M$ and $K \gg L$, with $L=1$ in main experiments), explicitly excluding invalid trajectories that hit the system length limit. Theoretical analysis (Propositions 1/2) shows this automatically works through the GRPO group baseline: For Pattern I prompts, $\mathcal{Y}_{\text{short}}$ captures high-reward samples, making $\mu(\mathcal{Y}_i^{\text{dual}}) \approx \bar r_{\text{short}}$ high, thereby reducing the advantage $r(o_i^{\text{long}}) - \mu(\mathcal{Y}_i^{\text{dual}})$ of verbose positive cases and suppressing verbosity. For Pattern II prompts, $\mathcal{Y}_{\text{short}}$ mostly contains incorrect short answers, resulting in a low $\mu$, which instead amplifies the positive advantage of necessary long chains, encouraging deep reasoning.
    - **Design Motivation**: Avoids explicit prompt classification. A single sampling rule adaptive to both modes via the group baseline avoids classifiers and additional learning problems. The dual-end combination ensures the training group contains both "short and correct" templates and "long and correct" exploration paths, preventing single-ended bias.

2. **Variance-Based Adaptive Redundancy Allocation (Budget Allocation Layer)**:
    - **Function**: Allocates more sampling redundancy to prompts with the most severe long tails under a total system budget constraint, while giving minimal budget to low-variance prompts.
    - **Mechanism**: The authors observe that response length variance $\sigma_L^2(q_i) = \sigma^2(\{l_i^1,\ldots,l_i^{M_i'}\})$ is strongly positively correlated with "long-tail severity + model uncertainty," thus using its moving average $\tilde\sigma_L^2(q_i)$ as a proxy. First, utility $U(\bar M) = E(\bar M) - \lambda T(\bar M)$ is defined, where effectiveness $E$ is estimated using dataset-level $\tilde\sigma_L^2$ and overhead $T$ is estimated using a rollout cost model based on PTL (per-token latency): $T_{\text{rollout}} = \sum_{m=1}^{M'}(l_{[m]}-l_{[m-1]})\cdot \mathrm{PTL}(d_{\mathrm{TP}}, M'-m+1)$. The optimal total budget $M_{\mathrm{total}}$ is found by solving $\partial U/\partial \bar M = 0$. Then, the convex optimization $\min \sum_i \mathrm{Norm}(\tilde\sigma_L(q_i))/M_i'$ s.t. $\sum_i M_i' = M_{\mathrm{total}}$, $M_{\mathrm{low}} \le M_i' \le M_{\mathrm{up}}$ (default $M_{\mathrm{low}}=M, M_{\mathrm{up}}=2M$) is solved via a greedy algorithm due to the positive second derivative. This reflects "diminishing marginal utility"—adding one candidate to a high-variance prompt is worth more than adding one to a low-variance prompt.
    - **Design Motivation**: Uniform allocation wastes redundancy budget on low-uncertainty prompts (where distributions are already short and tight), whereas concentrating allocation on high-variance prompts maximizes shaping effects while controlling total cost. The $M_{\mathrm{up}}=2M$ soft upper bound prevents a single prompt from consuming all redundancy, while $M_{\mathrm{low}}=M$ ensures basic GRPO training.

3. **Variance-Guided Tail Pruning + Token-Level Streaming (System Optimization Layer)**:
    - **Function**: Addresses extra efficiency optimization for extremely complex prompts (where the budget hits $M_{\mathrm{up}}$) and breaks granularity bottlenecks of "sample-level streaming."
    - **Mechanism**: (a) When $M_i' = M_{\mathrm{up}}$ (approx. ~5% of most expensive prompts), the system automatically switches from dual-end sampling to shortest-only sampling. The insight is that for these prompts, the entire distribution shifted right, so the "shortest" is already long/deep enough. Once switched to shortest-only, proactive early stopping can be triggered: terminate the remaining $M_i' - M$ long-tail trajectories still decoding once $M$ short trajectories are collected, saving expensive tail token computation. (b) Token-level streaming changes "sending a trajectory to training only upon completion" to "chunking and sending tokens to training every time a certain count is reached," allowing the training engine to start the forward pass of prefixes while long trajectory suffixes are still being generated.
    - **Design Motivation**: Dual-end sampling is actually a negative optimization for the 5% extreme prompts (must wait for the longest trajectory to get top-$L$), so using variance signals to downgrade these to shortest-only is a simple fix. Token-level streaming solves the specific bottleneck of "sample-level overlap failing" when each GPU only runs a few trajectories under large-scale data parallelism.

### Loss & Training
The training objective remains unchanged. The underlying RL algorithm uses GRPO (group size $M=8$) + DAPO optimization (clip-higher, token-level loss, overlong reward shaping). All prompt-level scheduling and distribution shaping occur in the rollout phase and are transparent to the training phase. Rewards follow the original task (e.g., verifiers for math problems). $\lambda$ controls the aggressiveness of redundancy allocation (large $\lambda$ is conservative, small $\lambda$ is aggressive), with $M_{\mathrm{up}}=2M$ and a default dual-end ratio of $L:K = 1:7$.

## Key Experimental Results

Experiments were mainly conducted on 8 nodes (each with 8×H20 96GB + NVLink + 1.6Tbps IB) using Qwen2.5-3B/7B-Math/14B/32B and Qwen3-30B-A3B (MoE), with datasets DAPO-MATH (7B–32B) and MATH-lighteval (3B), comparing against VeRL (SOTA open-source RL framework) and Tail Batching (prompt-level scheduling representative).

### Main Results: End-to-End Throughput Speedup (vs VeRL)
| Model | VeRL | Tail Batching | DARTS | Relative to VeRL | Relative to Tail Batching |
|------|------|----------------|-------|-----------|-------------------|
| Qwen2.5-3B | 1.00× | 1.07× | 1.29× | 1.29× | 1.21× |
| Qwen2.5-Math-7B | 1.00× | ~1.3× | 1.45× | 1.45× | ~1.12× |
| Qwen2.5-14B | 1.00× | ~1.4× | 1.63× | 1.63× | ~1.16× |
| Qwen2.5-32B / Qwen3-30B-A3B | 1.00× | 1.62× (max) | 1.77× (max) | 1.77× | 1.43× |
| BBH Zero-shot (Qwen2.5-32B) | 78.1 | — | 84.7 | +6.6 | — |
| BBH Zero-shot (Qwen2.5-Math-7B) | 56.6 | — | 58.8 | +2.2 | — |

### Ablation Study: Component Contribution Decomposition (Qwen2.5-14B, 32×H20)
| Configuration | Speedup | Description |
|------|---------|------|
| VeRL baseline | 1.00× | Standard GRPO + Synchronous on-policy |
| + Token-Level Streaming | 1.09× | Individual contribution ~9%, granularity optimization |
| + Distribution-Aware Sampling | 1.40× | Dual-end sampling (uniform budget), main source of gain |
| + Adaptive Allocation (DARTS) | 1.63× | Added variance adaptive allocation, $+0.23×$ |

### Key Findings
- The gains from distribution shaping primarily come from the sampling layer (1.40×), with adaptive allocation adding another 0.23×, while token streaming only contributes 1.09×—indicating that the throughput bottleneck is indeed the "distribution itself" rather than pipeline granularity.
- DARTS achieves higher speedups on larger models (1.29× on 3B → 1.77× on 32B) because larger models have deeper reasoning chains and more prominent long tails, providing more optimization space for distribution shaping.
- Training convergence curves and average scores across 5 math benchmarks (MATH500, GSM8K, AIME2024, AIME2025, Olympiad) are basically on par with VeRL. BBH zero-shot improved by up to 6.6 points (Qwen2.5-32B from 78.1 to 84.7), proving that distribution shaping does not damage reasoning ability but enhances generalization.
- Dual-end ratio sensitivity: $L:K = 1:7$ provides a 1.45× speedup (7B), $2:6$ drops slightly to 1.43×, and $4:4$ falls to 1.25×—the larger the long-tail share, the weaker the shaping, but $1:7$ is sufficiently stable. $L=1$ was fixed for main experiments.
- Control experiments simply increasing length penalties resulted in a 2–7% drop in downstream accuracy, proving that shaping via "group baseline modification" rather than "reward modification" is key—long chains still receive strong positive gradients as long as they are correct.
- Cross-domain preliminary experiments: Multi-modal reasoning (Qwen2.5-VL-3B / Geo3K) 1.20×, code generation (Phi-3-mini-3B / Eurus-2-RL-Data) 1.15× speedup. Gains are limited on small models where long tails are not significant, but the method is transferable.

## Highlights & Insights
- Using the "GRPO group baseline to automatically distinguish Pattern I/II" is the most ingenious design—dual-end sampling leverages the baseline to drive opposite gradient adjustments (Proposition 1 suppresses verbosity / Proposition 2 amplifies depth) without explicit classification. This idea of "leveraging existing normalization" can be transferred to any group-relative advantage algorithm (DAPO, ReMax, RLOO).
- Treating "system metrics (length variance $\sigma_L^2$)" as a bridge for "algorithmic metrics (long-tail severity + uncertainty)" is pragmatic—it requires no extra uncertainty estimation models; rollout statistics are sufficient and cost almost nothing to accumulate online. This is a good paradigm for system-algorithm co-design.
- Incorporating PTL profiling into the RL framework for cost-aware allocation is rare but necessary—most RL acceleration works only count "trajectory numbers / token numbers," ignoring the non-linear generation latency at different batch sizes, which leads to uniform budget allocation being severely suboptimal under large-scale parallelism.
- The distinction between intra-prompt and inter-prompt long tails is the biggest conceptual contribution—previous scheduling works assumed tails came from "varying prompt difficulty," whereas DARTS points out "10x differences within the same prompt," optimizing based on the model's intrinsic rollout behavior.

## Limitations & Future Work
- DARTS strongly depends on the GRPO group baseline mechanism—its core theory (Propositions 1/2) does not hold for single-sample estimation algorithms like PPO (where advantages aren't normalized by group mean), and no PPO adaptation is provided.
- Ambiguity effect: When a prompt contains both ineffective verbosity and necessary depth (intermediate state), dual-end sampling might suppress and preserve different parts of long chains simultaneously, but which specific ones are kept is determined by length ranking rather than reward signals—potentially "mis-killing" high-quality medium-long chains.
- The $M_{\mathrm{up}}=2M$ hard upper bound limits the shaping space—for extremely complex prompts (e.g., Olympiad proofs), 2M candidates might still be insufficient to cover correct long chains; increasing $M_{\mathrm{up}}$ triggers more tail pruning, degrading to single-ended sampling.
- Speedup significantly decays on small models and non-math domains, showing sensitivity to "long-tail severity." For tasks where rollouts are already compact (Short-answer QA, classification), DARTS has little utility.
- Training stability was only verified within ~500 steps; whether distribution shaping causes the model to fall into a "short-chain comfort zone" and lose exploration capability during ultra-long training (tens of thousands of steps) remains unanswered.

## Related Work & Insights
- **vs RollPacker / Tail Batching**: Both perform inter-prompt over-sampling, but RollPacker only schedules (deferring tails), while DARTS directly modifies the distribution; DARTS also introduces the intra-prompt perspective. DARTS provides 1.07–1.43× more speedup than Tail Batching.
- **vs Partial Rollout (Kimi/Moonshot)**: Partial Rollout truncates tails and resumes in the next step, introducing slight off-policy behavior; DARTS maintains on-policy semantics while compressing the distribution, which is theoretically cleaner.
- **vs Asynchronous RL Frameworks (AReaL, AsyncFlow)**: Asynchronous routes hide tail latency via overlapping but sacrifice on-policy semantics; DARTS chooses the opposite—preserving synchronous semantics while eliminating long tails from the distribution layer.
- **vs Length Penalty Routes**: Directly increasing length penalty is simple but drops accuracy by 2–7%; DARTS performs "soft" shaping via group baseline shifts without touching rewards or accuracy, serving as a more sophisticated equivalent.
- **Insight**: Treating rollouts as a "shapeable distribution" rather than "passive overhead" is a perspective that can be extended to all sampling-dependent RL—episode length distributions in video/robotics RL or tool-call step distributions in agentic RL could all use similar "sampling redundancy + group baseline shift" for active shaping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EchoRL: Reinforcement Learning via Rollout Echoing](echorl_reinforcement_learning_via_rollout_echoing.md)
- [\[ICML 2026\] LASER: Learning Active Sensing for Continuum Field Reconstruction](laser_learning_active_sensing_for_continuum_field_reconstruction.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] QuRL: Efficient Reinforcement Learning with Quantized Rollout](../../ICLR2026/reinforcement_learning/qurl_efficient_reinforcement_learning_with_quantized_rollout.md)
- [\[ICML 2026\] Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](safety_generalization_under_distribution_shift_in_safe_reinforcement_learning_a_.md)

</div>

<!-- RELATED:END -->
