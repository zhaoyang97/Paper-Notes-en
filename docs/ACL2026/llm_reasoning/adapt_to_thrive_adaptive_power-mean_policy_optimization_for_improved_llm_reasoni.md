---
title: >-
  [Paper Note] Adapt to Thrive! Adaptive Power-Mean Policy Optimization for Improved LLM Reasoning
description: >-
  [ACL 2026][LLM Reasoning][RLVR] This paper proposes APMPO, which unifies GRPO (arithmetic mean) and GMPO (geometric mean) objectives using a "Power-Mean" controlled by the current mean reward. Combined with an adaptive c…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "RLVR"
  - "GRPO"
  - "Power-Mean Objective"
  - "Adaptive Clipping"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: efcbb13303fda52a
---

# Adapt to Thrive! Adaptive Power-Mean Policy Optimization for Improved LLM Reasoning

**Conference**: ACL 2026  
**arXiv**: [2605.04066](https://arxiv.org/abs/2605.04066)  
**Code**: None currently public  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: RLVR, GRPO, Power-Mean Objective, Adaptive Clipping, Mathematical Reasoning

## TL;DR
This paper proposes APMPO, which unifies GRPO (arithmetic mean) and GMPO (geometric mean) objectives using a "Power-Mean" controlled by the current mean reward. Combined with an adaptive clip range based on reward stability, APMPO enables RLVR training to dynamically switch between "amplifying rare high rewards" and "emphasizing consistency" across different stages, consistently outperforming GRPO, DAPO, and GMPO across 9 mathematical, SQL, and multimodal reasoning benchmarks.

## Background & Motivation

**Background**: The current mainstream approach to enhancing LLM reasoning is Reinforcement Learning with Verifiable Rewards (RLVR). GRPO has become the de facto standard by utilizing group-based normalization for advantage estimation, thereby eliminating the need for a value model. Variants like DAPO and GMPO introduce asymmetric clipping and geometric mean objectives, respectively, on top of GRPO.

**Limitations of Prior Work**: Through a "pre-analysis" of GRPO and GMPO training curves (Figure 1), the authors identify two specific issues. First, GRPO's arithmetic mean objective is extremely sensitive to high-reward outliers, rapidly amplifying single high-scoring trajectories in early stages, which leads to premature entropy collapse and locks the model into sub-optimal policies. Second, GMPO's geometric mean is overly conservative; a single low score can pull down the entire group reward, causing the model to learn almost nothing when correct paths are scarce early on. Furthermore, existing methods use a fixed clipping threshold $\epsilon$, ignoring stability differences in reward distributions between batches—stable batches are over-constrained, while noisy batches are allowed to cause degrading updates.

**Key Challenge**: The mismatch between static objective functions and training dynamics. A single objective must "amplify rare positive signals" during the cold start phase and "emphasize path consistency" during the convergence phase. These two requirements cannot be simultaneously satisfied by either GRPO or GMPO. A similar mismatch exists for the clipping range.

**Goal**: (1) Design an objective function that can switch smoothly between arithmetic and geometric means based on training progress; (2) Design a clipping mechanism that dynamically adjusts the trust region based on the statistical stability of rewards in each batch.

**Key Insight**: The authors observe that the arithmetic mean and geometric mean are special cases of the generalized power mean $M_p$ at $p=1$ and $p\to 0$, respectively. Thus, a continuously adjustable exponent $p$ can incorporate both extremes into a unified framework. Simultaneously, the "mean-to-variance ratio of batch rewards" is viewed as a proxy for policy credibility, driving the expansion and contraction of the clip range.

**Core Idea**: Use the sample performance $\mu_R$ to real-time regulate the power-mean exponent $p=\exp(-\gamma\mu_R)$, allowing the objective to transition naturally from "amplifying" to "consistency-focused." Concurrently, a Feedback Stability Score $\text{FSS}=\mu_R/(\sigma_R+\delta)$ regulates the upper bound of the clip; more stable reward signals allow for more aggressive updates.

## Method

### Overall Architecture
APMPO follows the GRPO framework of "group sampling + group-relative advantage normalization": for each prompt $q$, $G$ responses $\{o_i\}$ are sampled, calculating rewards $R_i$, normalized advantages $\hat{A}_i=(R_i-\mu_R)/(\sigma_R+\delta)$, and token-level importance ratios $r_{i,t}(\theta)$. On top of this, APMPO replaces the arithmetic mean and fixed clip in GRPO with two orthogonal adaptive modules: PMPO (for objective aggregation) and FAC (for clip range). The final objective first uses PMPO to aggregate token-level "non-negative magnitudes" $\phi_{i,t}$ into a scalar per sequence, multiplies it by a direction control term $\text{sgn}(\hat{A}_i)$, averages across samples, and subtracts the KL penalty relative to the reference policy.

### Key Designs

1.  **Power-Mean Policy Optimization (PMPO)**:
    - **Function**: Uses a power-mean operator driven by the current batch reward mean to aggregate token-level objectives, automating the transition from "amplifying rare high rewards" to "emphasizing path consistency."
    - **Mechanism**: Token-level objectives are decomposed into a "non-negative magnitude $\phi_{i,t}(\theta)=|\min(r_{i,t}\hat{A}_i, \rho_{i,t}\hat{A}_i)|$" and a "direction $\text{sgn}(\hat{A}_i)$" (since power-mean requires non-negative inputs). The magnitude is aggregated via power-mean $M_p(\Phi_i)=(\frac{1}{|o_i|}\sum_t \phi_{i,t}^p)^{1/p}$, where the exponent $p=\exp(-\gamma\mu_R)$ decays exponentially as the batch mean reward $\mu_R\in[0,1]$ increases.
    - **Design Motivation**: The authors prove in Appendix D that GRPO and GMPO are two limiting cases of the power mean. Using a continuous exponent encodes the "exploration-consolidation" trade-off directly into the objective, avoiding manual phase switching. The exponential decay ensures a smooth transition from 1 to 0.

2.  **Feedback-Adaptive Clipping (FAC)**:
    - **Function**: Adaptively adjusts the upper bound of the PPO-style clip ratio $\epsilon$ based on the statistical stability of real-time rewards per batch.
    - **Mechanism**: The Feedback Stability Score $\text{FSS}=\mu_R/(\sigma_R+\delta)$ measures reward signal reliability (high mean and low variance imply high credibility). FSS is mapped to $[\epsilon_{\min},\epsilon_{\max}]$ via $\epsilon_{\text{ada}}=\epsilon_{\min}+(\epsilon_{\max}-\epsilon_{\min})\cdot\tanh(\text{FSS})$. The clipping function uses an asymmetric design: $\rho_{i,t}=\text{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{ada}})$, where the lower bound $\epsilon_{\text{low}}$ is fixed to ensure decisive negative pruning.
    - **Design Motivation**: A fixed $\epsilon$ is too conservative for stable batches and too permissive for noisy ones. Combining "reward mean" and the "inverse of variance" as a credibility signal rewards "high-mean stable batches" while penalizing "low-mean noisy batches."

3.  **Three-step Objective Assembly**:
    - **Function**: Integrates PMPO scalar magnitudes, FAC adaptive clipping, and direction control into the final training objective.
    - **Mechanism**: Step 1: Calculate importance ratios with adaptive clipping $\rho_{i,t}(\theta)$. Step 2: Compute token-level non-negative magnitudes $\phi_{i,t}(\theta)=|\min(r_{i,t}\hat{A}_i, \rho_{i,t}\hat{A}_i)|$. Step 3: Define the per-sequence objective $\mathcal{J}_i(\theta)=M_p(\Phi_i)\cdot\text{sgn}(\hat{A}_i)$, average across the batch, and subtract $\beta D_{KL}(\pi_\theta\|\pi_{\text{ref}})$.
    - **Design Motivation**: Decoupling magnitude and direction satisfies the power-mean's requirement for non-negative input while restoring the "advantage maximization" semantics via $\text{sgn}(\hat{A}_i)$.

### Loss & Training
The complete objective is $\mathcal{J}(\theta)=\frac{1}{G}\sum_i \mathcal{J}_i(\theta) - \beta D_{KL}(\pi_\theta\|\pi_{\text{ref}})$. Training uses AdamW, learning rate $1\times10^{-6}$, batch size 512, 8 rollouts per prompt, 400 steps, and temperature 1.0. Key hyperparameters: $\gamma=0.8$ (controls $p$ sensitivity to $\mu_R$), $(\epsilon_{\min},\epsilon_{\max})=(0.2,0.4)$, $\epsilon_{\text{low}}=0.2$, and $\beta=0.001$. Rewards are 0/1 binary rules.

## Key Experimental Results

### Main Results
Evaluation covers Qwen2.5-Math-1.5B-Instruct, Qwen2.5-3B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B across 6 math benchmarks (MATH500/AIME/AMC/Olympiad) + SQL (Spider/BIRD) + Multimodal (Geometry3K).

| Model | Method | MATH500 | AIME24 P@1 | AMC23 P@1 | Olympiad | Avg P@1 |
|------|------|---------|-----------|-----------|----------|---------|
| Qwen2.5-Math-1.5B | GRPO | 75.2 | 13.3 | 52.5 | 39.0 | 37.1 |
| Qwen2.5-Math-1.5B | DAPO | 77.2 | 16.7 | 57.5 | 40.4 | 39.6 |
| Qwen2.5-Math-1.5B | GMPO | 76.6 | 13.3 | 55.0 | 38.7 | 39.0 |
| Qwen2.5-Math-1.5B | **APMPO** | **78.0** | **20.0** | **62.5** | **42.4** | **41.7** (+2.1) |
| Qwen2.5-3B | GRPO | 66.0 | 6.7 | 40.0 | 31.5 | 29.4 |
| Qwen2.5-3B | **APMPO** | **68.4** | **10.0** | **45.0** | **33.2** | **32.4** (+3.0) |
| DS-R1-Distill-1.5B | GRPO | 75.4 | 13.3 | 57.5 | 43.2 | 39.9 |
| DS-R1-Distill-1.5B | DAPO | 79.8 | 20.0 | 60.0 | 43.8 | 42.8 |
| DS-R1-Distill-1.5B | **APMPO** | **81.6** | **23.3** | **65.0** | **46.6** | **46.0** (+3.2) |

### Ablation Study

| Configuration | Avg P@1 (Math) | Notes |
|------|---------------|------|
| Full APMPO | 41.7 | Complete PMPO + FAC model |
| GRPO baseline | 37.1 | Baseline without modifications |
| GRPO + FAC only | 38–39 | Adaptive clipping alone provides gains |
| GRPO + PMPO only | ~40–41 | Adaptive objective is the primary driver |
| FSS using only $\mu_R$ | Slight drop | Loss of stability awareness leads to over-permissiveness |
| FSS using only $1/\sigma_R$ | Slight drop | Incorrect but stable batches are penalized less |
| Linear decay for $p$ | Slight drop | Abrupt switching harms stability; exponential is superior |

### Key Findings
- The gain from the adaptive **objective** (PMPO) is significantly larger than that of the adaptive **clip** (FAC), confirming that the "static objective function" is the primary bottleneck in current RLVR; the two modules show synergistic effects.
- Training curves show APMPO decays entropy slower than GRPO and faster than GMPO, maintaining an optimal "exploration-convergence" balance.
- Computational overhead is negligible, as $p$ and $\epsilon_{\text{ada}}$ only require batch-level $\mu_R$ and $\sigma_R$ statistics; wall-clock time remains comparable to GRPO.

## Highlights & Insights
- **Unification via Power-Mean**: This is the first work to place GRPO's arithmetic mean and GMPO's geometric mean in a continuous parameter space, providing a clear "spectrum" perspective for RLVR algorithm design.
- **Training Progress as a Switching Signal**: Scheduling $p$ via $\mu_R$ rather than step counts makes the method adaptive to training length, warmup, and data difficulty.
- **Portability of Asymmetric Clipping**: The FAC mechanism (adaptive upper bound, fixed lower bound) is applicable to any PPO-style algorithm and can be transferred to RLHF, code RL, or agent RL.
- **Batch Statistics as Features**: Using $\mu_R/\sigma_R$ as a signal for trust region strength avoids the cost of training extra critics, achieving "reward quality awareness" in a minimalist fashion.

## Limitations & Future Work
- Experimental scale is limited to 1.5B/3B models; verification on 7B+ models is pending.
- Heavy reliance on verifiable outcome-based rewards makes it difficult to transfer directly to open-ended QA or summarization tasks.
- The adaptive exponent $p$ utilizes batch-level $\mu_R$ only; further potential may lie in per-prompt or per-token adaptive granularity.
- Entropy gain comparisons with DAPO were not explicitly reported; whether APMPO is orthogonal to DAPO's dynamic sampling is a subject for future investigation.

## Related Work & Insights
- **vs GRPO** (Shao et al. 2024): GRPO is the $p=1$ limit. APMPO matches GRPO early on but automatically tightens to a GMPO style later, yielding a 3-point gain.
- **vs GMPO** (Zhao et al. 2025): GMPO is the $p\to 0$ limit, which is insensitive to high-reward outliers. APMPO actively amplifies these outliers early to help the model discover correct trajectories.
- **vs DAPO** (Yu et al. 2025): DAPO introduces asymmetric clipping and dynamic sampling. APMPO's clip design is similar but data-driven, with the core innovation residing in the objective function.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifying GRPO/GMPO via power-mean is elegant and clean.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive benchmarks and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-theory-experiment chain.
- Value: ⭐⭐⭐⭐ High practical value as a drop-in replacement for GRPO/GMPO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Think Outside the Policy: In-Context Steered Policy Optimization](think_outside_the_policy_in-context_steered_policy_optimization.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](../../ICLR2026/llm_reasoning/temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](../../ICLR2026/llm_reasoning/slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](../../ICLR2026/llm_reasoning/adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)

</div>

<!-- RELATED:END -->
