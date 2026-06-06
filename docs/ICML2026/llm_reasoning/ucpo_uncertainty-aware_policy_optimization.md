---
title: >-
  [Paper Note] UCPO: Uncertainty-aware Policy Optimization
description: >-
  [ICML2026][LLM Reasoning][Uncertainty expression] UCPO addresses the advantage bias caused by fixed uncertainty rewards in existing RL paradigms through two mechanisms: Ternary Advantage Decoupling (TAD) and Dynamic Unce…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Uncertainty expression"
  - "Reinforcement Learning"
  - "Policy Optimization"
  - "Trustworthy AI"
  - "Overconfidence Mitigation"
date: 2026-05-08
content_hash: abea85596968d878
---

# UCPO: Uncertainty-aware Policy Optimization

**Conference**: ICML2026  
**arXiv**: [2601.22648](https://arxiv.org/abs/2601.22648)  
**Code**: https://github.com/xzhouzeng/ucpo  
**Area**: LLM Reasoning  
**Keywords**: Uncertainty expression, Reinforcement Learning, Policy Optimization, Trustworthy AI, Overconfidence Mitigation  

## TL;DR
UCPO addresses the advantage bias caused by fixed uncertainty rewards in existing RL paradigms through two mechanisms: Ternary Advantage Decoupling (TAD) and Dynamic Uncertainty Reward Adjustment (DURA). This enables LLMs to reliably express uncertainty at knowledge boundaries, achieving 79.63% PAQ in mathematical reasoning on Qwen3-8B.

## Background & Motivation

**Background**: LLMs perform excellently in complex reasoning tasks but tend to provide overconfident erroneous assertions (hallucinations) when facing problems beyond their knowledge boundaries. Building trustworthy AI requires models to possess the metacognitive ability to "know what they don't know."

**Limitations of Prior Work**: Existing uncertainty alignment methods follow two paths: (1) SFT—imitation learning using datasets with abstention labels, which suffers from high data synthesis costs and the inability of static data to capture dynamic uncertainty during reasoning; (2) RL—assigning fixed intermediate rewards (e.g., 0.5) to uncertain responses, though such static rewards are extremely sensitive to hyperparameters. On high-difficulty tasks, models engage in "reward hacking" via excessive rejection to obtain stable rewards (evasive degradation); on simple tasks, uncertainty signals are overwhelmed by high rewards from correct answers, leading to persistent overconfidence.

**Key Challenge**: RL frameworks like GRPO generate a fundamental "advantage bias" after introducing ternary rewards (correct/incorrect/uncertain). In high-performance regimes, the advantage of uncertain samples becomes negative (majority suppression), punishing rather than encouraging the expression of doubt. In low-performance regimes, the advantage of uncertain samples dominates the gradient (reward hacking), causing the model to degenerate into outputting uncertainty for all cases.

**Goal**: Design an adaptive RL framework that achieves a dynamic balance in a ternary decision space (correct/incorrect/uncertain) without exhaustive hyperparameter tuning.

**Key Insight**: Theoretically analyze the mathematical mechanism of advantage bias caused by fixed rewards in the GRPO framework—specifically how the sign of the advantage function for uncertain samples flips across different performance regimes, which is a structural issue that static rewards cannot resolve.

**Core Idea**: Decouple deterministic and uncertainty paths into independent channels for advantage estimation (eliminating semantic interference), while dynamically adjusting uncertainty reward weights based on real-time model capability and sample difficulty.

## Method

### Overall Architecture
UCPO introduces two key modifications to the standard GRPO framework: (1) Replacing global advantage normalization with Ternary Advantage Decoupling (TAD), where $G$ rollouts are divided into a deterministic set $\mathcal{S}_{det}$ (correct+incorrect) and an uncertainty set $\mathcal{S}_{unc}$, with advantages calculated in independent channels; (2) Replacing fixed intermediate rewards with Dynamic Uncertainty Reward Adjustment (DURA), which adjusts the gain coefficient $\gamma(q)$ of the uncertainty channel in real-time based on the ratio of correct/incorrect/uncertain responses in the current batch. Given a problem $q$, the model generates $G$ responses, each classified as correct, incorrect, or uncertain.

### Key Designs

1.  **Ternary Advantage Decoupling (TAD)**:
    - **Function**: Eliminates gradient interference between deterministic and uncertainty signals.
    - **Mechanism**: Divides $G$ rollouts into deterministic and uncertainty sets. The deterministic channel uses independent normalization: $\hat{A}_{i,t}^{det} = (r_i - \text{mean}(\mathbf{r}_{det})) / (\text{std}(\mathbf{r}_{det}) + \epsilon)$, ensuring correct paths receive positive reinforcement and incorrect paths receive negative punishment. The advantage of the uncertainty channel is defined as a dynamic projection of the correct sample advantage: $\hat{A}_{i,t}^{unc} = \gamma(q) \cdot \hat{A}_{right}$. By using the correct sample advantage as a "performance anchor," the incentive for uncertainty scales dynamically with the model's current peak reasoning capability. When a set of rollouts lacks correct or incorrect samples, Non-Ternary Filtering (NTF) is executed to discard that sample.
    - **Design Motivation**: In the global normalization of standard GRPO, when correct samples are the majority, the advantage of uncertain samples is pulled negative. The model is punished for "performing below average," even if its caution is justified (majority suppression effect). After decoupling, the uncertainty signal no longer competes with the global performance average.

2.  **Dynamic Uncertainty Reward Adjustment (DURA)**:
    - **Function**: Adaptively adjusts the uncertainty channel gain coefficient based on real-time model capability and sample difficulty.
    - **Mechanism**: The gain coefficient $\gamma(q)$ consists of two terms—an uncertainty gain term $(P_w/(P_u + P_w + \epsilon))(1 - P_u)$ and an uncertainty suppression term $w \cdot (P_r/(P_r + P_w + \epsilon))P_u$, where $P_r, P_w, P_u$ are the proportions of correct, incorrect, and uncertain rollouts in the current group. The gain term amplifies uncertainty incentives when error rates are high (encouraging a shift from false assertions to honest doubt) and uses $(1-P_u)$ to prevent saturation into total abstention. The suppression term punishes unnecessary evasion as model capability increases ($P_r$ grows), pushing the model toward deterministic correct answers.
    - **Design Motivation**: Fixed rewards $r_u$ cannot adapt to changes in model capability during training or difficulty variances across samples. DURA treats the uncertainty channel as a "regulatory buffer"—suppressing hallucinations early in training and pushing the model toward deterministic precision later.

3.  **Non-Ternary Filtering (NTF) and Low-Resource Extension (LRE)**:
    - **Function**: Enhances robustness against extreme distributions and small rollout groups.
    - **Mechanism**: NTF filters out groups lacking correct or incorrect rollouts (similar to zero-advantage handling for all-correct/all-wrong groups in standard GRPO). LRE addresses high variance in gain estimation for small rollout groups through batch-level smoothing and non-linear mapping.
    - **Design Motivation**: When $G$ is small (e.g., $G=4$), ternary proportion estimates within a single group are noisy, leading to drastic fluctuations in $\gamma(q)$. NTF+LRE ensures training stability.

## Key Experimental Results

### Main Results (Mathematical Reasoning, PAQ Metric)

| Method | AIME24 | AMC | MATH500 | Minerva | OlympiadBench | Average PAQ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B Baseline | 73.33 | 91.57 | 96.80 | 45.96 | 69.63 | 75.46 |
| GRPO | 77.01 | 88.35 | 96.46 | 47.18 | 69.22 | 75.64 |
| GRPO-UC ($r_u=0.2$) | 83.75 | 88.98 | 96.31 | 48.60 | 70.68 | 77.66 |
| **UCPO** | **86.11** | **91.95** | **97.28** | **49.15** | **73.67** | **79.63** |
| Llama-3.1-8B Baseline | 3.33 | 15.66 | 45.80 | 15.81 | 14.96 | 19.11 |
| GRPO-UC ($r_u=0.5$) | 0.00 | 21.43 | 57.61 | 26.16 | 19.28 | 24.90 |
| **UCPO** | **5.13** | **28.12** | **60.95** | **22.50** | **25.56** | **28.45** |

### Ablation Study (Llama-3.1-8B, Mathematical Reasoning)

| Configuration | Uncertainty Ratio | PAQ | F1 |
| :--- | :--- | :--- | :--- |
| w/o TAD | 50.33 | 22.56 | 16.21 |
| w/o DURA | 79.91 | 35.22 | 13.16 |
| w/o NTF | 37.96 | 28.51 | 22.93 |
| w/o LRE | 43.19 | 27.83 | 21.12 |
| Full UCPO | **39.09** | **28.45** | **22.65** |

### Key Findings
- Fixed rewards in GRPO-UC are extremely fragile: on Llama-3.1-8B math tasks, $r_u \geq 0.5$ triggers reward hacking, where the uncertainty ratio spikes to 100% and F1 collapses to single digits (9.01). Conversely, for general tasks, $r_u = 0.2$ provides insufficient incentive for uncertainty learning.
- Removing TAD leads to a significant drop in PAQ (28.45 $\rightarrow$ 22.56). Removing DURA causes the uncertainty ratio to surge to 79.91% (reward hacking), proving both components are indispensable.
- UCPO achieves an average PAQ of 79.63% on Qwen3-8B, approximately 2 percentage points higher than the best GRPO-UC variant, without requiring $r_u$ hyperparameter tuning.
- A group size of $G=8$ is optimal for PAQ, while $G=16$ yields better F1, indicating that larger groups provide more stable advantage estimates.

## Highlights & Insights
- **Anchoring uncertainty advantage to correct sample advantage** $\hat{A}_{unc} = \gamma(q) \cdot \hat{A}_{right}$ is an elegant design: it allow the incentive for uncertainty to automatically scale with the model's current peak reasoning capability, avoiding the suppression effect of global normalization and the hacking risks of fixed rewards. This "performance anchoring" concept is transferable to any RL scenario requiring balance between multiple reward types.
- The two-term formula of DURA implements a self-stabilizing system: it encourages abstention when errors are frequent and suppresses it when capability is high. This adaptive mechanism is a fundamental improvement over manual $r_u$ tuning—shifting from "one hyperparameter fits all" to "automatic adjustment based on current state."
- The theoretical analysis of the ternary imbalance problem is very clear—using ternary diagrams to visualize advantage function behavior across different performance regimes intuitively reveals the mathematical mechanism behind the failure of fixed rewards.

## Limitations & Future Work
- The authors acknowledge that the rollout type distribution (initial ratios of $P_r, P_w, P_u$) might affect uncertainty learning but have not fully explored this.
- In multiple-choice scenarios, F1 might decrease because "lucky guesses" are converted to uncertainty—UCPO optimizes for reliability (PAQ) rather than coverage.
- The DURA gain formula relies on intra-group statistics and may degenerate in extreme distributions (e.g., all correct or all wrong), necessitating NTF as a fallback.
- Future work could explore continuous uncertainty expressions (e.g., confidence scores) rather than discrete abstention decisions.

## Related Work & Insights
- TruthRL / KnowRL: Representative methods using fixed intermediate rewards for uncertainty alignment; hyperparameter sensitivity is their core bottleneck.
- GRPO / DeepSeek-R1: The foundational RL framework for UCPO, upon which UCPO introduces the ternary decision space.
- DAPO / Dr.GRPO: Concurrent works improving GRPO training stability, focusing on sampling strategies and clipping mechanisms rather than uncertainty modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Think Outside the Policy: In-Context Steered Policy Optimization](../../ACL2026/llm_reasoning/think_outside_the_policy_in-context_steered_policy_optimization.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](../../ACL2026/llm_reasoning/calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](../../ICLR2026/llm_reasoning/fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](../../ICLR2026/llm_reasoning/drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICML 2026\] Inference Time Optimization with Confidence Dynamics](inference_time_optimization_with_confidence_dynamics.md)

</div>

<!-- RELATED:END -->
