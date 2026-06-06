---
title: >-
  [Paper Note] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning
description: >-
  [ICML 2026][LLM/NLP][Multi-turn RL] T$^2$PO attributes the training collapse of multi-turn agentic RL to "hesitation"—characterized by over-thinking at the token level and redundant…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Multi-turn RL"
  - "Training Collapse"
  - "Self-calibrated Uncertainty"
  - "Token-level Thinking Intervention"
  - "Turn-level Dynamic Resampling"
date: 2026-05-08
content_hash: cf99500cbb884e0b
---

# T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.02178](https://arxiv.org/abs/2605.02178)  
**Code**: https://github.com/WillDreamer/T2PO (Available)  
**Area**: LLM Reasoning / Agentic RL / Multi-turn Reinforcement Learning  
**Keywords**: Multi-turn RL, Training Collapse, Self-calibrated Uncertainty, Token-level Thinking Intervention, Turn-level Dynamic Resampling

## TL;DR
T$^2$PO attributes the training collapse of multi-turn agentic RL to "hesitation"—characterized by over-thinking at the token level and redundant, ineffective turns at the turn level. By utilizing a self-calibrated uncertainty signal $M_t$, which fuses entropy and confidence, T$^2$PO simultaneously drives Token-level Thinking Intervention (dynamically truncating thought segments) and Turn-level Dynamical Sampling (resampling ineffective turns). This approach consistently outperforms PPO, GRPO, and GiGPO on WebShop, ALFWorld, and Search QA with superior stability.

## Background & Motivation

**Background**: Multi-turn agentic RL—where agents interact multiple times with environments like WebShop or ALFWorld and self-evolve—is a core paradigm for building reasoning-capable LLM agents. Prevailing methods include PPO, GRPO, and GiGPO (group-based critic-free), often combined with rejection-sampling fine-tuning (RFT) for cold starts and length penalties.

**Limitations of Prior Work**: All SOTA baselines suffer from "training collapse," where success rates drop sharply and both KL divergence and gradient norms explode across different seeds. Existing mitigation strategies (fine-grained credit assignment, internal reward shaping, trajectory filtering) are either too coarse-grained (trajectory-level) or rely on indirect reward shaping, making training dynamics highly sensitive to hyperparameters.

**Key Challenge**: Current research treats "training efficiency" and "training stability" as a trade-off—accelerating rollouts introduces off-policy drift, while dense reward shaping distorts the RL objective. **Ours argues these are not contradictory** if the root cause of collapse is identified.

**Goal**: 1) Explain the causes of poor stability by identifying a unified failure mechanism; 2) Design dual-scale interventions at both token and turn levels; 3) Synchronously improve efficiency and stability without introducing additional reward shaping.

**Key Insight**: Analysis of training trajectories reveals that collapse stems from **low exploration efficiency**, manifested as two types of hesitation: (i) token-level over-thinking, where long reasoning chains yield saturated information gain, and (ii) turn-level redundancy, where agents repeatedly attempt similar ineffective turns. This represents a systematic violation of the exploration-exploitation balance.

**Core Idea**: A self-calibrated signal $M_t = \alpha \tilde{H}_t + (1 - \alpha)(1 - \tilde{C}_t)$, capturing both distribution sharpness and top-1 confidence, is used to monitor the variation rate of $M_t$. If the variation is too small (signaling information saturation), the reasoning segment is truncated at the token level; if the change in $\Phi^k$ between turns is negligible, the turn is resampled.

## Method

### Overall Architecture
T$^2$PO inserts two uncertainty-guided intervention modules into the standard multi-turn RL pipeline: **TTI (Token-level Thinking Intervention)**, which dynamically truncates thinking segments during rollout, and **TDS (Turn-level Dynamical Sampling)**, which identifies and resamples ineffective turns. Both modules are grounded in $M_t$. Training is conducted using a memory context window (limited to the last $P$ turns), turn-level discounted returns $R(\tau^k) = \sum_{j=k}^K \beta^{j-k} r^j$, strict formatting penalties, and GRPO-style policy updates.

### Key Designs

1.  **Self-calibrated Uncertainty Signal $M_t$**:
    - **Function**: Provides a scalar signal that can distinguish "nearly uniform" from "highly sharp" distributions while remaining sensitive to tail probabilities in large vocabularies (e.g., 152K in Qwen3).
    - **Mechanism**: Shannon entropy $H_t = -\sum_i p_t^{(i)} \log p_t^{(i)}$ lacks discriminative power at extremes in large vocabularies, while top-$j$ confidence $C_t = -\frac{1}{j} \sum_{i=1}^j \log p_t^{(i)}$ ignores the tail. $M_t$ is constructed by normalizing these signals ($\tilde{H}_t$ and $\tilde{C}_t$) and fusing them: $M_t = \alpha \tilde{H}_t + (1 - \alpha)(1 - \tilde{C}_t)$. This retains both tail sensitivity and top-1 stratification.
    - **Design Motivation**: Single indicators have blind spots; the fused $M_t$ acts as a reliable scalar for "local distribution stability," allowing threshold rules to maintain consistent semantics across tokens and turns.

2.  **TTI (Token-level Thinking Intervention) — Adaptive Termination of Reasoning**:
    - **Function**: Dynamically determines when reasoning has saturated and injects the reasoning terminator `</think>` into the logits to stop over-thinking.
    - **Mechanism**: After a minimum prefix length $L_{\min}$, the algorithm monitors the change $\Delta_t^k = |M_t^k - M_{t-1}^k|$. When the average change within window $N$ falls below threshold $\varepsilon$, the `</think>` token logit is set to $+\infty$. A fixed queue $\mathcal{Q} = [\texttt{</think>}, \backslash n, \texttt{<action>}]$ is then injected to ensure structured output.
    - **Design Motivation**: Unlike fixed-length truncation or indirect reward control, TTI provides **direct, adaptive, token-level hard truncation**. It uses a sliding window to smooth spikes and avoids premature truncation of critical task-specific tokens.

3.  **TDS (Turn-level Dynamical Sampling) — Resampling Ineffective Turns**:
    - **Function**: Detects ineffective interactions at the turn level (trips where the agent's state remains stagnant) and resamples them to optimize rollout budgets.
    - **Mechanism**: A turn-level signal $\Phi^k = (\prod_{t=1}^T M_t)^{1/T}$ is calculated via geometric mean. If the change between turns $\Gamma^k = |\Phi^k - \Phi^{k-1}|$ is less than $\eta$ (indicating no belief shift), the agent discards the turn and rollouts again from the same state, up to $B_{\max}$ times.
    - **Design Motivation**: Repeated ineffective turns are a primary cause of multi-turn RL collapse. TDS removes them during the rollout phase, stabilizing gradient signals and saving computation.

### Loss & Training
The framework utilizes RFT cold starts, a memory context window for VRAM efficiency, and turn-level discounted returns. Strategy updates are performed using GRPO-style critic-free updates. TTI and TDS intervene during rollouts without modifying the core policy update equations.

## Key Experimental Results

### Main Results
Evaluation on WebShop and ALFWorld (averaging 5 seeds ± std) using Qwen3-4B + RFT:

| Method | WebShop Task Score | WebShop Success Rate | ALFWorld Success Rate |
| :--- | :--- | :--- | :--- |
| GPT-4o (Prompting) | 31.8 | 23.7 | 48.0 |
| Gemini-2.5-Pro (Prompting) | 42.5 | 35.9 | 60.3 |
| Claude Sonnet 4 (Prompting) | 45.6 | 39.8 | 63.7 |
| Qwen3-4B + SFT | 70.91 | 26.56 | 64.06 |
| PPO | 70.34 ± 8.63 | 61.93 ± 5.93 | 75.39 ± 3.81 |
| GRPO | 80.02 ± 7.94 | 68.56 ± 4.11 | 77.35 ± 0.62 |
| GiGPO | 86.03 ± 4.18 | 73.83 ± 3.04 | 80.47 ± 2.43 |
| **T$^2$PO (Ours)** | **Highest & Lowest Std** | **Highest** | **Highest** |

Ours achieves the best performance across all tasks with **significantly lower variance** across seeds, directly addressing training collapse.

### Ablation Study

| Configuration | Key Observation |
| :--- | :--- |
| Full T$^2$PO | Optimal performance and stability; TTI + TDS synergistic. |
| TTI Only | Shorter reasoning segments, improved average stability. |
| TDS Only | Fewer ineffective turns, higher rollout efficiency. |
| Pure Entropy $H_t$ | Threshold rules fail due to poor discriminative power in large vocabularies. |
| Pure Confidence $C_t$ | Loss of tail information; TTI causes mis-truncation. |
| Truncate at $M_t$ Peak | Performance decreases—critical task tokens are removed. |

### Key Findings
- The $M_t$ trajectory typically follows a "hump" shape. The peak area contains task-specific tokens (e.g., product names in WebShop), while the post-peak "convergence area" contains the redundant reasoning suitable for TTI.
- The combination of one-time activation, $L_{\min}$ prefix protection, and sliding windows ensures TTI does not degrade performance.
- Using the geometric mean for $\Phi^k$ is more stable than the arithmetic mean, as it prevents extreme token outliers from skewing the Turn-level belief state.

## Highlights & Insights
- **Unified Perspective**: Using a common self-calibrated uncertainty signal for dual-scale interventions (TTI/TDS) provides an elegant solution to previously disjoint control problems.
- **Hard Truncation vs. Soft Penalty**: Implementing stop-gradient hard truncation and token injection during rollouts is a more robust engineering tool than indirect signals like length penalties.
- **Strategic Truncation**: The insight that one should not truncate at the peak of $M$ demonstrates a nuanced understanding of reasoning traces—peaks represent high information density, not over-thinking.
- **Broad Applicability**: The TDS mechanism, based on belief shifts, can be generalized to various multi-turn tasks (tool-use, dialogue, coding) as a trajectory quality controller.

## Limitations & Future Work
- The abundance of hyperparameters ($\varepsilon, \eta, L_{\min}, N, B_{\max}$) requires tuning across different tasks; an automated tuning method is not yet provided.
- The self-calibrated signal depends on estimates of $H_{\min}$ and $H_{\max}$, which may drift in long horizons.
- Experiments were conducted on 4B-scale models; scalability to larger models (70B+) and hyper-complex environments like SWE-Bench remains to be tested.

## Related Work & Insights
- **vs. trajectory-level filters (SimpleTIR/rStar2-Agent)**: These filter whole trajectories post-hoc; T$^2$PO resamples at the turn level during rollouts, maintaining finer granularity and preserving valid data segments.
- **vs. GiGPO/DAPO**: While they focus on advantage estimation, T$^2$PO modifies the rollout itself, allowing the methods to be orthogonal and combinable.
- **vs. Internal Reward methods (SEED-GRPO)**: Instead of feeding internal signals back into rewards (which can pollute training dynamics), T$^2$PO uses them for explicit behavioral control, offering a cleaner logical framework.

## Rating
- Novelty: ⭐⭐⭐⭐ (Dual-scale hesitation perspective + unified signal)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-task coverage + stability analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear logical flow and progression)
- Value: ⭐⭐⭐⭐ (Plug-and-play stabilization tool for agentic RL)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](../../ICLR2026/llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)
- [\[ACL 2026\] Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards](../../ACL2026/llm_nlp/generative_floor_plan_design_with_llms_via_reinforcement_learning_with_verifiabl.md)
- [\[ICML 2026\] Multi-Agent Teams Hold Experts Back: Why Self-Organizing LLM Teams Cannot Retain "Experts"](multi-agent_teams_hold_experts_back.md)

</div>

<!-- RELATED:END -->
