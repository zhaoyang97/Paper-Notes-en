---
title: >-
  [Paper Note] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning
description: >-
  [ICML 2026][LLM/NLP][Multi-turn RL] T$^2$PO attributes training collapse in multi-turn agentic RL to "hesitation"—overthinking at the token level and repeated ineffective actions at the turn level. It introduces a self-c…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Multi-turn RL"
  - "training collapse"
  - "self-calibrated uncertainty"
  - "token-level thinking intervention"
  - "turn-level dynamic resampling"
date: 2026-05-08
content_hash: 5eea4a6078b588d7
---

# T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.02178](https://arxiv.org/abs/2605.02178)  
**Code**: https://github.com/WillDreamer/T2PO (available)  
**Area**: LLM Reasoning / Agentic RL / Multi-Turn Reinforcement Learning  
**Keywords**: Multi-turn RL, training collapse, self-calibrated uncertainty, token-level thinking intervention, turn-level dynamic resampling

## TL;DR
T$^2$PO attributes training collapse in multi-turn agentic RL to "hesitation"—overthinking at the token level and repeated ineffective actions at the turn level. It introduces a self-calibrated uncertainty signal $M_t$ (combining entropy and confidence) to simultaneously drive token-level Thinking Intervention (dynamic truncation of think segments) and turn-level Dynamical Sampling (resampling ineffective turns). This approach consistently outperforms PPO/GRPO/GiGPO on WebShop, ALFWorld, and Search QA, achieving stable training.

## Background & Motivation

**Background**: Multi-turn agentic RL (agents interacting and self-evolving in environments like WebShop and ALFWorld) is a core paradigm for constructing reasoning-capable LLM agents. Mainstream methods include PPO, GRPO, and GiGPO (group-based critic-free), often combined with rejection-FT cold start and length penalty techniques.

**Limitations of Prior Work**: All SOTA baselines suffer from "training collapse"—changing the random seed can cause the success rate to plummet, KL divergence and gradient norm to explode, and training to fail. Existing mitigation strategies (fine-grained credit assignment, internal reward shaping, trajectory filtering) are either too coarse (trajectory-level filter) or rely on indirect reward shaping, resulting in training dynamics that are extremely sensitive to hyperparameters.

**Key Challenge**: Existing work treats "training efficiency" and "training stability" as a trade-off—accelerating rollout introduces off-policy drift/stale policy; dense reward shaping undermines RL objectives. **This paper argues these are not fundamentally contradictory**—if the true cause of collapse is identified.

**Goal**: 1) Explain why stability is poor—identify a unified failure mechanism; 2) Design dual-scale interventions at the token and turn levels; 3) Achieve simultaneous improvements in efficiency and stability without extra reward shaping.

**Key Insight**: Analysis of training trajectories reveals that collapse stems from **low exploration efficiency**, manifesting as two types of hesitation: (i) token-level over-thinking—long reasoning chains with saturated information gain; (ii) turn-level repeated ineffectiveness—agents repeatedly attempt the same turn in an incorrect action space. This systematically violates the exploration-exploitation balance.

**Core Idea**: Use a self-calibrated signal $M_t=\alpha\tilde H_t+(1-\alpha)(1-\tilde C_t)$ that captures both "distribution sharpness" and "top-1 confidence" to monitor the rate of change between tokens. If the rate is too small (information saturated), forcibly truncate the think segment at the token level; if the change between turns $\Phi^k$ is too small, resample the turn.

## Method

### Overall Architecture
T$^2$PO inserts two uncertainty-guided intervention modules into the standard multi-turn RL pipeline (base LLM + RFT cold start + SOTA policy update): **TTI (Token-level Thinking Intervention)** dynamically truncates the think segment during rollout; **TDS (Turn-level Dynamical Sampling)** identifies and resamples ineffective turns during rollout. Both interventions are driven by $M_t$. Training uses a memory context window (only the most recent $P$ turns), turn-level discounted return $R(\tau^k)=\sum_{j=k}^K\beta^{j-k}r^j$, strict format penalties, and GRPO-like policy updates.

### Key Designs

1. **Self-Calibrated Uncertainty Signal $M_t$**:

    - **Function**: Provides a scalar signal under large vocabularies (e.g., Qwen3's 152K) that can distinguish between "nearly uniform" and "highly sharp" distributions and is sensitive to tail probabilities, serving as the unified driver for TTI/TDS.
    - **Mechanism**: Shannon entropy $H_t=-\sum_i p_t^{(i)}\log p_t^{(i)}$ alone has poor discrimination at extremes (e.g., the entropy difference between "(1,0,0,...)" and "(0.5,0.5,0,...)" is only $\log 2$ in a 152K vocabulary, negligible relative to the total scale); top-$j$ confidence $C_t=-\frac{1}{j}\sum_{i=1}^j\log p_t^{(i)}$ focuses only on arg-max, ignoring the tail. Trajectory normalization is applied: $\tilde H_t=(H_t-H_{\min})/(H_{\max}-H_{\min})$, $\tilde C_t=(C_t-C_{\min})/(C_{\max}-C_{\min})$, then fused as $M_t=\alpha\tilde H_t+(1-\alpha)(1-\tilde C_t)$. The paper uses contour plots to show that $M_t$ retains both the tail sensitivity of entropy and the top-1 stratification of confidence.
    - **Design Motivation**: Each single metric has blind spots; the fused $M_t$ is a reliable scalar for "local distribution stability," enabling threshold rules with consistent semantics across tokens/turns.

2. **TTI (Token-level Thinking Intervention)—Stopping Reasoning at the Right Moment**:

    - **Function**: Dynamically determines when "thinking is saturated," forcibly injecting the reasoning terminator `</think>` into the logits to stop overthinking.
    - **Mechanism**: After a minimum prefix length $L_{\min}$, monitor adjacent changes $\Delta_t^k=|M_t^k-M_{t-1}^k|$. When the average change within a window of size $N$ falls below threshold $\varepsilon$ ($\frac{1}{N+1}\sum_{i=0}^N\Delta_{t-i}^k<\varepsilon$), a non-hesitation event is triggered: at step $t^*+1$, set the logit of token 153668 (`</think>`) to $+\infty$ and all others to $-\infty$, ensuring $p_\theta(y_{t^*+1}=\texttt{</think>}\mid y_{\le t^*})=1$. Then inject a fixed queue $\mathcal{Q}=[\texttt{</think>},\backslash n,\texttt{<action>}]$ to guarantee structured output. **Key trick**: do not truncate at the $M_t$ peak (which corresponds to task-specific tokens; truncating here harms performance), but in the "convergence region" after the peak. Also includes one-time activation (at most once per generation) and a global $L_{\max}$ fallback.
    - **Design Motivation**: Previous work either does not truncate, truncates at a fixed length (coarse), or controls implicitly via reward (indirect); TTI is a **direct, adaptive, token-level hard truncation**, using a sliding window to smooth out spikes and avoid mis-truncation at key task tokens.

3. **TDS (Turn-level Dynamical Sampling)—Resampling Ineffective Turns**:

    - **Function**: Detects ineffective interactions at the turn level ("almost no difference from the previous turn"), discards the current turn's generation, and resamples to avoid wasting rollout budget.
    - **Mechanism**: Compute the geometric mean of $M_t$ across all tokens in a turn to obtain the turn-level signal $\Phi^k=(\prod_{t=1}^T M_t)^{1/T}$, then measure the change between adjacent turns $\Gamma^k=|\Phi^k-\Phi^{k-1}|$. When $\Gamma^k<\eta$ (agent's internal belief barely changes), trigger regeneration: discard $\mathbf{a}^k$ and rollout again in the same state, until $\Gamma^k\ge\eta$ or the resampling limit $B_{\max}$ is reached. **Key design**: cannot directly apply DAPO-style filters from single-turn RL—multi-turn RL lacks dense per-turn rewards, so TDS uses turn-level internal uncertainty change as a "proxy for accuracy."
    - **Design Motivation**: Repeatedly attempting ineffective turns on erroneous trajectories is another main cause of multi-turn RL training collapse; TDS directly removes them during rollout, saving computation and stabilizing gradient signals.

### Loss & Training
RFT cold start + memory context window (only the most recent $P$ turns to save memory) + turn-level discounted return $R(\tau^k)=\sum_{j=k}^K\beta^{j-k}r^j$ + strict format penalties (enforcing think/action labels) + GRPO-like critic-free policy updates. TTI/TDS intervene during rollout and do not alter policy updates.

## Key Experimental Results

### Main Results
On both WebShop and ALFWorld benchmarks (averaged over 5 seeds ± std), with Qwen3-4B + RFT cold start as the base model:

| Method | WebShop Task Score | WebShop Success Rate | ALFWorld Success Rate |
|------|---------------------|----------------------|------------------------|
| GPT-4o (Prompting) | 31.8 | 23.7 | 48.0 |
| Gemini-2.5-Pro (Prompting) | 42.5 | 35.9 | 60.3 |
| Claude Sonnet 4 (Prompting) | 45.6 | 39.8 | 63.7 |
| Qwen3-4B + SFT | 70.91 | 26.56 | 64.06 |
| PPO | 70.34 ± 8.63 | 61.93 ± 5.93 | 75.39 ± 3.81 |
| GRPO | 80.02 ± 7.94 | 68.56 ± 4.11 | 77.35 ± 0.62 |
| GiGPO | 86.03 ± 4.18 | 73.83 ± 3.04 | 80.47 ± 2.43 |
| **T$^2$PO (Ours)** | **Highest and lowest std** | **Highest** | **Highest** |

Key metric: T$^2$PO achieves the best results on all three tasks (WebShop, ALFWorld, Search QA), with **significantly lower cross-seed variance than baselines** (directly mitigating training collapse).

### Ablation Study

| Configuration | Key Phenomenon | Description |
|------|---------|------|
| Full T$^2$PO | Optimal and stable training | TTI + TDS both effective |
| TTI only | Shorter think segments per turn, improved average stability | Controls token-level hesitation |
| TDS only | Fewer ineffective turns, higher rollout efficiency | Controls turn-level hesitation |
| Pure entropy $H_t$ instead of $M_t$ | Threshold rules fail due to poor discrimination in large vocabularies | Validates necessity of $M_t$ |
| Pure confidence $C_t$ instead of $M_t$ | Tail information lost, TTI misfires | Validates need for fusion |
| Truncating at $M_t$ peak | Performance drops—task-specific key tokens are truncated | Validates sliding-window design |

### Key Findings
- $M_t$ exhibits a "rise-then-fall" hump along the response length; the peak often corresponds to task-specific tokens (e.g., product names in WebShop), and only after the peak does redundant reasoning appear—this empirical finding is central to TTI's design.
- The combination of one-time activation, $L_{\min}$ prefix protection, and sliding window is crucial for TTI's engineering robustness.
- TDS uses the geometric mean for $\Phi^k$ instead of the arithmetic mean because internal uncertainty is often skewed by a few high-entropy tokens; the geometric mean more stably reflects the turn's overall belief state.
- No external reward shaping is introduced, yet both efficiency and stability improve, validating the core argument that "hesitation is the root cause of collapse."

## Highlights & Insights
- Using a **self-calibrated uncertainty** to unify interventions at two scales (TTI/TDS) offers an elegant unified perspective—previously, token-level and turn-level controls were handled separately; this work shows a single $M_t$ suffices.
- "Hard truncation with stop-gradient + token queue injection" replaces "soft penalties" as a sharp engineering tool—deterministically stopping when needed during rollout is much simpler and more effective than indirect signals like length penalty.
- The counterintuitive detail of "not truncating at the $M_t$ peak" reflects careful analysis of the reasoning trace: the peak corresponds to "high information density" rather than "overthinking," so truncating here destroys task relevance—a textbook ablation lesson.
- TDS's "resample if belief shift is insufficient" mechanism can be transferred to any multi-turn RL (including tool use, multi-turn dialogue, code agents), serving as a general trajectory quality controller.

## Limitations & Future Work
- TTI/TDS involve several thresholds ($\varepsilon, \eta, L_{\min}, N, B_{\max}$); cross-task adaptation still requires tuning, and no automatic tuning method is provided.
- The self-calibrated signal depends on normalization ranges ($H_{\min}, H_{\max}$, etc.), which may drift statistically over long horizons.
- Experiments focus on 4B-scale Qwen and three environments; scalability to larger models (70B+) and more complex tool-use environments (e.g., SWE-Bench) remains untested.
- Combination with off-policy RL algorithms (e.g., KL-controlled importance sampling) is unexplored; compatibility with async rollout acceleration also needs verification.

## Related Work & Insights
- **vs SimpleTIR / rStar2-Agent (trajectory-level filter)**: These filter out entire trajectories containing void turns post hoc; T$^2$PO resamples individual turns during rollout, offering finer granularity without discarding valid data.
- **vs GiGPO / DAPO (group-based critic-free)**: These modify advantage estimation; T$^2$PO modifies the rollout itself. The two are orthogonal and can be combined, as demonstrated with GRPO-like updates in this work.
- **vs SEED-GRPO / DeepConf (internal reward using entropy/confidence)**: These feed internal signals back into the reward; T$^2$PO uses internal signals for explicit truncation/resampling, avoiding the training dynamics pollution introduced by reward shaping, resulting in a cleaner logic.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual-scale hesitation perspective + self-calibrated signal + hard truncation/resampling mechanism, with a distinctive combined approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers WebShop / ALFWorld / Search QA + multiple baselines + cross-seed variance comparison, with data supporting collapse mitigation.
- Writing Quality: ⭐⭐⭐⭐ The "hesitation is defeat" argument is presented coherently, with Figures 1-4 progressively illustrating phenomena, mechanisms, and effects.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play stabilization tool for agentic RL; open-source code will facilitate rapid community adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](../../ICLR2026/llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](../../AAAI2026/llm_nlp/lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)
- [\[ACL 2026\] Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards](../../ACL2026/llm_nlp/generative_floor_plan_design_with_llms_via_reinforcement_learning_with_verifiabl.md)
- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](../../AAAI2026/llm_nlp/quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)

</div>

<!-- RELATED:END -->
