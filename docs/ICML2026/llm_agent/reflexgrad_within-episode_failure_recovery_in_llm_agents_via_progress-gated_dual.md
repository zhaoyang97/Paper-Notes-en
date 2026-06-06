---
title: >-
  [Paper Note] ReflexGrad: Within-Episode Failure Recovery in LLM Agents via Progress-Gated Dual-Process Routing
description: >-
  [ICML 2026 Workshop (FoGen)][LLM Agent][Dual-Process Architecture] ReflexGrad treats TextGrad's "local gradient refinement every 3 steps" as a fast process and Reflexion-style "causal replanning triggered by consecutive…
tags:
  - "ICML 2026 Workshop (FoGen)"
  - "LLM Agent"
  - "Dual-Process Architecture"
  - "Progress-Gated Routing"
  - "TextGrad"
  - "Reflexion"
  - "ALFWorld"
date: 2026-05-08
content_hash: ceadcf4149034699
---

# ReflexGrad: Within-Episode Failure Recovery in LLM Agents via Progress-Gated Dual-Process Routing

**Conference**: ICML 2026 Workshop (FoGen)  
**arXiv**: [2511.14584](https://arxiv.org/abs/2511.14584)  
**Code**: https://github.com/qpiai/reflexgrad (Available)  
**Area**: LLM Agent / Failure Recovery / Test-time Learning  
**Keywords**: Dual-Process Architecture, Progress-Gated Routing, TextGrad, Reflexion, ALFWorld

## TL;DR
ReflexGrad treats TextGrad's "local gradient refinement every 3 steps" as a fast process and Reflexion-style "causal replanning triggered by consecutive low scores" as a slow process. Using a progress-gated routing rule to switch between both within the **same episode** without demonstrations, it improves Qwen-3-8B from 35.1% to 75.4% (+40.3pp) on 134 ALFWorld tasks, outperforming 1-shot LATS / ToT / Self-Refine under compute-equivalent conditions.

## Background & Motivation

**Background**: Mainstream improvement routes for LLM agents in long-horizon text environments like ALFWorld fall into two categories: the Reflexion family (including ReflAct), which relies on "completing a trial → writing self-reflection → retrying in the next round with reflection" for policy-level error correction; and the TextGrad / DSPy / OPRO family, which treats natural language policies as optimizable parameters and uses "textual gradients" for local refinement within a session. There are also search-based methods like ToT / LATS that broaden the action space at each step but **never update the policy itself**.

**Limitations of Prior Work**: A very typical failure mode observed in practice is that an agent commits to a wrong path early in the episode (e.g., attempting to use a stove instead of a microwave to heat an item). Since environment feedback is often vague ("nothing happened"), the agent continues to fine-tune under the incorrect policy until the step budget is exhausted. In other words, the **information required to escape the loop is already present in the failed trajectory**, but Reflexion must wait for the next trial to use it, TextGrad only makes the incorrect policy more precise, and search methods do not update the policy at all.

**Key Challenge**: Policy-level error correction (slow, requiring causal diagnosis) and tactical-level refinement (fast, local) occur at different time scales. Existing methods operate on only one of these scales, and Reflexion's policy-level correction further binds the recovery to the prerequisites of "restarting the trial" and "having demonstrations for bootstrapping."

**Goal**: Construct a **within-episode, zero-shot, training-free** failure recovery mechanism. This requires solving three sub-problems: (i) When should tactical refinement be escalated to strategic replanning? (ii) After escalation, how can new policies and old gradients be merged without conflict? (iii) How can the escalation trigger be made robust to evaluator noise?

**Key Insight**: The authors observe that "incorrect policy + continuous local refinement" leaves a specific signature on the trajectory—**$m$ consecutive low progress scores**. This serves as a gating signal to switch from fast to slow, which is more targeted than a fixed cadence or random switching.

**Core Idea**: Use a progress-gated router $R_t$ to allow the FAST process (TextGrad) to continuously perform local gradients during high scores, while triggering the SLOW process (Reflexion) for causal diagnosis and short-term planning only upon $m$ consecutive low scores. A cooldown window is used to protect plan execution from being corrupted by new gradients.

## Method

### Overall Architecture
The input consists of a natural language task $\tau$ and an initial policy $\pi_0$ (also a natural language string). At each step, the agent receives an observation $o_t$ from the environment and samples an action $a_t \sim \pi_{t-1}(o_t, \tau_{\text{act}}, M_t)$, where $M_t$ is a sliding window of the 10 most recent interactions. The environment returns $o_{t+1}$. An LLM evaluator $E$ scores each transition $s_t = E(o_t, a_t, o_{t+1}, \tau)$ on a scale of 0-10. The most recent $m=5$ scores form a rolling window $W_t$. Based on $W_t$ and a cooldown counter $c_t$, the router **selects exactly one** mode at each step: FAST (approx. 85% of steps), SLOW (approx. 15%), or COOL (plan execution phase). The outputs of both processes are written back to $\pi_{t+1}$ via a fixed-priority merge function, which is then fed to the agent in the next step. The final output is ALFWorld success/failure (determined by the environment oracle; **evaluator scores are used only for routing, not for determining final success**).

### Key Designs

1.  **Progress-Gated Router**:
    *   **Function**: Acts as the master switch to decide whether to execute local fast gradients, slow causal replanning, or cool plan execution at each step.
    *   **Mechanism**: The routing rule is: if $c_t = 0$ and all $s_i \in W_t < \theta_{\text{low}}$, then SLOW; if $c_t > 0$, then COOL; otherwise, FAST. The critical detail is **"$m$ consecutive low scores" rather than "window average is low"**: a single noisy low score will not trigger the switch; all $m$ must be below the threshold to signify that "the local search has converged into a dead end." Once SLOW is triggered, $c_t \leftarrow c = 5$, entering a 5-step cooldown where both processes are suppressed to focus on plan execution. The authors provide a union bound analysis: if the evaluator's false positive rate is $\eta_{\text{fp}} \approx 3\%$, the upper bound for $m=5$ consecutive independent false triggers is $\leq m \eta_{\text{fp}} \approx 15\%$, while in GPT-5 tests, it was effectively 0.
    *   **Design Motivation**: Fixed cadence and random gates ignore the structural signal of a FAST process "stuck on a wrong policy." TextGrad gradients are local and refine the wrong policy further; the only observable trace is "consecutive low scores despite local refinement," which this rule captures. Sensitivity sweeps show the rule is robust to all three thresholds.

2.  **FAST / SLOW Dual-Process + Cooldown Protection**:
    *   **Function**: FAST performs TextGrad-style local textual gradients every $k=3$ steps; SLOW performs Reflexion-style causal diagnosis + short planning only when triggered; Cooldown prevents the new plan from being polluted by stale gradients.
    *   **Mechanism**: In FAST mode, $g_t = \text{LLM}_{\text{grad}}(\pi_t, W_t[-k:], \{(o_i, a_i, o_{i+1})\}_{i=t-k+1}^t)$ produces a natural language suggestion. In SLOW mode, $\rho_t = \text{LLM}_{\text{diag}}(\pi_t, W_t, \{(o_i, a_i, o_{i+1}, s_i)\}_{i=t-m+1}^t)$ outputs a short plan of 1-3 sub-goals, identifying the "suspected root cause + corrective action sequence." Cooldown is necessary because, at the start of a new plan, the first few steps still reflect low scores from the failed trajectory. Without suppression, FAST would generate a new gradient based on these low scores, potentially overwriting the newly initiated plan.
    *   **Design Motivation**: FAST handles "fixable local errors" while SLOW handles "policy-level deviations." Neither is sufficient alone—ablations on GPT-5 show FAST-only at 69.4% and SLOW-only at 53.0%, while the combination yields 88.1%. The expected gain from addition is 29.8pp, but the actual gain is 41.8pp, demonstrating a **super-additive synergy of +12.0pp** facilitated by the cooldown and router.

3.  **Deterministic Priority Merge (plan ≻ gradient ≻ base policy)**:
    *   **Function**: Merges the SLOW plan and FAST gradient into the same natural language policy while ensuring consistency and preventing "averaging" into gibberish.
    *   **Mechanism**: The merge rule is $\pi_{t+1} = \text{Merge}(\pi_t, \text{plan}=\rho_t)$ (if SLOW) / $\text{Merge}(\pi_t, \text{grad}=g_t)$ (if FAST and $t \bmod k = 0$) / $\pi_t$ (otherwise). Priority follows plan ≻ gradient ≻ base policy—**natural language instructions are never averaged**; instead, lower priority instructions are discarded. To control policy drift, four mechanisms limit the length of $\pi_t$: (i) a working memory window of 10 steps, (ii) observed growth from ~150 tokens to ~380 (max 520) at step 15, (iii) each FAST update is based on the current $\pi_t$ where new gradients overwrite old instructions rather than appending, and (iv) SLOW triggers result in plans replacing accumulated gradient drift.
    *   **Design Motivation**: Simple concatenation or averaging leads to linear expansion or non-executable instructions. Fixed priority with automatic overwriting is the most robust solution.

### Loss & Training
**Entirely training-free**. All updates occur at inference time; no parameters are optimized via gradient descent. "Textual gradients" are natural language strings, not PyTorch-style gradients. Fixed hyperparameters: $k=3, m=5, \theta_{\text{low}}=4, c=5$, working memory 10, max\_steps 15. Ten fixed seeds are used for reproducibility.

## Key Experimental Results

### Main Results

**Cross-Model Ablation (ALFWorld 134 tasks, 10 seeds, zero-shot)**:

| Method | GPT-5 | Qwen-3-8B | Δ vs zero-shot |
| :--- | :--- | :--- | :--- |
| Zero-shot | 46.3±1.5 | 35.1±1.5 | — |
| Reflexion-only | 53.0±2.0 | 42.5±2.2 | +6.7 / +7.4 |
| TextGrad-only | 69.4±2.2 | 61.2±1.5 | +23.1 / +26.1 |
| **ReflexGrad** | **88.1±2.0** | **75.4±2.2** | **+41.8 / +40.3** |

Architectural gains across models vary by only 1.5pp (Welch's $t \approx 1.60, p \approx 0.13$), suggesting gains stem from the architecture rather than model scale. Super-additive synergy: +12.0pp on GPT-5 and +6.8pp on Qwen-3-8B.

**Compute-Equivalent Comparison (Qwen-3-8B)**:

| Method | Demos | Calls/Task | Success |
| :--- | :--- | :--- | :--- |
| ReAct | 1-shot | ~10 | 65.7 |
| Self-Refine | 1-shot | ~55 | 68.7±1.9 |
| Tree of Thoughts | 1-shot | ~100 | 69.7±2.2 |
| LATS | 1-shot | ~140 | 72.7±2.0 |
| **ReflexGrad** | **None** | **~100** | **75.4±2.2** |
| ReflAct | 1-shot | ~10 | 80.6 |

Zero-shot ReflexGrad outperforms 1-shot LATS by +2.7pp ($p \approx 0.01$) despite 30% lower compute, and significantly beats ToT and Self-Refine. It does not match the 80.6% of 1-shot ReflAct; the 5.2pp gap is attributed to "verb-receptacle world knowledge" which demonstrations provide implicitly.

### Ablation Study

**Routing Threshold Sensitivity Sweep (GPT-5)**:

| Parameter | Testing Range | Result Range |
| :--- | :--- | :--- |
| Gradient window $k$ | $\{2, 3, 5\}$ | 85.8% – 88.1% |
| Trigger threshold $m$ | $\{3, 5, 7\}$ | 84.3% – 88.1% |
| Score cutoff $\theta_{\text{low}}$ | $\{3, 4, 7\}$ | 84.3% – 88.1% |

Max fluctuation is 3.8pp across sweeps; even the worst configuration is significantly higher than zero-shot. Step budget scaling shows 15 steps is the "sweet spot": 5→10 steps gives +20.1pp, while 15→20 steps gives only +2.2pp, indicating remaining failures are due to lack of world knowledge rather than budget.

### Key Findings
*   **The greatest synergy comes from the coordination of cooldown + routing**, rather than the modules themselves.
*   **The "consecutive $m$ low scores" trigger is the soul of the design**, providing extreme robustness to evaluator noise.
*   **Failure Analysis**: Out of 33 failures, 21 were due to missing world knowledge, 8 due to navigation budget, and 4 due to evaluator false positives.
*   **Zero-shot ReflexGrad > 1-shot ToT**: The architecture partially substitutes for demonstrations by providing continuous gradients and diagnosis, though it cannot replace implicit world knowledge.

## Highlights & Insights
*   The transition condition for FAST/SLOW dual-process is a continuous stream of low scores, which is smoother than binary success/failure signals and cheaper than learned routing policies.
*   The refusal to average natural language instructions in favor of priority-based overwriting is a robust engineering decision that prevents "policy drift" into illegibility.
*   The three mandatory outputs of the SLOW process (trigger, diagnosis, plan) make the architecture auditable and explainable.
*   The authors frame the architectural success as a "falsifiable hypothesis" in Section 3.9, providing empirical support across model scales.

## Limitations & Future Work
*   **Limitations**: Cross-domain testing is limited (TextWorld/OSWorld); GPT-5 is closed-source; the gap with demo-bootstrapped ReflAct remains. The routing rules are tuned for environments with dense feedback and might fail in sparse reward tasks.
*   **Future Work**: Transition from "passive waiting for low scores" to "active detection of policy plateaus" to save budget. Adaptive cooldown lengths could further optimize performance.

## Related Work & Insights
*   **vs Reflexion**: ReflexGrad moves trial-level causal diagnosis into a single episode and removes the demo-bootstrap requirement.
*   **vs TextGrad**: Adds a SLOW route to solve the "local dead end" problem where gradients refine an incorrect policy.
*   **vs AdaPlanner / ReflAct**: ReflexGrad achieves parity or near-parity with demo-based methods using a zero-shot, progress-gated approach.
*   **vs Concurrent Works (2025)**: Although multiple papers explore dual-process or failure recovery, ReflexGrad occupies a unique position as a training-free, within-episode, progress-gated, dual-process, single-model, and demo-free solution.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ (Combination of parts is novel, though individual components are established).
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Solid isolation tests, cross-model verification, and sensitivity sweeps).
*   **Writing Quality**: ⭐⭐⭐⭐⭐ (Professional positioning and clear communication of hypotheses).
*   **Value**: ⭐⭐⭐⭐ (High practical value for agent recovery and audibility in industrial deployments).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DEPO: Dual-Efficiency Preference Optimization for LLM Agents](../../AAAI2026/llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)
- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](process_reward_agents_for_steering_knowledge-intensive_reasoning.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ICLR 2026\] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](../../ICLR2026/llm_agent/webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)
- [\[ICLR 2026\] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](../../ICLR2026/llm_agent/tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)

</div>

<!-- RELATED:END -->
