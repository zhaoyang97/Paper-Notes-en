---
title: >-
  [Paper Note] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors
description: >-
  [ACL 2026][LLM Agent][Tool Use] Fission-GRPO is proposed to dynamically transform tool execution errors into on-policy correction training instances within the RL training loop. By generating diagnostic feedback via a le…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Tool Use"
  - "Error Recovery"
  - "Reinforcement Learning"
  - "GRPO"
  - "Error Simulator"
date: 2026-05-08
content_hash: e8791d088854c865
---

# Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors

**Conference**: ACL 2026  
**arXiv**: [2601.15625](https://arxiv.org/abs/2601.15625)  
**Code**: [GitHub](https://github.com/zxzadm/Fission-GRPO)  
**Area**: LLM Alignment  
**Keywords**: Tool Use, Error Recovery, Reinforcement Learning, GRPO, Error Simulator

## TL;DR

Fission-GRPO is proposed to dynamically transform tool execution errors into on-policy correction training instances within the RL training loop. By generating diagnostic feedback via a learned error simulator and resampling recovery trajectories, it improves the error recovery rate of Qwen3-8B by 5.7% and the overall accuracy from 42.75% to 46.75%.

## Background & Motivation

**Background**: LLMs can effectively call tools, but in multi-turn executions, small models often fall into hallucinated retry loops after encountering API errors instead of interpreting feedback and recovering.

**Limitations of Prior Work**: (1) Standard RL (e.g., GRPO) treats errors only as sparse negative rewards, telling the model it is "wrong" without teaching "how to recover"; (2) When all sampled trajectories fail, the advantage variance is zero, leading to vanishing gradients; (3) Offline synthesized error correction datasets suffer from distribution shift as the policy evolves.

**Key Challenge**: Existing methods treat errors as "outcomes to be avoided" rather than "experiences to be learned."

**Goal**: Transform execution errors into dense, on-policy aligned correction training signals.

**Key Insight**: Analogous to nuclear fission—a single error event triggers a chain reaction, generating multiple correction trajectories.

**Core Idea**: Intercept failed trajectories → Use a learned error simulator to generate diagnostic feedback → Resample $G'$ recovery trajectories ("fission") from the augmented context to continuously align the current policy's error patterns within the training loop.

## Method

### Overall Architecture

A three-stage closed loop: Stage 1 standard GRPO exploration (establishing basic tool capabilities); Stage 2 error identification and correction sample construction (generating diagnostic feedback using an error simulator $S_\phi$); Stage 3 fission update (resampling $G'$ recovery trajectories from the correction context and updating the policy).

### Key Designs

1.  **Error Simulator $S_\phi$**:

    - **Function**: Generates realistic runtime diagnostic feedback.
    - **Mechanism**: SFT based on Qwen3-32B using approximately 2K instances (system prompt + tool description + dialogue state + failed call + correct call + teacher diagnostic message). During inference, it takes a failed call as input and outputs a diagnostic string resembling runtime errors, constrained to non-leaking descriptions (e.g., "parameter status expects value OPEN").
    - **Design Motivation**: Real API interactions are costly and non-reproducible; the learned simulator provides controllable diagnostic feedback. The non-leaking constraint prevents direct exposure of answers.

2.  **Fission Resampling**:

    - **Function**: Generates dense correction signals from a single error.
    - **Mechanism**: For each correction context $x_{\text{corr}} = [x; \tau_{\text{err}}; f]$, $G'$ recovery trajectories $\{\tau'_j\}_{j=1}^{G'} \sim \pi_\theta(\cdot | x_{\text{corr}})$ are sampled. Normalized advantages are calculated within the fission group to update the policy using the GRPO objective.
    - **Design Motivation**: Diagnostic feedback increases intra-group outcome diversity, alleviating the vanishing gradient problem in standard GRPO when all samples in a group fail.

3.  **Time-varying Composite Reward**:

    - **Function**: Guides the policy from format compliance to semantic precision.
    - **Mechanism**: $R(\tau,t) = \frac{1}{3}[w_{\text{fmt}}(t)R_{\text{fmt}} + w_{\text{corr}}(t)R_{\text{corr}} + R_{\text{len}}]$, where format weight decreases and correctness weight increases over time. The correctness reward combines function selection precision and parameter F1.
    - **Design Motivation**: Learn formatting in the early stages and focus on parameter accuracy later.

### Loss & Training

Standard GRPO and fission-correction GRPO are performed alternately. A LIFO buffer ensures that correction samples remain aligned with the current policy.

## Key Experimental Results

### Main Results

Qwen3 series models on BFCL v4 Multi-Turn:

| Method | 1.7B | 4B | 8B |
|------|------|------|------|
| Base | 7.80 | 19.37 | 42.75 |
| GRPO | 17.12 | 36.38 | 42.75 |
| DAPO | 16.00 | 38.25 | — |
| **Fission-GRPO** | **20.38** | **40.50** | **46.75** |

Generalization results on TAU-Bench show up to a +17.4% improvement on the Retail set.

### Ablation Study

| Configuration | Overall | Description |
|------|---------|------|
| GRPO only | 42.75 | No error recovery training |
| + Offline error data | 44.00 | Static distribution shift |
| + Fission (No simulator) | 44.50 | No diagnostic feedback |
| + Fission-GRPO | **46.75** | Full framework |

### Key Findings

- The error recovery rate improved by 5.7% (from ~20% to ~26%), which is the primary source of the overall accuracy gain.
- The simulator achieved a 96% non-leaking rate (human evaluation, Cohen's $\kappa=0.71$) and maintained generalization on unseen tool patterns.
- The Fission mechanism is consistently effective across model scales (1.7B/4B/8B).

## Highlights & Insights

- **"Errors are experiences rather than punishments"**: This philosophy changes the RL training paradigm for tool use—not just telling the model it is wrong, but teaching it how to fix the error.
- **LIFO Buffer**: Ensures correction samples are always aligned with the latest policy, avoiding the distribution shift common in offline datasets.
- **Fission Metaphor**: Intuitive and powerful—one error leads to multiple recovery attempts, providing dense training signals.

## Limitations & Future Work

- The error simulator is based on Qwen3-32B (significantly larger than the target models), so deployment costs must be considered.
- Validation was only conducted for tool calling; the transferability to reasoning or code error recovery remains to be verified.
- Tuning the LIFO buffer size and the fission group size $G'$ requires empirical effort.

## Related Work & Insights

- **vs DAPO/NGRPO**: These methods reshape the loss surface of negative signals but do not construct positive signals; Fission-GRPO actively constructs recovery trajectories.
- **vs ToolACE/LoopTool**: These rely on offline synthesized correction data, which suffers from severe distribution shift; Fission-GRPO maintains alignment through online generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of integrating error recovery training directly into the RL loop is highly novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple model scales, multiple benchmarks, and human evaluation of simulator reliability.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and a vivid fission analogy.
- Value: ⭐⭐⭐⭐⭐ Provides practical value for improving the robustness of tool-using agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](../../ICML2026/llm_agent/recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[ACL 2026\] Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning](temp-r1_a_unified_autonomous_agent_for_complex_temporal_kgqa_via_reverse_curricu.md)

</div>

<!-- RELATED:END -->
