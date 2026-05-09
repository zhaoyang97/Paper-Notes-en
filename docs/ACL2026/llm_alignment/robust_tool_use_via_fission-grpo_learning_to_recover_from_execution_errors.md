---
title: >-
  [Paper Note] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors
description: >-
  [ACL 2026][LLM Alignment][Tool Calling] This paper proposes Fission-GRPO, which dynamically converts tool execution errors into on-policy corrective training instances within the RL training loop. A learned error simulator generates diagnostic feedback, and recovery trajectories are resampled from the augmented context. The approach improves the error recovery rate of Qwen3-8B by 5.7% and raises overall accuracy from 42.75% to 46.75%.
tags:
  - ACL 2026
  - LLM Alignment
  - Tool Calling
  - Error Recovery
  - Reinforcement Learning
  - GRPO
  - Error Simulator
date: 2026-05-08
content_hash: 795f1a20c9270306
---

# Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors

**Conference**: ACL 2026
**arXiv**: [2601.15625](https://arxiv.org/abs/2601.15625)
**Code**: [GitHub](https://github.com/zxzadm/Fission-GRPO)
**Area**: LLM Alignment
**Keywords**: Tool Calling, Error Recovery, Reinforcement Learning, GRPO, Error Simulator

## TL;DR

This paper proposes Fission-GRPO, which dynamically converts tool execution errors into on-policy corrective training instances within the RL training loop. A learned error simulator generates diagnostic feedback, and recovery trajectories are resampled from the augmented context. The approach improves the error recovery rate of Qwen3-8B by 5.7% and raises overall accuracy from 42.75% to 46.75%.

## Background & Motivation

**State of the Field**: LLMs can effectively invoke tools, but when encountering API errors during multi-turn execution, smaller models tend to fall into hallucinated retry loops rather than interpreting the feedback and recovering.

**Limitations of Prior Work**: (1) Standard RL methods such as GRPO treat errors merely as sparse negative rewards, informing the model that something went wrong but not how to recover; (2) when all sampled trajectories fail, zero advantage variance causes gradient vanishing; (3) offline-synthesized error-correction datasets suffer from distribution shift as the policy evolves.

**Root Cause**: Existing methods treat errors as outcomes to be avoided rather than experiences from which to learn.

**Paper Goals**: Transform execution errors into dense, on-policy-aligned corrective training signals.

**Starting Point**: Drawing an analogy to nuclear fission — a single error event triggers a chain reaction that generates multiple corrective trajectories.

**Core Idea**: Intercept failed trajectories → generate diagnostic feedback via a learned error simulator → resample $G'$ recovery trajectories from the augmented context ("fission"), continuously aligning the corrective training signal with the current policy's error patterns throughout the training loop.

## Method

### Overall Architecture

A three-stage closed loop: Stage 1 applies standard GRPO exploration to establish basic tool-use capabilities; Stage 2 performs error identification and corrective sample construction by generating diagnostic feedback via the error simulator $S_\phi$; Stage 3 executes fission updates by resampling $G'$ recovery trajectories from the corrected context and updating the policy.

### Key Designs

1. **Error Simulator $S_\phi$**:

    - **Function**: Generates realistic runtime diagnostic feedback.
    - **Mechanism**: Fine-tuned from Qwen3-32B via SFT on approximately 2K training instances (system prompt + tool description + dialogue state + failed call + correct call + teacher diagnostic message). At inference time, it takes a failed call as input and outputs a diagnostic string resembling a runtime error, constrained to be non-leaking (e.g., "parameter `status` expected value `OPEN`").
    - **Design Motivation**: Real API interactions are costly and non-reproducible; a learned simulator provides controllable diagnostic feedback. The non-leaking constraint prevents direct exposure of the answer.

2. **Fission-Based Resampling**:

    - **Function**: Generates dense corrective signals from a single error.
    - **Mechanism**: For each corrected context $x_{\text{corr}} = [x; \tau_{\text{err}}; f]$, $G'$ recovery trajectories $\{\tau'_j\}_{j=1}^{G'} \sim \pi_\theta(\cdot | x_{\text{corr}})$ are sampled; normalized advantages are computed within the fission group and used to update the policy via the GRPO objective.
    - **Design Motivation**: Diagnostic feedback increases outcome diversity within the group, alleviating the gradient vanishing problem caused by all-failure groups in standard GRPO.

3. **Time-Varying Composite Reward**:

    - **Function**: Guides the policy from format compliance toward semantic precision.
    - **Mechanism**: $R(\tau,t) = \frac{1}{3}[w_{\text{fmt}}(t)R_{\text{fmt}} + w_{\text{corr}}(t)R_{\text{corr}} + R_{\text{len}}]$, where the format weight decays and the correctness weight increases over training. The correctness reward jointly considers function selection accuracy and parameter F1.
    - **Design Motivation**: The model first learns formatting conventions in early training, then shifts focus to parameter precision in later stages.

### Loss & Training

Standard GRPO and fission-corrective GRPO are interleaved. A LIFO buffer ensures that corrective samples remain aligned with the current policy.

## Key Experimental Results

### Main Results

Results on BFCL v4 Multi-Turn with Qwen3-series models:

| Method | 1.7B | 4B | 8B |
|---|---|---|---|
| Base | 7.80 | 19.37 | 42.75 |
| GRPO | 17.12 | 36.38 | 42.75 |
| DAPO | 16.00 | 38.25 | — |
| **Fission-GRPO** | **20.38** | **40.50** | **46.75** |

Generalization results on TAU-Bench show up to +17.4% improvement on the Retail split.

### Ablation Study

| Configuration | Overall | Notes |
|---|---|---|
| GRPO only | 42.75 | No error recovery training |
| + Offline error data | 44.00 | Static distribution shift |
| + Fission (no simulator) | 44.50 | No diagnostic feedback |
| + Fission-GRPO | **46.75** | Full framework |

### Key Findings

- The error recovery rate improves by 5.7% (from ~20% to ~26%), which is the primary driver of the overall accuracy gain.
- The simulator achieves a non-leaking rate of 96% (human evaluation, Cohen's $\kappa = 0.71$) and generalizes to unseen tool schemas.
- The fission mechanism consistently improves performance across model scales (1.7B / 4B / 8B).

## Highlights & Insights

- The philosophy of **"errors as experience rather than punishment"** reshapes the training paradigm for RL-based tool use — rather than merely signaling that the model was wrong, the framework teaches it how to recover.
- The **LIFO buffer** ensures corrective samples are always aligned with the latest policy, avoiding the distribution shift inherent in offline data.
- The **fission metaphor** is intuitive and compelling: a single error → multiple recovery attempts → dense training signal.

## Limitations & Future Work

- The error simulator is based on Qwen3-32B, which is substantially larger than the target training models; deployment cost warrants consideration.
- Validation is limited to tool-calling scenarios; transferability to error recovery in reasoning or code generation remains to be explored.
- The LIFO buffer size and fission group size $G'$ require empirical tuning.

## Related Work & Insights

- **vs. DAPO / NGRPO**: These methods reshape the loss landscape for negative signals but do not construct positive signals; Fission-GRPO actively constructs recovery trajectories.
- **vs. ToolACE / LoopTool**: These approaches rely on offline-synthesized corrective data and suffer from severe distribution shift; Fission-GRPO generates corrective data online to maintain policy alignment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Integrating error recovery training into the RL loop is a highly novel and elegant idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple model scales, multiple benchmarks, and human evaluation of simulator reliability.
- Writing Quality: ⭐⭐⭐⭐ — Architecture diagrams are clear; the fission analogy is vivid.
- Value: ⭐⭐⭐⭐⭐ — Practically advances the robustness of tool-use agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/llm_alignment/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](../../ICLR2026/llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)
- [\[NeurIPS 2025\] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](../../NeurIPS2025/llm_alignment/deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](../../ICLR2026/llm_alignment/no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)

</div>

<!-- RELATED:END -->
