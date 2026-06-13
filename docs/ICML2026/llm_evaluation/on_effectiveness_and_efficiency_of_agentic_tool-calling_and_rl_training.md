---
title: >-
  [Paper Note] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training
description: >-
  [ICML 2026][LLM Evaluation][Tool calling] The authors systematically examine LLM tool calling through the lenses of "evaluation effectiveness" and "training efficiency." They use BFCL as a case study to demonstrate that…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Tool calling"
  - "BFCL"
  - "GRPO"
  - "Evaluation reproducibility"
  - "RL efficiency"
date: 2026-05-08
content_hash: 6fe385ccff7d658a
---

# On Effectiveness and Efficiency of Agentic Tool-calling and RL Training

**Conference**: ICML 2026  
**arXiv**: [2606.00135](https://arxiv.org/abs/2606.00135)  
**Code**: To be confirmed  
**Area**: LLM Agent / Tool Calling / Reinforcement Learning  
**Keywords**: Tool calling, BFCL, GRPO, Evaluation reproducibility, RL efficiency

## TL;DR
The authors systematically examine LLM tool calling through the lenses of "evaluation effectiveness" and "training efficiency." They use BFCL as a case study to demonstrate that "minor details" such as random seeds, multi-turn templates, thought history, and system prompts can cause significant drift in leaderboard scores, making cross-paper comparisons unreliable. For training, they identify waste in the rollout and policy update phases of RL (GRPO) and propose a dual-strategy of "online pre-rollout filtering + max-variance rollout sub-sampling," achieving 1.7× and 2.6× end-to-end acceleration for single-turn and multi-turn tool calling, respectively, without performance degradation.

## Background & Motivation

**Background**: Tool calling (function calling) has become a core capability of modern LLM agents. The community ranks models based on benchmarks like BFCL and Tau-bench, commonly employing post-training RL methods like PPO or GRPO to improve accuracy and robustness.

**Limitations of Prior Work**: Evaluation protocols are chaotic—the majority of tool-calling papers report scores using a single seed, proprietary multi-turn concatenation methods, and custom system prompts, yet compare absolute scores on leaderboards. Simultaneously, RL training is computationally expensive; for multi-turn tool calling, context lengths (including tool schemas, dialogue history, and tool I/O) are extremely long, causing the wall-clock time for the policy update phase to be 3–5× higher than the rollout phase.

**Key Challenge**: Parallel neglect of "implicit degrees of freedom" in evaluation and "silent waste" in training exists. The former masks the true progress of methods within evaluation noise, while the latter spends massive compute on samples with zero gradients. Combined, a "5-point improvement" might stem from a superior method, or simply a prompt change or favorable seeds, making it difficult for the community to identify genuinely valuable directions.

**Goal**: (1) Systematically quantify the sensitivity of tool-calling evaluations (e.g., BFCL) to implementation details and provide minimum standards for reproducibility; (2) Locate specific sources of waste in RL tool-calling training and provide non-intrusive acceleration schemes.

**Key Insight**: Regarding evaluation, the authors treat every undocumented choice in the pipeline as an independent variable for controlled experiments. Regarding training, the authors dissect the rollout reward variance and wall-clock time distribution of GRPO, discovering that "zero-variance prompts" account for up to 80% of samples and exhibit significant temporal stability—an empirical observation that directly leads to the online filtering strategy.

**Core Idea**: Progress in tool calling must be built upon controlled evaluation; RL compute should prioritize prompts where the model can still learn and rollouts with the strongest reward signals.

## Method

### Overall Architecture
The paper follows two parallel threads:

- **Effectiveness Thread**: A sensitivity analysis of 5 common models (Qwen3-4B/8B, Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, Llama3.2-3B-Instruct) on BFCL across 5 categories (random seed, native vs. context multi-turn templates, inclusion of thought history, system prompts, and single/multi-turn data formats). 
- **Efficiency Thread**: GRPO training using the VERL framework for Qwen2.5-3B-Instruct (single-turn) and Qwen3-4B (multi-turn). The work analyzes the computational distribution of rollout and policy update phases and proposes two acceleration techniques integrated into standard GRPO.

The GRPO framework follows the standard definition: sample $n$ rollouts $\{y_{i,k+1,j}\}$ for a sample $s_{i,k}$, perform intra-group normalization of rewards $A_{i,k+1,j}=(r_{i,k+1,j}-\bar r_{i,k+1})/\sigma_{i,k+1}$, and enter the clipped objective with the importance ratio $\rho_{i,k+1,j}=\pi_\theta/\pi_{\text{old}}$. When all rollouts for a prompt yield identical rewards, $A=0$, which is termed a **zero-variance prompt**.

### Key Designs

1. **Evaluation Sensitivity Diagnostic Protocol**:
    - **Function**: Extracts implementation options usually ignored in tool-calling evaluations and subjects them to controlled experiments to quantify score drift.
    - **Mechanism**: On BFCL, a single model is fixed while one variable is modified at a time: (a) 10 seeds are run to check variance; single-turn is stable, but multi-turn fluctuates significantly. (b) Native templates (turn-by-turn concatenation) are compared with context templates (packing the history into one user turn); native templates consistently lead by 6–8%. (c) Retaining `<think>` segments improves Qwen3 multi-turn scores by 3–5%. (d) Manual addition of specific multi-turn instructions to the system prompt provides gains comparable to RL training. (e) Testing pure single-turn vs. pure multi-turn training (0.7k budget) reveals that pure multi-turn training surprisingly drops multi-turn BFCL scores from 22.7 to 15.9.
    - **Design Motivation**: To reveal that "leaderboard numbers $\neq$ model capability." If a prompt adjustment creates gains equivalent to RL, leaderboard comparisons without reported prompts or templates are meaningless.

2. **Online Pre-rollout Filtering**:
    - **Function**: Skips prompts that have been consistently answered correctly in recent epochs before the expensive rollout generation, focusing compute on samples with learning signals.
    - **Mechanism**: The authors observe that zero-variance prompts initially comprise ~80% of samples and are temporally stable—the conditional probability $P(\text{remains correct} \mid \text{correct for } k \text{ consecutive epochs})$ is $>0.8$ for single-turn and $>0.9$ for multi-turn when $k=1$. A per-prompt "win streak counter" $c_{i,k+1}^{(e)}$ is maintained. If a prompt is entirely correct in an epoch, $c$ increments; otherwise, it resets. If $c \ge k$, the prompt is temporarily excluded: $\mathcal{D}^{(e)}=\{s : c_{i,k+1}^{(e)} < k\}$.
    - **Design Motivation**: Zero-variance prompts in GRPO yield zero gradients; rolling them out is a waste of compute. Static pre-filtering is ineffective as prompt difficulty shifts with policy evolution.

3. **Max-variance Rollout Sub-sampling**:
    - **Function**: While GRPO still generates $n$ rollouts to maintain a stable baseline, only $m < n$ rollouts are used for backpropagation, reducing policy update computation by approximately $n/m$.
    - **Mechanism**: Dissections of wall-clock time in VERL show that tool-calling policy update time grows much faster than rollout time due to long sequences (schemas, dialogue context). Following Xu et al. (2025), $n$ rollouts are sorted by reward, and a subset $\mathcal{S}^*$ is formed from the $m'$ lowest and $m-m'$ highest reward samples to maximize intra-group variance. 
    - **Design Motivation**: Rollout requires a large $n$ for advantage estimation, but the update phase is bottlenecked by sequence length. "Generating many, updating few" redistributes compute to bypass the update bottleneck in long-context scenarios.

### Loss & Training
- **RL Algorithm**: GRPO, using standard clipped objectives with the VERL default $\epsilon$.
- **Framework**: VERL; Qwen2.5-3B-Instruct for single-turn, Qwen3-4B for multi-turn. Data sizes: 2.3k (single) and 2.6k-6k (multi).
- **Online Filtering Hyperparameters**: $k=1$ or $2$, updated rolling per epoch.
- **Sub-sampling Hyperparameters**: $m$ samples selected from $n$ rollouts (e.g., $n=8, m=4$), picking from both ends of the reward distribution.
- **Evaluation**: BFCL using 3-seed averages; user simulator uses Claude 4 with additional identity constraints. ACEBench reports overall accuracy across English categories.

## Key Experimental Results

### Main Results

Comparison with representative open/closed-source models on BFCL (Qwen3-4B baseline + strengthened system prompt + RL, 3-seed average):

| Model | Multi-turn | Single-turn | Avg. |
|------|-----------:|------------:|-----:|
| Claude Sonnet 4.5 (FC) | 61.4 | 84.9 | 73.2 |
| Gemini-3-Pro-Preview (FC) | 63.1 | 83.8 | 73.4 |
| GPT-4.1-2025-04-14 (FC) | 38.9 | 76.4 | 57.7 |
| Qwen3-235B-A22B-Instruct-2507 (FC) | 45.4 | 53.2 | 49.3 |
| Qwen3-4B w. BFCL default prompt | 22.7±0.9 | 83.9±0.5 | 53.3±0.5 |
| Qwen3-4B w. stronger prompt | 37.2±1.4 | 84.8±0.7 | 61.0±0.8 |
| **Qwen3-4B-RL (Ours)** | **39.4±0.7** | **84.8±0.9** | **62.1±0.5** |

ACEBench (English overall accuracy): The proposed RL training improves Qwen3-4B from 65.4 to 77.5 (+12.1), surpassing Nova-1-Lite (73.4) and several 24B+ open-source models.

### Ablation Study

| Configuration | BFCL Multi-turn | Notes |
|------|----------------:|------|
| Qwen3-4B base | 22.7 ±0.9 | Default prompt + native template |
| → Context template | ↓ ~6–8% | Same info, different concatenation |
| → Drop `<think>` history | ↓ ~3–5% | Significant impact on multi-turn consistency |
| → Stronger system prompt | 37.2 ±1.4 | Prompt-only change; gain matches RL |
| Single-turn training (0.7k) | 20.2 ±0.6 | Multi-turn flat; single-turn slight rise |
| Multi-turn training (0.7k) | 15.9 ±0.4 | Multi-turn drops; suggests high supervision noise |

Efficiency Ablation: For the same wall-clock budget, vanilla GRPO vs. GRPO + dual acceleration—1.7× speedup for single-turn and 2.6× for multi-turn to reach equivalent performance. No degradation found in downstream general tasks (e.g., MMLU 68.3→68.3).

### Key Findings
- **Evaluation fragility matches method gains**: Multi-turn gains from system prompt rewriting can match or exceed gains from RL training.
- **"Multi-turn data is always better" is a myth**: In controlled data experiments, pure multi-turn training reduced BFCL multi-turn scores. The authors hypothesize that multi-turn trajectories contain accumulated errors and ambiguous labels.
- **Zero-variance prompts dominate**: Initially, ~80% of prompts generate no gradient. High temporal stability allows short-window filtering to save rollout compute without harming the learning signal.
- **Compute bottleneck shifts to Update**: Unlike math reasoning, policy updates dominate wall-clock time in tool calling even at $n=4$ due to the long context of tool schemas and histories.

## Highlights & Insights
- **Quantifying evaluation "degrees of freedom"**: Treating undocumented engineering choices as research subjects provides a necessary critique of current tool-calling leaderboards.
- **Diagnosis-driven design**: The acceleration methods are direct responses to observed bottlenecks (80% zero-gradient + update bottleneck), resulting in simple yet effective solutions (counting streaks + sorting ends).
- **"Max-variance sub-sampling" is a universal trick**: For any group-based advantage RL (GRPO family), this strategy effectively reallocates compute when update costs dominate.
- **Evidence against noisy multi-turn supervision**: The study suggests focusing on trajectory quality over quantity, as multi-turn noise can be detrimental.

## Limitations & Future Work
- Evaluation is primarily focused on BFCL and ACEBench; validation on more complex scenarios like Tau-bench or Web agents is needed.
- Scalability has been tested only at 3B and 4B parameters; while Appendix G suggests similar trends for larger models, full end-to-end curves are missing.
- Online filtering focuses on "all-correct" samples; the treatment of "all-wrong" samples (also zero-variance but potentially valuable for curriculum learning) requires further study.
- The user simulator used (Claude 4) may introduce simulator bias.

## Related Work & Insights
- **vs. ToolRL / Tool-N1**: While prior work focuses on "what data/rewards to use," this paper orthogonally investigates training waste and evaluation reliability. It highlights that reported gains in prior work could be confounded by prompt/template drift.
- **vs. Hochlehnert et al. 2025**: Similar concerns regarding reproducibility in math reasoning are systemically ported to tool calling, adding multi-turn-specific factors.
- **vs. Xu et al. 2025**: This work adopts max-variance sub-sampling and proves it particularly effective for tool calling due to the dominance of the update phase.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematic diagnosis of evaluation in tool calling is a first; the training "twin-pack" is an excellent application of engineering insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-seed, multi-model analysis across multiple benchmarks; missing larger scale verification.
- **Writing Quality**: ⭐⭐⭐⭐ Clear dual-axis structure; figures directly support the arguments.
- **Value**: ⭐⭐⭐⭐⭐ High. The reproducibility guidelines should be adopted by the community, and the acceleration schemes are immediately applicable for any team training agents with GRPO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Toward Training Superintelligent Software Agents through Self-Play SWE-RL](toward_training_superintelligent_software_agents_through_self-play_swe-rl.md)
- [\[ACL 2026\] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](../../ACL2026/llm_evaluation/rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)
- [\[ICML 2026\] REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation](real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)
- [\[ICML 2026\] Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](beyond_trajectory-level_attribution_graph-based_credit_assignment_for_agentic_re.md)
- [\[ICML 2026\] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)

</div>

<!-- RELATED:END -->
