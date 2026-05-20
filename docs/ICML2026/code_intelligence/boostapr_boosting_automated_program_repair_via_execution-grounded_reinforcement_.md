---
title: >-
  [Paper Note] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models
description: >-
  [ICML 2026][Code Intelligence][Automated Program Repair] BoostAPR introduces a three-stage pipeline for "training program-repair models with RL": execution-verified SFT → training with both sequence-level and line-level…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Automated Program Repair"
  - "PPO"
  - "Dual Reward Models"
  - "Line-level Credit Assignment"
  - "SWE-bench"
date: 2026-05-08
content_hash: 4de398db5f58868d
---

# BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models

**Conference**: ICML 2026  
**arXiv**: [2605.09134](https://arxiv.org/abs/2605.09134)  
**Code**: https://github.com/yuanhao2023/BoostAPR  
**Area**: Code Intelligence / Automated Program Repair / Reinforcement Learning  
**Keywords**: Automated Program Repair, PPO, Dual Reward Models, Line-level Credit Assignment, SWE-bench

## TL;DR
BoostAPR introduces a three-stage pipeline for "training program-repair models with RL": execution-verified SFT → training with both sequence-level and line-level rewards → during PPO, redistributing sequence rewards to key edit lines using the line-level model. On Qwen2.5-Coder-32B, it boosts SWE-bench Verified from 17.8% to 40.7% (+22.9pp), and achieves 24.8% on Defects4J via cross-lingual transfer.

## Background & Motivation
**Background**: LLM-based Automated Program Repair (APR) has evolved from zero-shot prompting (e.g., GPT-4o + Agentless) to fine-tuning (SWE-Llama, Lingma-SWE-GPT, RepairLLaMA), and then to RL (SWE-RL achieves 41% on SWE-bench Verified, but with 70B parameters). Agentic systems (SWE-agent, AutoCodeRover) leverage tool use and fault localization for strong results.

**Limitations of Prior Work**: RL for APR faces three fundamental challenges: (1) **Execution feedback is extremely sparse**—a patch either passes all tests or none, providing only binary signals with no notion of "almost correct"; (2) **Sequence-level rewards cause severe credit assignment issues**—when a 50-line patch succeeds or fails, the model cannot tell which lines were critical and which were not, leading to high gradient variance; (3) **Distribution shift**—curated training data diverges from real-world bug patterns. Token-level reward models (Yoon 2024) are too fine-grained—single tokens lack semantics; process reward models (Lightman 2024) work for math reasoning but lack natural correspondence for code edits.

**Key Challenge**: To enable PPO to truly learn "which lines to fix," a reward signal is needed that is finer than sequence-level but more structured than token-level, and does not rely on counterfactual patch evaluation (too expensive) or ground-truth patch matching (no unique solution).

**Goal**: (i) Train a **line-level credit allocator** $R_{\text{line}}$ using execution feedback, capable of learning "which edit-line spans matter" without counterfactual evaluation; (ii) combine it with **sequence-level reward** $R_{\text{seq}}$ to produce token-level rewards for PPO; (iii) provide RL with a high-quality starting point via execution-verified SFT.

**Key Insight**: Parse unified diffs into maximal contiguous edit-line spans as "natural code modification units"—finer than hunks, more general than statements (parser-independent), and robust to malformed/cross-language diffs. Use stack traces to label spans on failing traceback paths as negatives, enabling execution-grounded contrastive supervision and avoiding costly counterfactual patch evaluation.

**Core Idea**: Dual reward = sequence-level (assessing overall patch quality) + line-level (learning edit-line importance); during PPO, redistribute $R_{\text{seq}}$'s total score to tokens in each edit-line span according to $R_{\text{line}}$'s softmax weights, achieving fine-grained credit redistribution.

## Method

### Overall Architecture
The three-stage pipeline is trained entirely in SWE-Gym. The base policy is Qwen2.5-Coder-32B-Instruct; both $R_{\text{seq}}$ and $R_{\text{line}}$ use Qwen2.5-Coder-7B-Instruct backbone with a scalar value head:

**Stage I (Execution-Verified Reasoning Transfer)**: Use Claude 3.5 Sonnet as teacher to generate (reasoning trace + final patch); run patches through all tests in SWE-Gym runner, **retaining only samples with resolved=True** (about 35% pass rate), then SFT for 3 epochs, lr 2e-5, batch 32.

**Stage II (Dual Reward Learning)**: For each instance, use SFT policy to nucleus sample $K=4$ candidates; compute execution reward $r^* = r_{\text{env}} + \gamma_{\text{diff}} r_{\text{diff}}$, where $r_{\text{env}} = w_{\text{apply}} r_{\text{apply}} + w_{\text{test}} r_{\text{test}}$, $r_{\text{diff}} = -\min(\eta |\Delta(y)|, r_{\max})$ penalizes large patches. Train $R_{\text{seq}}$ and $R_{\text{line}}$ separately.

**Stage III (Online PPO with Dual Rewards)**: Use VERL + vLLM; token reward $r_t = s \cdot a_t + \mathbb{I}[t=T] \cdot r_{\text{fmt}}(y)$, where $s = R_{\text{seq}}(y)$, $a_t$ is the normalized weight for the span containing token $t$ based on $R_{\text{line}}$'s span weights $w_\ell = \exp(s_\ell/\tau)/\sum_j \exp(s_j/\tau)$ ($\tau=0.5$), and $r_{\text{fmt}}$ applies a structure penalty to the final token (valid diff 0 / recoverable -0.4 / malformed -1.0 / not-a-diff -1.5). Use clipped PPO + GAE, $\epsilon=0.2$, adaptive KL target 0.1, LoRA rank 64, train for 300 steps.

### Key Designs

1. **Execution-verified SFT with Reasoning Trace ($\pi_0$)**:

    - **Function**: Ensures the base policy starts with "runnable patches + diagnostic reasoning," avoiding RL divergence from a weak policy.
    - **Mechanism**: Teacher output is forced to (reasoning trace, unified diff); only samples passing strict SWE-Gym test suite are retained (35% pass rate); next-token loss $\mathcal{L}_{\text{SFT}}=-\mathbb{E}_{(x,y)}[\sum_t \log \pi_\theta(y_t \mid x, y_{<t})]$ with prompt mask.
    - **Design Motivation**: Retaining traces allows the student to learn "how to diagnose bugs" rather than just "what patch to produce"; strict execution filtering avoids plausible-but-wrong demonstrations that may fool surface evaluation but are actually broken.

2. **Sequence-level $R_{\text{seq}}$ + Line-level $R_{\text{line}}$ Dual Reward Architecture**:

    - **Function**: $R_{\text{seq}}$ calibrates overall patch quality for PPO reward scaling; $R_{\text{line}}$ learns edit-line importance for fine-grained credit redistribution.
    - **Mechanism**: $R_{\text{seq}}$ is **patch-only scoring** (only unified diff, no bug context) to prevent shortcuts like "giving easy problems high scores"; trained with hybrid loss $\mathcal{L}_{\text{seq}} = \lambda_{\text{reg}} \mathbb{E}[(R_{\text{seq}}(y;\theta) - r^*(x,y))^2] + \mathbb{E}_{(y^+, y^-)}[-w \log \sigma(R_{\text{seq}}(y^+) - R_{\text{seq}}(y^-))]$, where regression ensures absolute score calibration and preference ensures correct relative ranking; $R_{\text{line}}$ parses unified diff into maximal contiguous edit-line spans (excluding header/context), encodes each span (edit content + context + file path + position), and outputs a scalar score.
    - **Design Motivation**: (i) Patch-only input is key for debiasing; (ii) Hybrid regression + preference—pure preference lacks absolute scale, destabilizing PPO advantage; pure regression overfits noisy execution labels; the combination is most robust; (iii) Line-span is an intermediate granularity, more semantic than token-level, more structured than sequence-level.

3. **Execution-grounded Stack-trace Supervision + Token-level Reward Redistribution**:

    - **Function**: Provides span-level supervision for $R_{\text{line}}$ without counterfactual patch evaluation; redistributes $R_{\text{seq}}$'s total score during PPO according to $R_{\text{line}}$'s weights.
    - **Mechanism**: For each candidate patch, use **priority cascade** to label spans based on execution results: (i) **Passing** patch—all edit spans labeled positive; (ii) **Failing** patch—priority 1: if a specific failing assertion is available from traceback, label the intersection of stack call chain and edit-line spans as negative (62%); priority 2: if traceback exists but no explicit assertion, downweight "edited functions" in traceback (27%); priority 3: if patch cannot be applied, use uniform fallback label (11%). Contrastive loss $\mathcal{L}_{\text{line}}=\mathbb{E}_{(\ell^+, \ell^-)}[-\log \sigma(R_{\text{line}}(\ell^+) - R_{\text{line}}(\ell^-))]$. During PPO, token reward $r_t = s \cdot a_t + \mathbb{I}[t=T] r_{\text{fmt}}$, where $a_t$ is the normalized span weight distributed to tokens within the span, **ensuring total reward = $s$** while redistributing according to learned edit importance.
    - **Design Motivation**: Stack-trace supervision is a cheap-and-grounded alternative to expensive leave-one-line-out counterfactual evaluation; priority cascade controls label noise; keeping total token reward unchanged ensures PPO advantage is not distorted; format penalty enforces valid unified diff output, otherwise reward is discounted.

### Loss & Training
- SFT: $\mathcal{L}_{\text{SFT}}$, 3 epochs, lr 2e-5, batch 32
- Reward: $\mathcal{L}_{\text{seq}}$ (hybrid regression + preference), $\mathcal{L}_{\text{line}}$ (contrastive), 5 epochs, lr 1e-5, batch 64
- PPO: clipped objective + GAE + adaptive KL (target 0.1), 300 steps, batch 64, rollouts/inst 4, LoRA rank 64
- Token reward formula $r_t = s \cdot a_t + \mathbb{I}[t=T] r_{\text{fmt}}$, format penalty $\in \{0, -0.4, -1.0, -1.5\}$

## Key Experimental Results

### Main Results
Evaluated with pass@1 (greedy), strict evaluation with no patch post-processing:

| Method | Backbone | SWE-V | D4J v2.0 | HE-Java | QuixBugs |
|--------|----------|-------|----------|---------|----------|
| Agentless | GPT-4o | 38.8 | 12.4* | 71.3* | 87.5* |
| SWE-agent | Claude 3.5 Sonnet | 33.6 | 10.8* | 68.9* | 85.0* |
| AutoCodeRover | GPT-4o | 28.8 | — | — | — |
| Qwen2.5-Coder-32B (base) | — | 17.8 | — | — | — |
| SWE-RL (70B) | — | 41.0 | — | — | — |
| **BoostAPR (Ours)** | Qwen2.5-Coder-32B | **40.7** (+22.9 vs base) | **24.8** | **84.5** | **95.0** |

Highlights: (1) Matches 70B SWE-RL with only 32B and single-machine training; (2) Pure Python training data, achieves 24.8% cross-lingual transfer to Defects4J (Java) with zero Java data; (3) Outperforms all agentic baselines on HumanEval-Java and QuixBugs, even though those use GPT-4o/Claude.

### Ablation Study
Stepwise breakdown on SWE-bench Verified (numbers based on original ablation section):

| Configuration | SWE-V Pass@1 | Notes |
|---------------|--------------|-------|
| Base (Qwen2.5-Coder-32B) | 17.8 | Starting point |
| + Stage I SFT (execution-verified) | ~30 | High-quality demonstrations are the main driver of improvement |
| + Stage II + Stage III ($R_{\text{seq}}$ only, sequence reward) | ~37 | PPO + $R_{\text{seq}}$ provides major accuracy gain |
| + $R_{\text{line}}$ (full BoostAPR) | **40.7** | Line-level credit provides complementary improvement |
| Full BoostAPR vs GRPO/rejection sampling RL | Significantly better | Dual reward outperforms common RL baselines |

| Key Variable | Phenomenon | Interpretation |
|--------------|------------|---------------|
| Patch-only $R_{\text{seq}}$ vs context-aware $R_{\text{seq}}$ | Patch-only slightly better | Prevents shortcuts, blocks "easy problems get high scores" bias |
| Hybrid (regression + preference) vs preference only | More stable convergence | Absolute scale calibrates PPO advantage |
| Stack-trace cascade vs uniform negative labels | Greater benefit | Fine-grained negative attribution matters |
| Format penalty | Enforces valid diff output | Prevents reward hacking with invalid formats |

### Key Findings
- **Stage I + $R_{\text{seq}}$ account for over 60% of total gain** (17.8 → ~37), serving as the main engine of the pipeline; $R_{\text{line}}$ adds ~4pp complementary improvement, and more importantly brings **out-of-distribution generalization** (Defects4J Python→Java) and **improved training efficiency + gradient quality**.
- **Impressive cross-lingual transfer**—pure Python training achieves 24.8% on Java benchmarks, indicating dual reward learns "general code modification signals" rather than language-specific patterns.
- **Patch-only $R_{\text{seq}}$ is key for shortcut prevention**—giving the reward model context allows it to learn "which problems are easy," a general lesson for reward shaping.
- **Line-span granularity** is more stable than token-level and finer than hunk-level—a pragmatic trade-off; the authors do not claim line-span is semantically optimal, only that it is practical.
- **Stack-trace supervision replaces counterfactual evaluation**—the latter requires leave-one-line-out execution for each candidate patch, which is computationally prohibitive; this work uses traceback paths + function-level fallback for cheap-and-grounded labeling.

## Highlights & Insights
- **Three-stage pipeline with clear boundaries**—SFT solves cold-start, dual reward solves credit assignment, PPO enables online improvement; each stage is independently ablatable and clearly engineered.
- **Edit-line span as intermediate granularity**—more semantic than token, more structured than sequence, more general than statement (parser-independent), robust to language and malformed diffs.
- **Token reward redistribution with constant sum**—$r_t = s \cdot a_t$ with $\sum_t a_t = 1$ ensures PPO advantage scale is unchanged, only the distribution shifts; this "reward redistribution rather than rescaling" design avoids unstable advantage scaling.
- **Hybrid regression-preference reward**—combines RLHF-style preference with RL-from-execution regression, more stable than pure DPO/PPO, and generalizable to any RL setting with both scalar and relative signals.
- **Execution-grounded supervision**—all labels are derived from actual execution, not human annotation, making it scalable and grounded; priority cascade fully exploits traceback information.

## Limitations & Future Work
- **Dependence on high-quality teacher models**—Stage I uses Claude 3.5 Sonnet for demonstrations; without a strong teacher, performance may degrade.
- **$R_{\text{line}}$ labels are noisy**—authors acknowledge ~11% uniform fallback and 27% function-level attribution are coarse; reducing label noise is future work.
- **PPO training is only 300 steps**—potential for further improvement or degradation with longer training is unexplored.
- **Line-span cannot handle "should-edit-but-missed" cases**—only scores actually edited lines, unable to address missing edits.
- **Format penalty is hard-coded** (0, -0.4, -1.0, -1.5); different tasks may require different scales, and learnable format rewards are a future direction.
- **No comparison with inference-time scaling (self-consistency/agentic search)**—unclear if BoostAPR's advantage holds under best-of-N or multi-turn agentic settings.
- **Reward hacking risk**—$R_{\text{seq}}$ may be fooled by superficial patterns (e.g., specific file names, diff lengths); no systematic analysis of hacking cases is provided.

## Related Work & Insights
- **vs SWE-RL (Wei et al. 2025)**: SWE-RL uses sequence-level reward + 70B parameters for 41% on SWE-V; BoostAPR achieves 40.7% with 32B + dual reward, halving parameters and improving interpretability.
- **vs CodeRL (Le et al. 2022)**: CodeRL is actor-critic + single execution reward; BoostAPR adds SFT warm-start and line-level credit allocation.
- **vs Token-level reward (Yoon et al. 2024)**: Token-level is too fine-grained for code; line-span is a more suitable intermediate granularity.
- **vs Process reward (Lightman 2024)**: PRM's "step" is natural for math reasoning but lacks code-edit correspondence; line-span is the code-specific equivalent.
- **vs RepairLLaMA / MORepair**: Pure SFT does not use execution feedback for RL; BoostAPR leverages all three stages.
- **Insights**: (i) "Decomposing tasks into cheap-and-grounded intermediate units (e.g., edit-line spans) for credit assignment" is a general RL-for-code strategy; (ii) Hybrid regression+preference rewards are more stable than pure preference, generalizable to any RLHF; (iii) Execution-grounded stack-trace supervision offers a cheap supervision paradigm for scenarios where counterfactual evaluation is infeasible, applicable to debugging, refactoring, test generation, etc.

## Rating
- Novelty: ⭐⭐⭐⭐ Line-level credit allocator is a genuine methodological innovation; stack-trace cascade supervision is also an ingenious engineering solution
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 benchmarks + cross-lingual + thorough ablation; lacks comparison with inference-time agentic search
- Writing Quality: ⭐⭐⭐⭐ Clear motivation; "why line-span" and "why hybrid reward" are thoroughly explained
- Value: ⭐⭐⭐⭐⭐ 32B model matches 70B SWE-RL + strong cross-lingual transfer, directly valuable for the open-source APR community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](../../ACL2026/code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](../../ACL2026/code_intelligence/socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](../../ACL2026/code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] The Path Not Taken: Duality in Reasoning about Program Execution](../../ACL2026/code_intelligence/the_path_not_taken_duality_in_reasoning_about_program_execution.md)

</div>

<!-- RELATED:END -->
