---
title: >-
  [Paper Note] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models
description: >-
  [ICML 2026][Code Intelligence][Automated Program Repair] BoostAPR establishes a three-stage pipeline for training program repair models with RL: execution-verified SFT $\rightarrow$ training sequence-level + line-level d…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Automated Program Repair"
  - "PPO"
  - "Dual Reward Models"
  - "Line-level Credit Assignment"
  - "SWE-bench"
date: 2026-05-08
content_hash: daf311fb3a735f84
---

# BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models

**Conference**: ICML 2026  
**arXiv**: [2605.09134](https://arxiv.org/abs/2605.09134)  
**Code**: https://github.com/yuanhao2023/BoostAPR  
**Area**: Code Intelligence / Automated Program Repair / Reinforcement Learning  
**Keywords**: Automated Program Repair, PPO, Dual Reward Models, Line-level Credit Assignment, SWE-bench

## TL;DR
BoostAPR establishes a three-stage pipeline for training program repair models with RL: execution-verified SFT $\rightarrow$ training sequence-level + line-level dual reward models $\rightarrow$ using the line-level model during PPO to redistribute sequence rewards to critical edit line spans. Using Qwen2.5-Coder-32B, it pushes SWE-bench Verified from 17.8% up to 40.7% (+22.9pp) and achieves 24.8% on Defects4J via cross-lingual transfer to Java.

## Background & Motivation
**Background**: LLM-based Automated Program Repair (APR) has evolved from zero-shot prompting (e.g., GPT-4o + Agentless) to fine-tuning (SWE-Llama, Lingma-SWE-GPT, RepairLLaMA) and recently to RL (SWE-RL reached 41% on SWE-bench Verified but used 70B parameters). Agentic systems (SWE-agent, AutoCodeRover) achieve strong results via tool use and fault localization.

**Limitations of Prior Work**: Training APR with RL faces three fundamental difficulties: (1) **extremely sparse execution feedback**—a patch either passes all tests or fails, offering no "near-miss" signal; (2) **severe credit assignment issues** with sequence-level rewards—when a 50-line patch succeeds or fails, the model cannot distinguish which lines were critical vs. decorative, leading to high gradient variance; (3) **distribution shift** between curated training data and real-world repository bug patterns. Token-level reward models are too fine-grained (lacking semantics), while process reward models (PRM) lack a natural "step" definition in the context of code editing.

**Key Challenge**: To enable PPO to effectively learn "which lines to fix," reward signals must be finer than the sequence level but more structured than the token level. Moreover, these signals must not rely on expensive counterfactual patch evaluations or the existence of a unique ground-truth patch.

**Goal**: (i) Train a **line-level credit allocator** $R_{\text{line}}$ using execution feedback to learn the importance of edit-line spans without counterfactuals; (ii) Combine it with a **sequence-level reward** $R_{\text{seq}}$ for PPO; (iii) Provide a high-quality starting point via execution-verified SFT.

**Key Insight**: Parsing unified diffs into maximal contiguous edit-line spans serves as a natural unit of code modification—finer than hunks, more general than statements (language-independent), and robust to malformed diffs. Using stack traces to label spans on the failing traceback path as negative samples provides execution-grounded contrastive supervision without expensive counterfactuals.

**Core Idea**: Dual reward system = sequence-level (evaluating overall patch quality) + line-level (learning edit-line importance). During PPO, the total score from $R_{\text{seq}}$ is redistributed to tokens within each edit-line span based on softmax weights from $R_{\text{line}}$, achieving fine-grained credit redistribution.

## Method

### Overall Architecture
The three-stage pipeline is trained entirely on SWE-Gym. The base policy is Qwen2.5-Coder-32B-Instruct. Both $R_{\text{seq}}$ and $R_{\text{line}}$ use a Qwen2.5-Coder-7B-Instruct backbone with a scalar value head:

**Stage I (Execution-Verified Reasoning Transfer)**: Claude 3.5 Sonnet acts as a teacher to generate (reasoning trace + final patch). Patches are executed on the SWE-Gym runner; **only samples with resolved=True are retained** (approx. 35% pass rate). SFT is performed for 3 epochs (lr 2e-5, batch 32).

**Stage II (Dual Reward Learning)**: Each instance is nucleus sampled $K=4$ times using the SFT policy. After execution, the reward $r^* = r_{\text{env}} + \gamma_{\text{diff}} r_{\text{diff}}$ is calculated, where $r_{\text{env}} = w_{\text{apply}} r_{\text{apply}} + w_{\text{test}} r_{\text{test}}$ and $r_{\text{diff}} = -\min(\eta |\Delta(y)|, r_{\max})$ penalizes excessively large patches. $R_{\text{seq}}$ and $R_{\text{line}}$ are trained respectively.

**Stage III (Online PPO with Dual Rewards)**: Using VERL + vLLM, the token reward is defined as $r_t = s \cdot a_t + \mathbb{I}[t=T] \cdot r_{\text{fmt}}(y)$, where $s = R_{\text{seq}}(y)$, and $a_t$ is the normalized weight from $R_{\text{line}}$ ($w_\ell = \exp(s_\ell/\tau)/\sum_j \exp(s_j/\tau)$ with $\tau=0.5$) distributed across tokens in a span. $r_{\text{fmt}}$ applies a structural penalty to the final token (valid diff 0 / recoverable -0.4 / malformed -1.0 / not-a-diff -1.5). Clipped PPO + GAE is used with $\epsilon=0.2$ and adaptive KL target of 0.1 for 300 steps.

### Key Designs

1.  **Execution-Verified + Reasoning Trace SFT Initialization ($\pi_0$)**:
    *   **Function**: Ensures the base policy starts with "runnable patches + diagnostic reasoning," preventing RL divergence from weak policies.
    *   **Mechanism**: Teacher output format is forced to (reasoning trace, unified diff); only samples passing the strict SWE-Gym test suite are kept. Optimization uses next-token loss $\mathcal{L}_{\text{SFT}}=-\mathbb{E}_{(x,y)}[\sum_t \log \pi_\theta(y_t \mid x, y_{<t})]$ with prompt masking.
    *   **Design Motivation**: Reasoning traces help the student learn "how to diagnose" rather than just "what patch to produce." Strict filtering prevents contamination from plausible-but-wrong demonstrations.

2.  **Dual Reward Architecture ($R_{\text{seq}}$ + $R_{\text{line}}$)**:
    *   **Function**: $R_{\text{seq}}$ evaluates overall quality to calibrate PPO reward scale; $R_{\text{line}}$ learns edit-line importance for redistribution.
    *   **Mechanism**: $R_{\text{seq}}$ employs **patch-only scoring** (ignoring bug context) to prevent learning shortcuts like "high scores for easy problems." It uses a hybrid loss $\mathcal{L}_{\text{seq}} = \lambda_{\text{reg}} \mathbb{E}[(R_{\text{seq}}(y;\theta) - r^*(x,y))^2] + \mathbb{E}_{(y^+, y^-)}[-w \log \sigma(R_{\text{seq}}(y^+) - R_{\text{seq}}(y^-))]$. $R_{\text{line}}$ encodes edit spans (content + context + file path + location) for scoring.
    *   **Design Motivation**: Patch-only input is a critical de-biasing design. The hybrid regression + preference objective balances absolute scale calibration with relative ranking. Line-spans offer a "just right" granularity.

3.  **Execution-Grounded Stack-Trace Supervision + Token-level Redistribution**:
    *   **Function**: Provides span-level supervision for $R_{\text{line}}$ without counterfactuals; redistributes sequence rewards during PPO.
    *   **Mechanism**: A **priority cascade** labels spans: (i) **Passing patches**—all spans marked positive; (ii) **Failing patches**—Priority 1: if a failing assertion is identified via traceback, the intersection of stack call chains and edit spans is marked negative (62%); Priority 2: failing edited functions in the traceback are penalized (27%); Priority 3: fallback uniform label for patches that fail to apply (11%). Contrastive loss $\mathcal{L}_{\text{line}}=\mathbb{E}_{(\ell^+, \ell^-)}[-\log \sigma(R_{\text{line}}(\ell^+) - R_{\text{line}}(\ell^-))]$ is used. Reward $r_t$ ensures $\sum r_t \approx s$.
    *   **Design Motivation**: Stack-trace supervision is a "cheap-and-grounded" alternative to leave-one-line-out evaluation. Maintaining the total reward sum ensures PPO advantage scale stability.

### Loss & Training
*   **SFT**: $\mathcal{L}_{\text{SFT}}$, 3 epochs, lr 2e-5, batch 32.
*   **Reward**: $\mathcal{L}_{\text{seq}}$ (hybrid) and $\mathcal{L}_{\text{line}}$ (contrastive), 5 epochs, lr 1e-5, batch 64.
*   **PPO**: Clipped objective + GAE + adaptive KL (target 0.1), 300 steps, batch 64, rollouts/inst 4, LoRA rank 64.
*   Token reward formula: $r_t = s \cdot a_t + \mathbb{I}[t=T] r_{\text{fmt}}$, focusing on redistribution.

## Key Experimental Results

### Main Results
Pass@1 (greedy) results using strict evaluation without patch post-processing:

| Method | Backbone | SWE-V | D4J v2.0 | HE-Java | QuixBugs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Agentless | GPT-4o | 38.8 | 12.4* | 71.3* | 87.5* |
| SWE-agent | Claude 3.5 Sonnet | 33.6 | 10.8* | 68.9* | 85.0* |
| AutoCodeRover | GPT-4o | 28.8 | — | — | — |
| Qwen2.5-Coder-32B (base) | — | 17.8 | — | — | — |
| SWE-RL (70B) | — | 41.0 | — | — | — |
| **BoostAPR (Ours)** | Qwen2.5-Coder-32B | **40.7** | **24.8** | **84.5** | **95.0** |

**Highlights**: (1) Matches 70B SWE-RL performance at a 32B scale with single-machine training; (2) Zero-shot cross-lingual transfer to Java (Defects4J) at 24.8% despite being trained only on Python; (3) Outperforms all agentic baselines on HumanEval-Java and QuixBugs.

### Ablation Study
Breakdown on SWE-bench Verified:

| Config | SWE-V Pass@1 | Description |
| :--- | :--- | :--- |
| Base (Qwen2.5-Coder-32B) | 17.8 | Starting point |
| + Stage I SFT (execution-verified) | ~30 | High-quality demos are the primary driver |
| + Stage II + Stage III ($R_{\text{seq}}$ only) | ~37 | PPO + $R_{\text{seq}}$ provides main accuracy gain |
| + $R_{\text{line}}$ (Full BoostAPR) | **40.7** | Line-level credit provides complementary gain |

| Key Insight | Observation | Interpretation |
| :--- | :--- | :--- |
| Patch-only $R_{\text{seq}}$ vs Context-aware | Patch-only is better | Prevents shortcutting by "difficulty" |
| Hybrid vs Preference-only | Better stability | Absolute scale calibrates PPO advantage |
| Stack-trace cascade | More gain than uniform | Precise negative attribution is critical |

### Key Findings
*   **Stage I + $R_{\text{seq}}$ account for over 60% of total gains** (17.8 → ~37); $R_{\text{line}}$ provides a ~4pp complement and significantly improves out-of-distribution generalization.
*   **Cross-lingual transfer is remarkable**: Python-only training reaching 24.8% in Java suggests the dual reward learns "generalized modification importance."
*   **Patch-only scoring is essential**: Contextual info allows the RM to estimate problem difficulty rather than patch quality.
*   **Intermediate Granularity**: Line-spans are more stable than tokens and more structured than sequences.

## Highlights & Insights
*   **Clear Stage Boundaries**: SFT (cold-start), Dual Reward (credit assignment), PPO (online improvement) are modular and independently glass-boxed.
*   **Intermediate Unit Design**: Edit-line spans are language-agnostic and robust to malformed outputs.
*   **Reward Redistribution vs Scaling**: Using $r_t = s \cdot a_t$ with $\sum a_t = 1$ shifts the reward distribution without distorting the overall advantage scale.
*   **Stack-trace Supervision**: A novel paradigm for "cheap-and-grounded" feedback in environments where counterfactual testing is too expensive.

## Limitations & Future Work
*   **Teacher Model Dependency**: Relies on Claude 3.5 Sonnet for Stage I demonstrations.
*   **Label Noise**: Traceback-based attribution is still heuristic and suffers from ~11% uniform fallback noise.
*   **Short PPO Training**: 300 steps might not fully explore the potential for long-term improvement.
*   **Missing Edits**: The current line-span approach cannot reward locations where an edit *should* have occurred but was missing.
*   **Inference-time Scaling**: No comparison yet with best-of-N or multi-turn agentic search at inference time.

## Related Work & Insights
*   **vs SWE-RL (2025)**: BoostAPR achieves similar results to a 70B model using only 32B and a more interpretable credit assignment method.
*   **vs Process Reward (Lightman 2024)**: While PRM focuses on logical steps in math, BoostAPR adapts this "intermediate signal" concept to code modifications via line-spans.
*   **Insight**: Designing "intermediate units" for credit assignment is a universal strategy for RL-for-code. Hybrid regression+preference objectives are more stable for RLHF-derived signal training.

## Rating
*   Novelty: ⭐⭐⭐⭐ (Line-level allocator and stack-trace cascade are genuine innovations)
*   Experimental Thoroughness: ⭐⭐⭐⭐ (Cross-lingual and multi-benchmark, though missing inference search comparisons)
*   Writing Quality: ⭐⭐⭐⭐ (Coherent motivation for line-level design)
*   Value: ⭐⭐⭐⭐⭐ (High utility for the open-source APR community with 32B models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](../../ACL2026/code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](../../ACL2026/code_intelligence/duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](../../ACL2026/code_intelligence/socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](../../ACL2026/code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](../../ACL2026/code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](../../ACL2026/code_intelligence/duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](../../ACL2026/code_intelligence/socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](../../ACL2026/code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)

</div>

<!-- RELATED:END -->
