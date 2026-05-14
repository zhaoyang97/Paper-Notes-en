---
title: >-
  [Paper Note] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
description: >-
  [ICLR 2026][Multimodal VLM][mllm-as-verifier] This paper identifies a pervasive "agreement bias" in multimodal large language models (MLLMs) when used as agent behavior verifiers—whereby models systematically over-approv…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "mllm-as-verifier"
  - "agreement-bias"
  - "self-grounded-verification"
  - "agent-evaluation"
  - "robotics"
date: 2026-05-08
content_hash: c057193ae0c31a25
---

# Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification

**Conference**: ICLR 2026
**arXiv**: [2507.11662](https://arxiv.org/abs/2507.11662)
**Code**: [Project Page](https://github.com/GT-RIPL/SGV)
**Area**: Multimodal VLM
**Keywords**: mllm-as-verifier, agreement-bias, self-grounded-verification, agent-evaluation, robotics

## TL;DR

This paper identifies a pervasive "agreement bias" in multimodal large language models (MLLMs) when used as agent behavior verifiers—whereby models systematically over-approve agent actions—and proposes Self-Grounded Verification (SGV), a two-step generation framework (first extracting behavioral priors, then performing conditioned verification) to mitigate this bias. SGV achieves up to 25 pp improvement in failure detection rate and 14 pp improvement in accuracy across web navigation, desktop manipulation, and robotic manipulation tasks.

## Background & Motivation

1. **Verifiers are a core engine of AI progress**: From Go to code reasoning, the search-plus-verifier paradigm has driven numerous breakthroughs. However, open-ended tasks (e.g., web navigation, robotic grasping) lack formal success criteria, making it difficult to build reliable automatic verifiers.

2. **MLLMs are expected to serve as general-purpose verifiers**: Leveraging broad world knowledge, human preference alignment, and multimodal reasoning, MLLMs are theoretically well-suited for scoring and judging agent trajectories, and have been applied in trajectory filtering, Reflexion-style self-improvement, and online supervision.

3. **Agreement bias is widespread and severe**: Across 13+ model families and 28+ evaluation templates, the authors find that MLLMs systematically assign high scores to agent behaviors—with true negative rate (TNR) as low as 50%, meaning half of all failed trajectories are incorrectly judged as successful. More concerning, MLLMs generate chain-of-thought reasoning to rationalize their erroneous judgments.

4. **Existing test-time scaling techniques fail to address the problem**: Methods such as CoT, SoM, majority voting, and reasoning models are all ineffective at mitigating the bias; sampling may even exacerbate hallucinations.

5. **The root cause lies in a knowledge extraction bottleneck**: MLLMs actually possess correct behavioral priors (they can describe correct behavior when given only partial information), but these priors are "overridden" when confronted with complete trajectories—consistent with known limitations of pretraining and RLHF.

6. **Downstream applications are severely degraded**: Agreement bias contaminates self-improvement pipelines (Reflexion), online supervisory feedback, and behavior cloning data filtering—any application relying on MLLM judgment—preventing agents from receiving corrective signals.

## Method

### Core Idea: Self-Grounded Verification (SGV)

SGV is a lightweight, zero-shot two-step verification framework. The central principle is to **first allow the MLLM to freely extract behavioral priors under information-constrained conditions, then use these priors as anchors for conditioned verification over the complete trajectory**.

### Step 1: Prior Extraction

Given task $q$ and partial trajectory information $\tau_{u:v}$ (e.g., an initial screenshot), the MLLM generates broad behavioral priors $\hat{k}$ regarding expected behavior:

$$\hat{k}_{q,u:v} = g\left(\prod_{i=1}^{n} P(y_i | y_{<i}, \tau_{u:v}, C, q)\right)$$

Since only partial information is available, the model can more freely explore its probability distribution and extract task-relevant knowledge without interference from the trajectory under evaluation.

### Step 2: Grounded Verification

Conditioned on the self-generated priors, the MLLM reasons over and evaluates the complete trajectory:

$$r_{\text{SGV}}(\tau_t, C, q) = h\left(\prod_{i=1}^{n} P(y_i | y_{<i}, q, \tau_{p:t}, C, \hat{k}_q)\right)$$

The priors serve as a reference baseline for an "impartial judge," preventing the model from blindly deferring to information presented in the trajectory.

### Key Design Choices

- **Scoring template**: A three-level Likert scale (SUCCESS / PARTIAL SUCCESS / FAILURE) is adopted; experiments confirm it reduces bias more effectively than binary judgment.
- **Trajectory representation**: Screenshot–action sequence pairs, optionally augmented with Set-of-Marks annotations for improved visual grounding.
- **Generality**: SGV can be applied on top of any MLLM without fine-tuning, is compatible with reasoning models, and can bring non-reasoning models to the level of reasoning models.

## Key Experimental Results

### Experimental Setup

- **Environments**: VisualWebArena (910 tasks, web navigation), OSWorld (369 tasks, desktop manipulation), robomimic (robotic manipulation, tool-hang task)
- **Models**: 14 models including GPT-5/o4, Gemini 2.5, Qwen3-235B, Llama-4, and others
- **Agents**: VWA uses a Gemini-2.5-Flash ReAct agent (SR=47%); OSWorld uses UI-TARS-1.5 (SR=22%); robomimic uses diffusion policy

### Table 1: Offline Verification Performance (VWA + OSWorld Combined)

| Model | Acc (w/o SGV) | TNR (w/o SGV) | Acc (SGV) | TNR (SGV) | Acc↑ | Bias↓ |
|-------|--------------|--------------|----------|----------|------|-------|
| GPT-5 (T) | 81 | 78 | 86 | 87 | +5 | -6 |
| GPT-o4 (T) | 78 | 71 | 84 | 82 | +6 | -6 |
| GPT-4.1 Mini | 60 | 40 | 74 | 65 | +14 | -20 |
| Gemini-2.5-Flash (T) | 74 | 64 | 82 | 78 | +8 | -15 |
| Qwen3-235b (T) | 66 | 53 | 77 | 71 | +11 | -12 |
| Llama-4-Maverick | 60 | 44 | 65 | 54 | +5 | -7 |

SGV consistently improves TNR (up to +25 pp) and accuracy (up to +14 pp) across all models, with weaker models benefiting the most.

### Table 2: Downstream Tasks—Online Supervision and Self-Improvement

| Method | VWA All | VWA S/C/R | OSWorld | robomimic SR |
|--------|---------|-----------|---------|-------------|
| Base Agent | 45 | 50/35/48 | 22 | 24 |
| + Verifier w/o SGV | 46 | 52/36/49 | 24 | 16 |
| + Verifier SGV | **54** | 56/43/58 | **27** | **32** |

SGV improves performance by 9 pp on VWA (+20% relative), 5 pp on OSWorld (+22%), and 8 pp on robomimic (+33%). VWA achieves a new state of the art, surpassing the previous best by 20 pp. Notably, the verifier without SGV degrades robomimic performance (24→16), underscoring the particularly harmful effect of agreement bias in robotic tasks.

## Highlights & Insights

- **Precise problem formulation**: This work is the first to systematically define and quantify agreement bias, validating its prevalence across 13+ model families and demonstrating its concrete harm to downstream applications.
- **Remarkably simple method**: SGV is a zero-shot, training-free two-step prompting approach that integrates seamlessly into existing pipelines.
- **Comprehensive evaluation**: Coverage spans offline evaluation and two downstream applications (self-improvement and online supervision), with fine-grained metrics beyond accuracy alone.
- **Non-trivial findings**: The paper reveals that reasoning models are equally susceptible to agreement bias, and SGV yields an additional 6–11 pp improvement even on top of reasoning models.

## Limitations & Future Work

- **Bias is mitigated, not eliminated**: SGV reduces but does not fully resolve agreement bias; residual failures are largely attributable to insufficient visual perception and language integration in the base model.
- **Increased computational overhead**: The two-step inference increases token consumption by a factor of 1.5–2.2×, requiring cost-benefit analysis at scale.
- **Prior quality is bounded by model capability**: Prior generation depends on the MLLM's own domain knowledge; if the model lacks sufficient expertise in the task domain, prior quality cannot be guaranteed.
- **Limited environment coverage**: Validation is confined to three environment types (web, desktop, robotics); more complex open-world scenarios (e.g., autonomous driving) remain unexplored.

## Related Work & Insights

### vs. Pan et al. (2024) — GPT-4V Evaluator

Pan et al. employ GPT-4V with benchmark-specific rubrics for binary judgment, a design adopted by numerous subsequent works. This paper shows that binary scoring amplifies agreement bias, and even manually crafted rubrics fail to resolve it (Tab. 3, row 6: Acc only 66%). SGV achieves higher accuracy (76+) without requiring manual rubrics, offering greater scalability.

### vs. Reasoning Models (DeepSeek-R1, GPT-o1/o4)

Reasoning models generate chain-of-thought via RL training and are theoretically better suited for verification. However, experiments show they remain susceptible to agreement bias (GPT-o1 TNR: 62%). SGV yields an additional 6–11 pp improvement on reasoning models, suggesting the root cause lies in the knowledge extraction bottleneck rather than reasoning capability per se.

### vs. Majority Voting / Tree Search

Majority voting relies on the mode of the output distribution, but agreement bias skews the distribution itself (failed trajectories yield correct judgments only 48% of the time), so voting cannot correct systematic bias. SGV fundamentally alters the conditional distribution rather than aggregating over a skewed one.

## Rating

- ⭐⭐⭐⭐⭐ Novelty: First to formally define agreement bias and propose a concise, effective solution.
- ⭐⭐⭐⭐⭐ Experimental Thoroughness: 14 models, 3 environments, 28+ templates, offline and downstream evaluation—exceptionally comprehensive.
- ⭐⭐⭐⭐ Writing Quality: Clear structure and rigorous argumentation, though heavy mathematical notation makes some passages dense.
- ⭐⭐⭐⭐ Value: SGV is plug-and-play and directly beneficial for agent systems, though token overhead warrants attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](../../CVPR2026/multimodal_vlm/self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)
- [\[CVPR 2026\] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](../../CVPR2026/multimodal_vlm/demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)
- [\[AAAI 2026\] SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias](../../AAAI2026/multimodal_vlm/sage_spuriousness-aware_guided_prompt_exploration_for_mitigating_multimodal_bias.md)
- [\[CVPR 2026\] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](../../CVPR2026/multimodal_vlm/mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)
- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](../../ACL2026/multimodal_vlm/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)

</div>

<!-- RELATED:END -->
