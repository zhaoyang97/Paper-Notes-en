---
title: >-
  [Paper Note] Reward Modeling from Natural Language Human Feedback
description: >-
  [ICML 2026][LLM Reasoning][Generative Reward Model (GRM)] This paper demonstrates that generative reward models (GRMs) trained on binary preference rewards suffer significantly from "outcome-process inconsistency" (corre…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Generative Reward Model (GRM)"
  - "Process Reward"
  - "Natural Language Feedback"
  - "MetaRM"
  - "GRPO"
date: 2026-05-08
content_hash: 035e0e4379fed6da
---

# Reward Modeling from Natural Language Human Feedback

**Conference**: ICML 2026  
**arXiv**: [2601.07349](https://arxiv.org/abs/2601.07349)  
**Code**: Not released  
**Area**: LLM Alignment / Reward Modeling / RLHF  
**Keywords**: Generative Reward Model (GRM), Process Reward, Natural Language Feedback, MetaRM, GRPO

## TL;DR
This paper demonstrates that generative reward models (GRMs) trained on binary preference rewards suffer significantly from "outcome-process inconsistency" (correct preference but incorrect critique), ranging from 20-30% up to 44%. The authors propose RM-NLHF, which utilizes the similarity between model critiques and human critique core arguments as an additional process reward. By using MetaRM to automatically predict these process rewards and updating it online alongside the policy, the method consistently outperforms SOTA GRMs trained via outcome-only GRPO across multiple benchmarks.

## Background & Motivation

**Background**: Generative Reward Models (GRMs) are mainstream in LLM alignment and RLHF because they output reasoning critiques alongside preference labels, offering better robustness and interpretability than traditional scalar RMs. Training typically relies on RLVR + GRPO, where the model generates reasoning and critiques for a pair of responses to produce an A/B label. The binary reward $R_{\text{outcome}}\in\{0,1\}$ is derived from whether the label matches the ground truth.

**Limitations of Prior Work**: Comparative experiments were conducted on MATH-500 (math, large solution space) and HelpSteer3 (pairwise rewards, binary solution space). In math tasks, a correct outcome almost always implies a correct process. However, in pairwise rewarding, RM-R1-DeepSeek-Distilled-Qwen-7B exhibits a 44.24% rate of "correct outcome / incorrect critique," while Gemini-2.5-Pro and Claude-3.7-Sonnet show 26.1% and 33.6% respectively. This phenomenon of "guessing the label without a correct critique" injects significant pseudo-rewards, causing the RL policy to converge toward generating flawed critiques.

**Key Challenge**: The size of the solution space determines the reliability of outcome supervision. In math, the answer space is vast (obtaining "42" almost necessitates correct reasoning), whereas binary preference tasks have a solution space of only {A, B}. Random guessing yields a 50% hit rate, making the outcome signal highly noisy. Furthermore, binary judgments cannot be easily converted into fill-in-the-blank formats to expand the solution space like math problems.

**Goal**: To provide a reliable process reward for GRMs without modifying the pairwise task structure, integrating critique quality directly into the training loop while overcoming the scalability bottleneck of scarce human critique data.

**Key Insight**: Natural language feedback (critiques) provided by humans naturally serves as process supervision. The overlap in core arguments between model critiques and human critiques serves as a direct proxy for critique validity. Additionally, a MetaRM can be trained to synthesize pseudo-critique data from limited human annotations.

**Core Idea**: The similarity between "GRM critiques and human core arguments" is used as a process reward, combined with the outcome reward for GRPO. MetaRM extrapolates this signal from small-scale human data to unlabeled data and is updated online during RL training to mitigate policy drift.

## Method

### Overall Architecture
The framework extends GRPO: given a query $q$ and candidates $y_A, y_B$ with preference label $l\in\{A,B\}$, the GRM $\pi_\theta$ generates a CoT, critique, and predicted label $\hat l$. For each prompt, $N$ rollouts yield an outcome reward $R_{\text{outcome}}^i$, used to calculate the advantage $\hat A_i$ via group normalization. RM-NLHF adds a process reward path: (1) when human critique $h$ is available, the similarity between core arguments of the GRM critique $\hat c$ and $h$ is computed; (2) when $h$ is absent, MetaRM predicts the reward; (3) MetaRM is updated online throughout training to match the current policy distribution. Finally, the advantage is determined by both outcome and process rewards.

### Key Designs

1. **Similarity w/ Core HC as Process Reward**:
    - **Function**: To compress the validity of a GRM critique into a machine-computable numerical reward while avoiding interference from nitpicky critiques.
    - **Mechanism**: An external strong LLM (Gemini-2.5-Pro) extracts core arguments from both human critique $h$ and GRM critique $\hat c$. F1, Recall, and Precision variants of similarity are then calculated. Tests on a 49-sample human-annotated subset showed that LLM-as-a-Meta-Judge was unstable, and All HC similarity was easily degraded by nitpicky points. Core HC similarity most closely aligned with human labels. The final process reward is $R_{\text{process}}=\text{sim}(\text{core}(h), \text{core}(\hat c))$.
    - **Design Motivation**: Direct LLM evaluation is prone to judge bias and stylistic preferences. Core argument overlap maintains semantic judgment while filtering out noise, proving quantitatively superior. This reward is compatible with the RLVR verifier framework.

2. **MetaRM: Predicting Process Rewards from Human Critiques**:
    - **Function**: To address the scalability bottleneck of scarce human critique data. Most preference datasets (e.g., UltraFeedback, HelpSteer) contain only outcome labels.
    - **Mechanism**: An auxiliary model, MetaRM, is trained on a subset with human critiques to map $(q, y_A, y_B, \hat c)$ to a process reward estimate (fitting the core similarity between $\hat c$ and $h$). At inference, it predicts rewards for data lacking human critiques.
    - **Design Motivation**: Human critique annotation is extremely costly. MetaRM distills the "critique evaluation capability" into a lightweight model, allowing it to generalize to large-scale datasets with only outcome labels.

3. **Online MetaRM: Synchronized Evolution with GRM**:
    - **Function**: To alleviate distribution mismatch caused by policy drift during RL training.
    - **Mechanism**: Training alternates between updating the GRM and MetaRM. After a GRM update step via GRPO, the current policy generates a batch of $\hat c$ on new prompts. These are paired with ground-truth $h$ (from the critique-labeled subset) to update MetaRM, which then provides rewards for the next GRM step.
    - **Design Motivation**: Static reward models often fail as the rollout distribution shifts (Reward Hacking). Online updates allow MetaRM to follow the policy, mitigating Goodhart’s Law issues.

### Loss & Training
The base is GRPO (Eq. 1-3) where the normalized advantage is $\hat A_i=(R_i-\bar R)/\sigma$. The policy is updated via clipped policy gradient with KL regularization. RM-NLHF replaces the reward with $R = R_{\text{outcome}} + \lambda \cdot R_{\text{process}}$, where the process reward originates from Core HC similarity or MetaRM. Online MetaRM is supervised via MSE or ranking loss and updated every $k$ GRPO steps. MetaRM and GRM can share a backbone with independent heads.

## Key Experimental Results

### Main Results
Comparisons were conducted on HelpSteer3, RewardBench, and PandaLM using base GRMs such as the RM-R1 series, Qwen, and closed-source models.

| Training Paradigm | Critique Quality (Core Argument F1) | Outcome Accuracy | Remarks |
| :--- | :--- | :--- | :--- |
| Outcome-only GRPO (SOTA baseline) | Low | High but 20–44% inconsistency | Standard approach |
| RM-NLHF + Full Human Critique | Highest | Significant improvement | Upper bound |
| RM-NLHF + Offline MetaRM | Near Full Human Critique | Significantly > Outcome-only | Annotation saving |
| **RM-NLHF + Online MetaRM** | Closest to Upper Bound | Significantly > Outcome-only | Practical optimal |

### Ablation Study (Process Reward Proxy, 49-sample subset)

| Process Reward Scheme | Alignment with Human Labels |
| :--- | :--- |
| LLM-as-a-Meta-Judge (Direct) | Low |
| Similarity w/ All HC (F1) | Medium |
| Similarity w/ All HC (Recall) | Medium-Low |
| Similarity w/ All HC (Precision) | Medium |
| **Similarity w/ Core HC** | Highest |

### Key Findings
- Math tasks show nearly 100% outcome-process correspondence, whereas pairwise tasks exhibit 20–44% inconsistency even in SOTA GRMs, proving outcome-only supervision is fundamentally unreliable for binary tasks.
- "Core HC similarity" consistently outperforms both "All HC" and "Direct LLM judgment," highlighting that removing nitpicky critiques is crucial for process reward design.
- Online MetaRM achieves results close to full human supervision while significantly reducing annotation requirements; offline MetaRM performs worse due to distribution shift.
- Even when outcome accuracy gains are modest, the significant boost in critique quality ensures the GRM provides better signals for downstream RLHF.

## Highlights & Insights
- **Clear Diagnosis of Inconsistency**: Explaining "why GRMs guess" through the theoretical lens of solution space size—large spaces provide implicit verification, while small spaces require explicit supervision.
- **"Core Argument Similarity" as Process Reward**: A key insight to avoid sensitivity to nitpicky details, applicable to LLM-as-a-judge and QA evaluation tasks.
- **Online MetaRM for Reward Drift**: Addresses the classic Goodhart problem via an actionable engineering protocol (alternating policy and MetaRM updates).
- **Extreme Data Efficiency**: Using a tiny 49-sample subset to validate proxies and limited subsets for MetaRM training makes this a model for cost-efficient alignment.

## Limitations & Future Work
- The assumption that "likelihood equals correctness" is not strictly true; models stylistically similar to human critiques may receive inflated rewards.
- Core HC extraction depends on a strong external LLM (Gemini-2.5-Pro), introducing cost and potential bias that MetaRM might amplify.
- Online MetaRM increases training complexity and wall-clock time; detailed efficiency analysis is missing.
- Verification is limited to pairwise tasks; the effect on listwise or scalar reward tasks remains unexamined.
- Lack of human evaluation for critique quality specifically compared against verifier-based RL (e.g., updated RM-R1 versions).

## Related Work & Insights
- **vs. Outcome-only GRPO GRMs (RM-R1, Wang 2025c)**: Directly uses these as baselines, quantifies their critique failure rates, and provides a dual-reward solution.
- **vs. PRM (Process Reward Model)**: While PRMs provide stepwise rewards for reasoning, this work provides critique-level rewards; both share the philosophy that process supervision is superior to outcome supervision.
- **vs. RLAIF / Constitutional AI**: While those use AI feedback, this method anchors on human critiques as ground truth before distilling to MetaRM, offering better interpretability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Originality in the "solution space" framework, Core HC similarity, and Online MetaRM, though components have separate precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple benchmarks and proxy comparisons; however, the 49-sample subset is small and human evaluation is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Intuitive motivation and clear contributions, though terminology is dense.
- **Value**: ⭐⭐⭐⭐ Adds essential process supervision to GRM training, easily transferable to existing RLHF/RLAIF pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GRPO is Secretly a Process Reward Model](grpo_is_secretly_a_process_reward_model.md)
- [\[ACL 2026\] C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences](../../ACL2026/llm_reasoning/c2_scalable_rubric-augmented_reward_modeling_from_binary_preferences.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](../../ICLR2026/llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)

</div>

<!-- RELATED:END -->
